"""
Module implementing the ImaginedBehavior.
Generates imagined trajectories and trains the actor and critic networks.
Uses a proper rollout: the decoder predicts the next observation which is then encoded.
Also applies RewardObjective and ensures that actor gradients do not update the world model.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
from utils.helper_functions import (
    imagine_trajectory, 
    lambda_return_target, 
    compute_actor_loss, 
    tensor_stats,
    RewardObjective,
    discretize_symlog
)
from utils.optimizer import Optimizer
from agent.networks import MLP
import copy

class ImaginedBehavior(nn.Module):
    def __init__(self, configuration: Any, world_model: Any) -> None:
        super(ImaginedBehavior, self).__init__()
        self.use_amp = (configuration.precision == 16)
        self.configuration = configuration
        self.world_model = world_model

        if configuration.dynamics_use_discrete:
            feature_dimension = (configuration.discrete_latent_num *
                               configuration.discrete_latent_size +
                               configuration.dynamics_deterministic_dimension)
        else:
            feature_dimension = (configuration.dynamics_stochastic_dimension +
                               configuration.dynamics_deterministic_dimension)

        self.actor = MLP(
            input_dim=feature_dimension,
            output_shape=(configuration.number_of_possible_actions,),
            layers=configuration.actor["number_of_layers"],
            units=configuration.units,
            activation=configuration.activation_function,
            normalization=configuration.normalization_type,
            dist=configuration.actor["distribution_type"],
            std=configuration.actor["standard_deviation"],
            min_std=configuration.actor["minimum_standard_deviation"],
            max_std=configuration.actor["maximum_standard_deviation"],
            absmax=1.0,
            temperature=configuration.actor["temperature"],
            unimix_ratio=configuration.actor["unimix_ratio"],
            outscale=configuration.actor["output_scale"],
            device=configuration.computation_device,
            name="Actor",
            use_orthogonal=configuration.use_orthogonal_initialization
        )
        self.value = MLP(
            input_dim=feature_dimension,
            output_shape=(255,) if configuration.critic["distribution_type"] == "symlog_disc" else (),
            layers=configuration.critic["number_of_layers"],
            units=configuration.units,
            activation=configuration.activation_function,
            normalization=configuration.normalization_type,
            dist=configuration.critic["distribution_type"],
            outscale=configuration.critic["output_scale"],
            device=configuration.computation_device,
            name="Value",
            use_orthogonal=configuration.use_orthogonal_initialization
        )
        if configuration.critic["use_slow_target"]:
            self.slow_value = copy.deepcopy(self.value)
            self.update_counter = 0
        else:
            self.slow_value = None

        optimizer_args = dict(weight_decay=configuration.weight_decay_value,
                            opt=configuration.optimizer_type,
                            use_amp=self.use_amp)
        self.actor_optimizer = Optimizer("actor", self.actor.parameters(),
                                       learning_rate=configuration.actor["lr"],
                                       eps=configuration.actor["eps"],
                                       clip=configuration.actor["grad_clip"],
                                       **optimizer_args)
        self.value_optimizer = Optimizer("value", self.value.parameters(),
                                       learning_rate=configuration.critic["lr"],
                                       eps=configuration.critic["eps"],
                                       clip=configuration.critic["grad_clip"],
                                       **optimizer_args)
        self.reward_objective = RewardObjective(clip_value=5.0, alpha=0.01)
        
    def train_step(self, starting_state: Any, objective_function: Any = None) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, Dict[str, float]]:
        self._update_slow_target()
        metrics: Dict[str, float] = {}
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Generate imagined trajectories (now returns 4 values)
            imagined_features, imagined_state, imagined_actions, imagined_action_indices = imagine_trajectory(
                starting_state,
                self.configuration.imag_horizon,
                self.world_model.dynamics,
                self.actor,
                self.world_model,
                self.world_model.encoder,
                self.configuration
            )  # [B, H, feature_dim], Dict[B, H, ...], [B, H, action_dim], [B, H] or [B, H, action_dim]
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] Imagined features shape:", imagined_features.shape)
                print("[DEBUG ImaginedBehavior] Imagined actions shape:", imagined_actions.shape)
                print("[DEBUG ImaginedBehavior] Imagined action indices shape:", imagined_action_indices.shape)

            # Compute intrinsic reward with batch-first features
            intrinsic_reward = self.reward_objective(
                imagined_features,  # [B, H, feature_dim]
                self.world_model
            )  # [B, H, 1]
            if intrinsic_reward.dim() == 2:
                intrinsic_reward = intrinsic_reward.unsqueeze(-1)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] Intrinsic reward shape:", intrinsic_reward.shape)

            # Ensure state has 'deter'
            if "deter" not in imagined_state:
                B = imagined_features.shape[0]
                horizon = self.configuration.imag_horizon
                device = imagined_features.device
                im_state = imagined_state.copy()
                im_state["deter"] = torch.zeros(B, horizon, self.configuration.dynamics_deterministic_dimension, device=device)
            else:
                im_state = imagined_state
            full_imagined_features = self.world_model.dynamics.get_features(im_state)  # [B, H, feature_dim]
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] Full imagined features shape:", full_imagined_features.shape)

            # Compute value predictions and lambda-return targets
            value_preds = self.value(full_imagined_features).mode()  # [B, H, 1]
            target, weights, baseline = lambda_return_target(
                intrinsic_reward.transpose(0, 1),  # [H, B, 1]
                value_preds.transpose(0, 1),       # [H, B, 1]
                self.configuration.discount_factor,
                self.configuration.discount_lambda
            )  # target: List[H] of [B, 1], weights: List[H] of [B, 1], baseline: [B, H, 1]
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] Lambda-return target stack shape (full):", torch.stack(target, dim=1).shape)

            # Compute actor loss with action indices
            actor_loss, loss_metrics = compute_actor_loss(
                actor=self.actor,
                features=full_imagined_features.detach(),
                actions=imagined_actions,
                action_indices=imagined_action_indices,
                target=target,
                weights=weights,
                baseline=baseline,
                value_network=self.value,
                configuration=self.configuration
            )
            actor_loss = torch.mean(actor_loss)
            metrics.update(loss_metrics)

        # Compute value loss with correct slicing
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predicted_value = self.value(full_imagined_features[:, :-1].detach())  # [B, H-1, ...]
            target_stack = torch.stack(target[:-1], dim=1)  # [B, H-1, 1]
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] predicted_value shape:", 
                      predicted_value.logits.shape if hasattr(predicted_value, 'logits') else predicted_value.shape)
                print("[DEBUG ImaginedBehavior] target_stack shape (trimmed):", target_stack.shape)

            if self.configuration.critic["distribution_type"] == "symlog_disc":
                target_stack_bins = discretize_symlog(target_stack, num_bins=255)  # [B, H-1]
                value_loss = -predicted_value.log_prob(target_stack_bins.detach())  # [B, H-1]
            else:
                value_loss = -predicted_value.log_prob(target_stack.detach())  # [B, H-1]
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] value_loss shape after log_prob:", value_loss.shape)

            if self.configuration.critic.get("use_slow_target", False) and self.slow_value is not None:
                slow_target_output = self.slow_value(full_imagined_features[:, :-1].detach())
                if self.configuration.critic["distribution_type"] == "symlog_disc":
                    slow_target_bins = discretize_symlog(slow_target_output.mode(), num_bins=255)
                    value_loss = value_loss - predicted_value.log_prob(slow_target_bins.detach())
                else:
                    value_loss = value_loss - predicted_value.log_prob(slow_target_output.mode().detach())

            weights_stack = torch.stack(weights[:-1], dim=1).squeeze(-1)  # [B, H-1]
            value_loss = torch.mean(weights_stack * value_loss)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG ImaginedBehavior] Final value_loss:", value_loss.item())

        # Update metrics
        metrics.update(tensor_stats(predicted_value.mode(), "value"))
        metrics.update(tensor_stats(target_stack, "target"))
        metrics.update(tensor_stats(intrinsic_reward, "intrinsic_reward"))  # Already batch-first
        metrics.update(self.actor_optimizer(actor_loss, self.actor.parameters()))
        metrics.update(self.value_optimizer(value_loss, self.value.parameters()))

        return imagined_features, imagined_state, imagined_actions, weights, metrics

    def _update_slow_target(self) -> None:
        if self.configuration.critic.get("use_slow_target", False) and self.slow_value is not None:
            if self.update_counter % self.configuration.critic["slow_target_update_interval"] == 0:
                mix_fraction = self.configuration.critic["slow_target_update_fraction"]
                for fast_param, slow_param in zip(self.value.parameters(), self.slow_value.parameters()):
                    slow_param.data = mix_fraction * fast_param.data + (1 - mix_fraction) * slow_param.data
            self.update_counter += 1

    def collect_optimizer_states(self) -> Dict[str, Any]:
        return {
            "actor": self.actor_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict()
        }