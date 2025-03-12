"""
Module implementing the ImaginedBehavior.
Generates imagined trajectories and trains the actor and critic networks.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from utils.helper_functions import static_scan_imagine, lambda_return_target, compute_actor_loss, tensor_stats
from utils.optimizer import Optimizer
from agent.networks import MLP

class ImaginedBehavior(nn.Module):
    def __init__(self, configuration: Any, world_model: Any) -> None:
        super(ImaginedBehavior, self).__init__()
        self.use_amp = (configuration.precision == 16)
        self.configuration = configuration
        self.world_model = world_model

        if configuration.dynamics_use_discrete:
            feature_dimension = configuration.discrete_latent_num * configuration.discrete_latent_size + configuration.dynamics_deterministic_dimension
        else:
            feature_dimension = configuration.dynamics_stochastic_dimension + configuration.dynamics_deterministic_dimension

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
            self.slow_value = self.value
            self.update_counter = 0

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
        if getattr(configuration, "reward_EMA", False):
            self.register_buffer("ema_values", torch.zeros((2,), device=configuration.computation_device))
            from utils.helper_functions import RewardEMA
            self.reward_ema = RewardEMA(device=configuration.computation_device)

    def train_step(self, starting_state: Any, objective_function: Any) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, Dict[str, float]]:
        self._update_slow_target()
        metrics: Dict[str, float] = {}
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            imagined_features, imagined_state, imagined_actions = static_scan_imagine(
                starting_state, 
                self.configuration.imag_horizon, 
                self.world_model.dynamics,
                self.configuration.number_of_possible_actions
            )
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] Imagined features shape:", imagined_features.shape)
            intrinsic_reward = objective_function(imagined_features, imagined_state, imagined_actions)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] Intrinsic reward shape:", intrinsic_reward.shape)
            if "deter" not in imagined_state:
                if getattr(self.configuration, "debug", False):
                    print("[DEBUG] 'deter' not found in imagined_state; creating dummy tensor.")
                B = imagined_features.shape[1]
                horizon = self.configuration.imag_horizon
                device = imagined_features.device
                im_state = imagined_state.copy()
                im_state["deter"] = torch.zeros(horizon, B, self.configuration.dynamics_deterministic_dimension, device=device)
            else:
                im_state = imagined_state
            full_imagined_features = self.world_model.dynamics.get_features(im_state)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] Full imagined features shape:", full_imagined_features.shape)
            target, weights, baseline = lambda_return_target(
                intrinsic_reward, self.value(full_imagined_features).mode(),
                self.configuration.discount_factor, self.configuration.discount_lambda
            )
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] Lambda-return target stack shape (full):", torch.stack(target, dim=1).shape)
            actor_loss, loss_metrics = compute_actor_loss(
                actor=self.actor,
                features=full_imagined_features,
                actions=imagined_actions,
                target=target,
                weights=weights,
                baseline=baseline,
                value_network=self.value,
                configuration=self.configuration
            )
            actor_loss = torch.mean(actor_loss)
            metrics.update(loss_metrics)
            value_input = full_imagined_features
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            predicted_value = self.value(value_input[:-1].detach())
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] predicted_value shape:", predicted_value.logits.shape)
            target_stack = torch.stack(target[:-1], dim=1)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] target_stack shape (trimmed):", target_stack.shape)
            value_loss = -predicted_value.log_prob(target_stack.detach())
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] value_loss shape after log_prob:", value_loss.shape)
            if self.configuration.critic["use_slow_target"]:
                slow_target_output = self.slow_value(value_input[:-1].detach())
                value_loss = value_loss - predicted_value.log_prob(slow_target_output.mode().detach())
            value_loss = torch.mean(weights[:-1] * value_loss)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG] Final value_loss:", value_loss.item())
        metrics.update(tensor_stats(predicted_value.mode(), "value"))
        metrics.update(tensor_stats(target_stack, "target"))
        metrics.update(tensor_stats(intrinsic_reward, "intrinsic_reward"))
        metrics.update(self.actor_optimizer(actor_loss, self.actor.parameters()))
        metrics.update(self.value_optimizer(value_loss, self.value.parameters()))
        return imagined_features, imagined_state, imagined_actions, weights, metrics

    def _update_slow_target(self) -> None:
        if self.configuration.critic["use_slow_target"]:
            if self.update_counter % self.configuration.critic["slow_target_update_interval"] == 0:
                mix_fraction = self.configuration.critic["slow_target_update_fraction"]
                for fast_param, slow_param in zip(self.value.parameters(), self.slow_value.parameters()):
                    slow_param.data = mix_fraction * fast_param.data + (1 - mix_fraction) * slow_param.data
            self.update_counter += 1

    def get_optimizer_state(self) -> Dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict()
        }
