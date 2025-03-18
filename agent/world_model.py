"""
WorldModel module.
Encapsulates the environment dynamics with an encoder, a recurrent state-space model (RSSM),
and prediction heads for reconstruction, reward prediction, and continuation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, Tuple
from copy import deepcopy

from agent.networks import MultiEncoder, RSSM, MultiDecoder, MLP, DistributionWrapper
from utils.optimizer import Optimizer
from utils.helper_functions import tensor_to_numpy, augment_image, lambda_return_target

class WorldModel(nn.Module):
    def __init__(self,
                 observation_space: Any,
                 action_space: Any,
                 current_step: int,
                 configuration: Any) -> None:
        super(WorldModel, self).__init__()
        self.current_step = current_step
        self.use_amp = (configuration.precision == 16)
        self.configuration = configuration

        # Determine input shapes from the observation space.
        if hasattr(observation_space, "spaces"):
            input_shapes = {key: tuple(value.shape) for key, value in observation_space.spaces.items()}
        else:
            input_shapes = {"image": tuple(observation_space.shape)}

        if getattr(self.configuration, "debug", False):
            print("Input shapes:", input_shapes)

        # Encoder: maps raw observations into an embedding.
        self.encoder = MultiEncoder(
            input_shapes,
            output_dimension=configuration.encoder["output_dimension"],
            use_orthogonal=configuration.use_orthogonal_initialization
        )
        self.embedding_dimension = configuration.encoder["output_dimension"]

        # Choose latent representation: if using discrete, compute feature dimension accordingly.
        if configuration.dynamics_use_discrete:
            # e.g., discrete_latent_num=32 and discrete_latent_size=32 yields feature_dimension = 32*32 + deter_dim.
            feature_dimension = (configuration.discrete_latent_num *
                                 configuration.discrete_latent_size +
                                 configuration.dynamics_deterministic_dimension)
        else:
            feature_dimension = (configuration.dynamics_stochastic_dimension +
                                 configuration.dynamics_deterministic_dimension)

        # Dynamics model (RSSM) – note that RSSM expects time-first inputs [T, B, ...].
        # (Our encoder outputs batch-first [B, T, ...], so we will transpose before calling dynamics.)
        self.dynamics = RSSM(
            stoch_dimension=configuration.dynamics_stochastic_dimension,
            deter_dimension=configuration.dynamics_deterministic_dimension,
            hidden_units=configuration.dynamics_hidden_units,
            rec_depth=configuration.dynamics_recurrent_depth,
            use_discrete=configuration.dynamics_use_discrete,
            activation_function=configuration.activation_function,
            normalization_type=configuration.normalization_type,
            mean_activation=configuration.dynamics_mean_activation,
            std_activation=configuration.dynamics_standard_deviation_activation,
            min_std=configuration.dynamics_minimum_standard_deviation,
            unimix_ratio=configuration.unimix_ratio,
            initial_state_type=configuration.initial_state_type,
            number_of_actions=configuration.number_of_possible_actions,
            embedding_dimension=self.embedding_dimension,
            device=configuration.computation_device,
            use_orthogonal=configuration.use_orthogonal_initialization
        )

        # Heads: decoder for reconstruction, reward predictor, and continuation predictor.
        self.heads = nn.ModuleDict({
            "decoder": MultiDecoder(
                feature_dimension,
                input_shapes,
                dummy_parameter=configuration.decoder["dummy_parameter"],
                use_orthogonal=configuration.use_orthogonal_initialization
            ),
            "reward": MLP(
                feature_dimension,
                output_shape=(255,) if configuration.reward_head["distribution_type"] == "symlog_disc" else (),
                layers=configuration.reward_head["number_of_layers"],
                units=configuration.units,
                activation=configuration.activation_function,
                normalization=configuration.normalization_type,
                dist=configuration.reward_head["distribution_type"],
                outscale=configuration.reward_head["output_scale"],
                device=configuration.computation_device,
                name="RewardHead",
                use_orthogonal=configuration.use_orthogonal_initialization
            ),
            "continuation": MLP(
                feature_dimension,
                output_shape=(),
                layers=configuration.continuation_head["number_of_layers"],
                units=configuration.units,
                activation=configuration.activation_function,
                normalization=configuration.normalization_type,
                dist="binary",
                outscale=configuration.continuation_head["output_scale"],
                device=configuration.computation_device,
                name="ContinuationHead",
                use_orthogonal=configuration.use_orthogonal_initialization
            )
        })

        self.model_optimizer = Optimizer(
            name="world_model",
            parameters=self.parameters(),
            learning_rate=configuration.model_learning_rate,
            eps=configuration.optimizer_epsilon,
            clip=configuration.gradient_clip_value,
            weight_decay=configuration.weight_decay_value,
            opt=configuration.optimizer_type,
            use_amp=self.use_amp
        )
        self.loss_scales = {
            "reward": configuration.reward_head["loss_scale"],
            "continuation": configuration.continuation_head["loss_scale"]
        }

        # If using slow target for critic, create a deep copy.
        if configuration.critic["use_slow_target"]:
            self.slow_value = deepcopy(
                MLP(
                    input_dim=feature_dimension,
                    output_shape=(255,) if configuration.critic["distribution_type"] == "symlog_disc" else (),
                    layers=configuration.critic["number_of_layers"],
                    units=configuration.units,
                    activation=configuration.activation_function,
                    normalization=configuration.normalization_type,
                    dist=configuration.critic["distribution_type"],
                    outscale=configuration.critic["output_scale"],
                    device=configuration.computation_device,
                    name="Value_slow",
                    use_orthogonal=configuration.use_orthogonal_initialization
                )
            )
            self.update_counter = 0

    def preprocess(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        processed = {}
        for key, value in observations.items():
            if key == "action":
                # If action_space is discrete, cast to long and one-hot; otherwise, keep as float.
                if hasattr(self.configuration, "action_space_type") and self.configuration.action_space_type == "continuous":
                    processed[key] = torch.tensor(value, device=self.configuration.computation_device, dtype=torch.float32)
                else:
                    action_tensor = torch.tensor(value, device=self.configuration.computation_device, dtype=torch.long)
                    if getattr(self.configuration, "debug", False):
                        print(f"[DEBUG preprocess] Raw action shape: {action_tensor.shape}, expected num_actions: {self.configuration.number_of_possible_actions}", flush=True)
                    action_tensor = torch.nn.functional.one_hot(action_tensor, num_classes=self.configuration.number_of_possible_actions).float()
                    if getattr(self.configuration, "debug", False):
                        print(f"[DEBUG preprocess] One-hot action shape: {action_tensor.shape}", flush=True)
                    processed[key] = action_tensor
            else:
                processed[key] = torch.tensor(value, device=self.configuration.computation_device, dtype=torch.float32)
        if "action" not in processed:
            processed["action"] = torch.zeros((self.configuration.sequence_length, self.configuration.number_of_possible_actions),
                                                device=self.configuration.computation_device, dtype=torch.float32)
        if "image" in processed:
            processed["image"] = processed["image"] / 255.0
            # Ensure image dimensions are in [B, T, C, H, W] or [C, H, W] as needed.
            if processed["image"].ndim == 5 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(0, 1, 4, 2, 3)
            elif processed["image"].ndim == 3 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(2, 0, 1)
            if getattr(self.configuration, "debug", False):
                print("Preprocessed image shape:", processed["image"].shape)
            if getattr(self.configuration, "augmentation_enabled", False):
                crop_size = getattr(self.configuration, "augmentation_crop_size", 64)
                processed["image"] = augment_image(processed["image"], crop_size)
        if "discount" in processed:
            processed["discount"] *= self.configuration.discount_factor
            processed["discount"] = processed["discount"].unsqueeze(-1)
        # Ensure required episode flags are present.
        assert "is_first" in processed, "Observation must include 'is_first'."
        assert "is_terminal" in processed, "Observation must include 'is_terminal'."
        processed["continuation"] = (1.0 - processed["is_terminal"]).unsqueeze(-1)
        return processed

    def train_step(self, data: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, float]]:
        data = self.preprocess(data)
        if getattr(self.configuration, "debug", False):
            print("Data['image'] shape after preprocessing:", data["image"].shape)
            print("Data['action'] shape after preprocessing:", data["action"].shape)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Convert encoder outputs from batch-first to time-first.
            embeddings = self.encoder(data).transpose(0, 1)  # Now shape [T, B, embed_dim]
            # For discrete actions, assume data["action"] is [B, T, ...] already; convert to time-first.
            actions = data["action"].transpose(0, 1) if data["action"].ndim > 1 else data["action"]
            is_first = data["is_first"].transpose(0, 1) if data["is_first"].ndim > 1 else data["is_first"]
            posterior, prior = self.dynamics.observe(embeddings, actions, is_first)
            # Compute KL loss.
            kl_free = getattr(self.configuration, "kl_free", 1.0)
            dynamics_scale = getattr(self.configuration, "dynamics_loss_scale", 1.0)
            representation_scale = getattr(self.configuration, "representation_loss_scale", 1.0)
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.compute_kl_loss(posterior, prior, kl_free, dynamics_scale, representation_scale)
            predictions = {}
            # Obtain features from the dynamics (RSSM) using the posterior.
            features = self.dynamics.get_features(posterior)
            # Pass features (detached if gradients should not flow) to each head.
            for head_name, head_module in self.heads.items():
                propagate_gradient = head_name in self.configuration.gradient_head_keys
                features_to_use = features if propagate_gradient else features.detach()
                predictions[head_name] = head_module(features_to_use)
                if getattr(self.configuration, "debug", False):
                    if isinstance(predictions[head_name], dict):
                        print(f"Prediction '{head_name}' logits shape:", predictions[head_name]['image'].logits.shape)
                    else:
                        print(f"Prediction '{head_name}' logits shape:", predictions[head_name].logits.shape)
            losses = {}
            for head_name, prediction in predictions.items():
                if isinstance(prediction, dict):
                    losses[head_name] = -prediction["image"].log_prob(data["image"])
                else:
                    losses[head_name] = -prediction.log_prob(data[head_name])
            scaled_losses = {name: loss * self.loss_scales.get(name, 1.0) for name, loss in losses.items()}
            total_loss = sum(scaled_losses.values()) + kl_loss
        metrics = self.model_optimizer(torch.mean(total_loss), self.parameters())
        metrics.update({f"{name}_loss": tensor_to_numpy(loss) for name, loss in losses.items()})
        metrics.update({
            "kl_loss": tensor_to_numpy(torch.mean(kl_value)),
            "dynamics_loss": tensor_to_numpy(dyn_loss),
            "representation_loss": tensor_to_numpy(rep_loss)
        })
        # Detach posterior for use by other modules.
        posterior_detached = {key: value.detach() for key, value in posterior.items()}
        context = {
            "embedding": embeddings,
            "features": self.dynamics.get_features(posterior),
            "kl": kl_value
        }
        if getattr(self.configuration, "debug", False):
            print("Training step metrics:", metrics)
        return posterior_detached, context, metrics

    def generate_video(self, data: Dict[str, Any]) -> torch.Tensor:
        data = self.preprocess(data)
        # Convert to time-first.
        embeddings = self.encoder(data).transpose(0, 1)
        # Use the first 6 episodes (batch) and first 5 time steps for observation.
        states, _ = self.dynamics.observe(embeddings[:,:5],
                                          data["action"].transpose(0,1)[:,:5],
                                          data["is_first"].transpose(0,1)[:,:5])
        reconstruction = self.heads["decoder"](self.dynamics.get_features(states))["image"].mode()[:6]
        initial_state = {key: value[:, -1] for key, value in states.items()}
        predicted_states = self.dynamics.imagine_with_action(data["action"].transpose(0,1)[:,:6],
                                                               initial_state)
        open_loop = self.heads["decoder"](self.dynamics.get_features(predicted_states))["image"].mode()
        model_video = torch.cat([reconstruction[:, :5], open_loop], dim=1)
        ground_truth = data["image"][:6]
        error = (model_video - ground_truth + 1.0) / 2.0
        return torch.cat([ground_truth, model_video, error], dim=2)

    def get_optimizer_state(self) -> Dict:
        return self.model_optimizer.state_dict()
