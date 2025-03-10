"""
WorldModel module.
Encapsulates the environment dynamics with an encoder, a recurrent state-space model (RSSM),
and prediction heads for reconstruction, reward prediction, and continuation.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Tuple
import numpy as np

from agent.networks import MultiEncoder, RSSM, MultiDecoder, MLP, DistributionWrapper
from utils.optimizer import Optimizer
from utils.helper_functions import tensor_to_numpy, augment_image

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

        if hasattr(observation_space, "spaces"):
            input_shapes = {key: tuple(value.shape) for key, value in observation_space.spaces.items()}
        else:
            input_shapes = {"image": tuple(observation_space.shape)}

        if getattr(self.configuration, "debug", False):
            print("Input shapes:", input_shapes)

        self.encoder = MultiEncoder(
            input_shapes,
            output_dimension=configuration.encoder["output_dimension"],
            use_orthogonal=configuration.use_orthogonal_initialization
        )
        self.embedding_dimension = configuration.encoder["output_dimension"]

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

        if configuration.dynamics_use_discrete:
            feature_dimension = configuration.dynamics_stochastic_dimension * configuration.dynamics_use_discrete + configuration.dynamics_deterministic_dimension
        else:
            feature_dimension = configuration.dynamics_stochastic_dimension + configuration.dynamics_deterministic_dimension

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

    def preprocess(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        processed = {}
        # Convert each observation key to a tensor.
        for key, value in observations.items():
            if key == "action":
                # Convert actions to an integer tensor then to one-hot.
                action_tensor = torch.tensor(value, device=self.configuration.computation_device, dtype=torch.long)
                action_tensor = torch.nn.functional.one_hot(action_tensor, num_classes=self.configuration.number_of_possible_actions).float()
                processed[key] = action_tensor
            else:
                processed[key] = torch.tensor(value, device=self.configuration.computation_device, dtype=torch.float32)
        # If "action" is missing, provide a default one-hot tensor.
        if "action" not in processed:
            processed["action"] = torch.zeros((self.configuration.sequence_length, self.configuration.number_of_possible_actions),
                                                device=self.configuration.computation_device, dtype=torch.float32)
        if "image" in processed:
            processed["image"] = processed["image"] / 255.0
            # If the image tensor is 5D: (B, T, H, W, C), permute to (B, T, C, H, W).
            if processed["image"].ndim == 5 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(0, 1, 4, 2, 3)
            # Else if the image tensor is 3D: (H, W, C), permute to (C, H, W).
            elif processed["image"].ndim == 3 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(2, 0, 1)
            if getattr(self.configuration, "debug", False):
                print("Preprocessed image shape:", processed["image"].shape)
            # Optionally apply augmentation if enabled.
            if getattr(self.configuration, "augmentation_enabled", False):
                crop_size = getattr(self.configuration, "augmentation_crop_size", 64)
                processed["image"] = augment_image(processed["image"], crop_size)
        if "discount" in processed:
            processed["discount"] *= self.configuration.discount_factor
            processed["discount"] = processed["discount"].unsqueeze(-1)
        assert "is_first" in processed, "Observation must include 'is_first'."
        assert "is_terminal" in processed, "Observation must include 'is_terminal'."
        processed["continuation"] = (1.0 - processed["is_terminal"]).unsqueeze(-1)
        return processed

    def train_step(self, data: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, float]]:
        data = self.preprocess(data)
        if getattr(self.configuration, "debug", False):
            print("Data['image'] shape after preprocessing:", data["image"].shape)
            print("Data['action'] shape after preprocessing:", data["action"].shape)
        # Now data["action"] is a one-hot tensor.
        with torch.cuda.amp.autocast(self.use_amp):
            embeddings = self.encoder(data)
            if getattr(self.configuration, "debug", False):
                print("Embeddings shape:", embeddings.shape)
            posterior, prior = self.dynamics.observe(embeddings, data["action"], data["is_first"])
            kl_free = getattr(self.configuration, "kl_free", 0.0)
            dynamics_scale = getattr(self.configuration, "dynamics_loss_scale", 1.0)
            representation_scale = getattr(self.configuration, "representation_loss_scale", 1.0)
            kl_loss, kl_value, dynamics_loss, representation_loss = self.dynamics.compute_kl_loss(
                posterior, prior, kl_free, dynamics_scale, representation_scale
            )
            predictions = {}
            for head_name, head_module in self.heads.items():
                propagate_gradient = head_name in self.configuration.gradient_head_keys
                features = self.dynamics.get_features(posterior)
                features = features if propagate_gradient else features.detach()
                predictions[head_name] = head_module(features)
                if getattr(self.configuration, "debug", False):
                    if isinstance(predictions[head_name], dict):
                        print(f"Prediction '{head_name}' logits shape:", predictions[head_name]['image'].logits.shape)
                    else:
                        print(f"Prediction '{head_name}' logits shape:", predictions[head_name].logits.shape)
            losses = {}
            for head_name, prediction in predictions.items():
                if isinstance(prediction, dict):
                    losses[head_name] = -prediction["image"].log_prob(data["image"])
                    if getattr(self.configuration, "debug", False):
                        print(f"Loss for head '{head_name}' computed using conv branch")
                        print("Prediction logits shape:", prediction["image"].logits.shape)
                        print("Target shape:", data["image"].shape)
                else:
                    losses[head_name] = -prediction.log_prob(data[head_name])
            scaled_losses = {name: loss * self.loss_scales.get(name, 1.0) for name, loss in losses.items()}
            total_loss = sum(scaled_losses.values()) + kl_loss
        metrics = self.model_optimizer(torch.mean(total_loss), self.parameters())
        metrics.update({f"{name}_loss": tensor_to_numpy(loss) for name, loss in losses.items()})
        metrics.update({
            "kl_loss": tensor_to_numpy(torch.mean(kl_value)),
            "dynamics_loss": tensor_to_numpy(dynamics_loss),
            "representation_loss": tensor_to_numpy(representation_loss)
        })
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
        embeddings = self.encoder(data)
        states, _ = self.dynamics.observe(embeddings[:6, :5],
                                          data["action"][:6, :5],
                                          data["is_first"][:6, :5])
        reconstruction = self.heads["decoder"](self.dynamics.get_features(states))["image"].mode()[:6]
        initial_state = {key: value[:, -1] for key, value in states.items()}
        predicted_states = self.dynamics.imagine_with_action(data["action"][:6, 5:], initial_state)
        open_loop = self.heads["decoder"](self.dynamics.get_features(predicted_states))["image"].mode()
        model_video = torch.cat([reconstruction[:, :5], open_loop], dim=1)
        ground_truth = data["image"][:6]
        error = (model_video - ground_truth + 1.0) / 2.0
        return torch.cat([ground_truth, model_video, error], dim=2)

    def get_optimizer_state(self) -> Dict:
        return self.model_optimizer.state_dict()
