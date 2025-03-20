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
from utils.helper_functions import tensor_to_numpy, augment_image, lambda_return_target, symlog

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
        self.device = configuration.computation_device if torch.cuda.is_available() else "cpu"
        self.configuration.debug = getattr(configuration, "debug", False)  # Allow configurable debug

        if hasattr(observation_space, "spaces"):
            input_shapes = {key: tuple(value.shape) for key, value in observation_space.spaces.items()}
        else:
            input_shapes = {"image": tuple(observation_space.shape)}

        if self.configuration.debug:
            print("[DEBUG] Input shapes:", input_shapes, flush=True)

        self.encoder = MultiEncoder(
            input_shapes,
            output_dimension=configuration.encoder["output_dimension"],
            use_orthogonal=configuration.use_orthogonal_initialization
        ).to(self.device)
        self.embedding_dimension = configuration.encoder["output_dimension"]

        if configuration.dynamics_use_discrete:
            feature_dimension = (configuration.discrete_latent_num *
                               configuration.discrete_latent_size +
                               configuration.dynamics_deterministic_dimension)
        else:
            feature_dimension = (configuration.dynamics_stochastic_dimension +
                               configuration.dynamics_deterministic_dimension)

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
            device=self.device,
            use_orthogonal=configuration.use_orthogonal_initialization
        ).to(self.device)

        self.heads = nn.ModuleDict({
            "decoder": MultiDecoder(
                feature_dimension,
                input_shapes,
                dummy_parameter=configuration.decoder["dummy_parameter"],
                use_orthogonal=configuration.use_orthogonal_initialization
            ).to(self.device),
            "reward": MLP(
                feature_dimension,
                output_shape=(255,) if configuration.reward_head["distribution_type"] == "symlog_disc" else (),
                layers=configuration.reward_head["number_of_layers"],
                units=configuration.units,
                activation=configuration.activation_function,
                normalization=configuration.normalization_type,
                dist=configuration.reward_head["distribution_type"],
                outscale=configuration.reward_head["output_scale"],
                device=self.device,
                name="RewardHead",
                use_orthogonal=configuration.use_orthogonal_initialization
            ).to(self.device),
            "continuation": MLP(
                feature_dimension,
                output_shape=(),
                layers=configuration.continuation_head["number_of_layers"],
                units=configuration.units,
                activation=configuration.activation_function,
                normalization=configuration.normalization_type,
                dist="binary",
                outscale=configuration.continuation_head["output_scale"],
                device=self.device,
                name="ContinuationHead",
                use_orthogonal=configuration.use_orthogonal_initialization
            ).to(self.device)
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
            "decoder": configuration.decoder.get("loss_scale", 1.0),
            "reward": configuration.reward_head["loss_scale"],
            "continuation": configuration.continuation_head["loss_scale"]
        }

        if configuration.critic.get("use_slow_target", False):
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
                    device=self.device,
                    name="Value_slow",
                    use_orthogonal=configuration.use_orthogonal_initialization
                )
            ).to(self.device)
            self.update_counter = 0
        else:
            self.slow_value = None

    def preprocess(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        processed = {}
        for key, value in observations.items():
            if key == "action":
                if hasattr(self.configuration, "action_space_type") and self.configuration.action_space_type == "continuous":
                    processed[key] = torch.tensor(value, device=self.device, dtype=torch.float32)
                else:
                    action_tensor = torch.tensor(value, device=self.device, dtype=torch.long).clone().detach()
                    if action_tensor.dim() == 1:  # [B]
                        action_tensor = action_tensor.unsqueeze(1)  # [B, 1]
                    if self.configuration.debug:
                        print(f"[DEBUG preprocess] Raw action shape: {action_tensor.shape}", flush=True)
                    action_tensor = F.one_hot(action_tensor, num_classes=self.configuration.number_of_possible_actions).float()
                    processed[key] = action_tensor  # [B, T, num_actions]
                    if self.configuration.debug:
                        print(f"[DEBUG preprocess] One-hot action shape: {action_tensor.shape}", flush=True)
            else:
                processed[key] = torch.tensor(value, device=self.device, dtype=torch.float32).clone().detach()
        
        if "action" not in processed:
            processed["action"] = torch.zeros((self.configuration.batch_size, self.configuration.sequence_length, self.configuration.number_of_possible_actions),
                                           device=self.device, dtype=torch.float32)
        if "image" in processed:
            processed["image"] = processed["image"] / 255.0
            if processed["image"].ndim == 5 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
            elif processed["image"].ndim == 4 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(0, 3, 1, 2)  # [B, C, H, W]
            elif processed["image"].ndim == 3 and processed["image"].shape[-1] in [1, 3]:
                processed["image"] = processed["image"].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            if self.configuration.debug:
                print(f"[DEBUG preprocess] Preprocessed image shape: {processed['image'].shape}", flush=True)
            if getattr(self.configuration, "augmentation_enabled", False):
                crop_size = getattr(self.configuration, "augmentation_crop_size", 64)
                processed["image"] = augment_image(processed["image"], crop_size)
        if "discount" in processed:
            processed["discount"] = processed["discount"] * self.configuration.discount_factor
            if processed["discount"].dim() == 1:
                processed["discount"] = processed["discount"].unsqueeze(-1)  # [B, 1]
        assert "is_first" in processed, "Observation must include 'is_first'."
        assert "is_terminal" in processed, "Observation must include 'is_terminal'."
        if processed["is_first"].dim() == 1:
            processed["is_first"] = processed["is_first"].unsqueeze(1)  # [B, 1]
        if processed["is_terminal"].dim() == 1:
            processed["is_terminal"] = processed["is_terminal"].unsqueeze(1)  # [B, 1]
        processed["continuation"] = (1.0 - processed["is_terminal"]).float()
        if self.configuration.debug:
            print(f"[DEBUG preprocess] Preprocessed continuation shape: {processed['continuation'].shape}", flush=True)
        return processed

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Forward pass to generate predictions for all heads.
        Input: Dict with [B, T, ...] tensors
        Output: Dict with predictions as DistributionWrapper objects
        """
        embeddings = self.encoder(data)  # [B, T, embed_dim]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # [B, 1, embed_dim]
        
        actions = data["action"]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # [B, 1, num_actions]
        
        is_first = data["is_first"]
        if is_first.dim() == 1:
            is_first = is_first.unsqueeze(1)  # [B, 1]
        
        posterior, _ = self.dynamics.observe(
            embeddings.transpose(0, 1),  # [T, B, embed_dim]
            actions.transpose(0, 1),     # [T, B, num_actions]
            is_first.transpose(0, 1)     # [T, B]
        )
        features = self.dynamics.get_features(posterior)  # [B, T, feature_dim]
        if features.dim() == 2:  # [B, feature_dim]
            features = features.unsqueeze(1)  # [B, 1, feature_dim]
        
        predictions = {}
        for name, head in self.heads.items():
            predictions[name] = head(features)  # [B, T, ...]
            if self.configuration.debug:
                output = predictions[name].logits if hasattr(predictions[name], 'logits') else predictions[name]
                print(f"[DEBUG forward] {name} prediction shape: {output.shape}", flush=True)
        
        return predictions

    def train_step(self, data: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, float]]:
        data = self.preprocess(data)
        if self.configuration.debug:
            print(f"[DEBUG train_step] Data['image'] shape after preprocessing: {data['image'].shape}", flush=True)
            print(f"[DEBUG train_step] Data['continuation'] shape after preprocessing: {data['continuation'].shape}", flush=True)
        
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # Encode data
            embeddings = self.encoder(data)  # [B, T, embed_dim]
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)  # [B, 1, embed_dim]
            
            # Prepare inputs for dynamics
            actions = data["action"]
            if actions.dim() == 2:
                actions = actions.unsqueeze(1)  # [B, 1, num_actions]
            is_first = data["is_first"]
            if is_first.dim() == 1:
                is_first = is_first.unsqueeze(1)  # [B, 1]
            
            # Dynamics model
            posterior, prior = self.dynamics.observe(
                embeddings.transpose(0, 1),  # [T, B, embed_dim]
                actions.transpose(0, 1),     # [T, B, num_actions]
                is_first.transpose(0, 1)     # [T, B]
            )
            features = self.dynamics.get_features(posterior)  # [B, T, feature_dim]
            if features.dim() == 2:
                features = features.unsqueeze(1)  # [B, 1, feature_dim]
            if self.configuration.debug:
                print(f"[DEBUG train_step] Features shape: {features.shape}", flush=True)
            
            # KL loss
            kl_free = getattr(self.configuration, "kl_free", 1.0)
            dynamics_scale = getattr(self.configuration, "dynamics_loss_scale", 1.0)
            representation_scale = getattr(self.configuration, "representation_loss_scale", 1.0)
            kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.compute_kl_loss(
                posterior, prior, kl_free, dynamics_scale, representation_scale
            )
            
            # Generate predictions
            predictions = {}
            for name, head in self.heads.items():
                predictions[name] = head(features)  # [B, T, ...]
            
            # Compute losses
            losses = {}
            for head_name, prediction in predictions.items():
                if head_name == "decoder":
                    target = data["image"]  # [B, T, C, H, W]
                    losses[head_name] = -prediction["image"].log_prob(target)
                elif head_name == "reward":
                    target = data["reward"]  # [B, T, 1]
                    if target.dim() == 2:
                        target = target.unsqueeze(-1)  # [B, T, 1]
                    if self.configuration.reward_head["distribution_type"] == "symlog_disc":
                        target = symlog(target)
                    losses[head_name] = -prediction.log_prob(target.squeeze(-1) if target.shape[-1] == 1 else target)
                else:  # continuation
                    target = data["continuation"]  # [B, T]
                    if target.dim() == 3 and target.shape[-1] == 1:
                        target = target.squeeze(-1)  # [B, T]
                    losses[head_name] = -prediction.log_prob(target)
                
                if self.configuration.debug:
                    print(f"[DEBUG train_step] {head_name} target shape: {target.shape}", flush=True)
                    print(f"[DEBUG train_step] {head_name} loss shape: {losses[head_name].shape}", flush=True)
            
            # Scale and combine losses
            scaled_losses = {name: loss.mean() * self.loss_scales.get(name, 1.0) for name, loss in losses.items()}
            total_loss = sum(scaled_losses.values()) + kl_loss
        
        # Optimize
        metrics = self.model_optimizer(total_loss, self.parameters())
        metrics.update({f"{name}_loss": loss.item() for name, loss in scaled_losses.items()})
        metrics.update({
            "kl_loss": kl_value.item(),
            "dynamics_loss": dyn_loss.item() if isinstance(dyn_loss, torch.Tensor) else dyn_loss,
            "representation_loss": rep_loss.item() if isinstance(rep_loss, torch.Tensor) else rep_loss
        })
        
        posterior_detached = {key: value.detach() for key, value in posterior.items()}
        context = {"embedding": embeddings, "features": features, "kl": kl_value}
        if self.configuration.debug:
            print(f"[DEBUG train_step] Training step metrics: {metrics}", flush=True)
        
        return posterior_detached, context, metrics

    def generate_video(self, data: Dict[str, Any]) -> torch.Tensor:
        data = self.preprocess(data)
        embeddings = self.encoder(data)  # [B, T, embed_dim]
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)  # [B, 1, embed_dim]
        
        actions = data["action"]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # [B, 1, num_actions]
        
        is_first = data["is_first"]
        if is_first.dim() == 1:
            is_first = is_first.unsqueeze(1)  # [B, 1]
        
        states, _ = self.dynamics.observe(
            embeddings.transpose(0, 1)[:, :5],  # [T, B, embed_dim]
            actions.transpose(0, 1)[:, :5],     # [T, B, num_actions]
            is_first.transpose(0, 1)[:, :5]     # [T, B]
        )
        features = self.dynamics.get_features(states)  # [B, T, feature_dim]
        reconstruction = self.heads["decoder"](features)["image"].mode()  # [B, T, C, H, W]
        
        initial_state = {key: value[:, -1] for key, value in states.items()}  # [B, ...]
        predicted_states = self.dynamics.imagine_with_action(actions[:, :6], initial_state)
        predicted_features = self.dynamics.get_features(predicted_states)  # [B, H, feature_dim]
        open_loop = self.heads["decoder"](predicted_features)["image"].mode()  # [B, H, C, H, W]
        
        model_video = torch.cat([reconstruction[:, :5], open_loop], dim=1)  # [B, T+H, C, H, W]
        ground_truth = data["image"][:, :6]  # [B, T, C, H, W]
        error = (model_video - ground_truth + 1.0) / 2.0
        return torch.cat([ground_truth, model_video, error], dim=2)  # [B, T, C, H*3, W]

    def get_optimizer_state(self) -> Dict:
        return self.model_optimizer.state_dict()