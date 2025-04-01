import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from typing import Any, Dict, Tuple
from agent.networks import MultiEncoder, RecurrentStateSpaceModel, MultiLayerPerceptron, Decoder, symmetric_logarithm_transformation, DistributionWrapper
from utils.optimizer import Optimizer
from utils.helper_functions import augment_image
import logging
import traceback
import numpy as np

# Setup logger for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

class WorldModel(nn.Module):
    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        configuration: Any,
    ) -> None:
        super(WorldModel, self).__init__()
        self.configuration = configuration
        self.action_space = action_space
        self.device = torch.device(configuration.computation_device if torch.cuda.is_available() else "cpu")
        self.use_automatic_mixed_precision = getattr(configuration, "use_automatic_mixed_precision", False)
        self.enable_debugging = getattr(configuration, "enable_debugging", False)
        self.current_step = 0  # Placeholder; updated by DreamerAgent
        
        # Determine action space type
        self.has_discrete_actions = hasattr(self.action_space, 'n')
        self.action_dimension = self.action_space.n if self.has_discrete_actions else np.prod(self.action_space.shape)
        
        # Image properties
        self.image_channels = observation_space.shape[2]  # C from (H, W, C)
        self.image_size = observation_space.shape[0]      # H from (H, W, C)
        
        # Initialize components (unchanged parts omitted)
        self.encoder = MultiEncoder(observation_space.shape, configuration)
        self.dynamics = RecurrentStateSpaceModel(
            hidden_units=getattr(configuration, "units", 512),
            recurrent_depth=getattr(configuration, "recurrent_depth", 1),
            use_discrete_latents=getattr(configuration, "use_discrete_latents", True),
            activation_function=getattr(configuration, "activation_function", "relu"),
            normalization_type=getattr(configuration, "normalization_type", "layer"),
            uniform_mix_ratio=getattr(configuration, "uniform_mix_ratio", 0.01),
            action_dimension=self.action_dimension,
            observation_embedding_dimension=getattr(configuration, "encoder_output_dimension", 1024),
            device=self.device,
            use_orthogonal_initialization=getattr(configuration, "use_orthogonal_initialization", True),
            discrete_latent_num=getattr(configuration, "discrete_latent_num", 32),
            discrete_latent_size=getattr(configuration, "discrete_latent_size", 32),
            has_discrete_actions=self.has_discrete_actions
        )
        
        self.heads = nn.ModuleDict({
            "decoder": Decoder(
                input_dimension=self.dynamics.feature_dimension,
                output_shape=(self.image_channels, self.image_size, self.image_size),
                hidden_dimension=getattr(configuration, "decoder_hidden_dimension", 128),
                number_of_layers=getattr(configuration, "decoder_number_of_layers", 4)
            ),
            "reward": MultiLayerPerceptron(
                input_dimension=self.dynamics.feature_dimension,
                output_shape=(255,),
                number_of_layers=getattr(configuration, "reward_head_number_of_layers", 5),
                units_per_layer=getattr(configuration, "units", 512),
                distribution_type="categorical",
                temperature=getattr(configuration, "reward_head_temperature", 1.0),
                uniform_mix_ratio=getattr(configuration, "reward_head_unimix_ratio", 0.01)
            ),
            "continuation": MultiLayerPerceptron(
                input_dimension=self.dynamics.feature_dimension,
                output_shape=(1,),
                number_of_layers=getattr(configuration, "continuation_head_number_of_layers", 5),
                units_per_layer=getattr(configuration, "units", 512),
                distribution_type="binary",  # Changed from "bernoulli" to "binary"
                temperature=getattr(configuration, "continuation_head_temperature", 1.0),
                uniform_mix_ratio=getattr(configuration, "continuation_head_unimix_ratio", 0.01)
            )
        })
        
        # Loss scales and optimizer initialization (unchanged)
        self.dynamics_loss_scale = getattr(configuration, "dynamics_loss_scale", 0.5)
        self.decoder_loss_scale = getattr(configuration, "decoder_loss_scale", 1.0)
        self.reward_head_loss_scale = getattr(configuration, "reward_head_loss_scale", 0.5)
        self.continuation_head_loss_scale = getattr(configuration, "continuation_head_loss_scale", 0.1)
        
        self.model_optimizer = Optimizer(
            name="world_model",
            parameters=self.parameters(),
            learning_rate=configuration.model_learning_rate,
            epsilon=configuration.optimizer_epsilon,
            gradient_clip=configuration.gradient_clip_value,
            weight_decay=configuration.weight_decay_value,
            optimizer_type=getattr(configuration, "world_model_optimizer_type", "adam"),
            use_automatic_mixed_precision=self.use_automatic_mixed_precision,
            warmup_steps=getattr(configuration, "model_warmup_steps", 0)
        )
        
        self.to(self.device)

    def preprocess(self, observations: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Preprocess observations into a format suitable for the world model.
        """
        processed_observations = {}
        device = self.device

        # Handle None observations
        if observations is None:
            logger.error("Received None for observations in preprocess")
            batch_size = 1
            return self._create_default_processed_data(batch_size, device)

        # Determine batch size and sequence length
        batch_size, sequence_length = 1, 1
        for key in ["image", "action", "reward"]:
            if key in observations and observations[key] is not None:
                if hasattr(observations[key], 'shape'):
                    shape = observations[key].shape
                    if len(shape) >= 2:
                        batch_size = shape[0]
                        sequence_length = shape[1] if len(shape) > (4 if key == "image" else 2) else 1
                        break

        if self.enable_debugging:
            logger.debug(f"Preprocessing with batch_size={batch_size}, sequence_length={sequence_length}")

        # Process each observation key
        for key, value in observations.items():
            if value is None:
                processed_observations[key] = self._handle_none_value(key, batch_size, sequence_length, device)
                continue

            try:
                if key == "image":
                    processed_observations[key] = self._process_image(value, batch_size, sequence_length, device)
                elif key == "action":
                    processed_observations[key] = self._process_action(value, batch_size, sequence_length, device)
                else:
                    processed_value = self._convert_to_tensor(value, device)
                    if processed_value.dim() == 1:
                        processed_value = processed_value.reshape(batch_size, sequence_length)
                    elif processed_value.dim() == 2 and processed_value.shape[1] != sequence_length:
                        if processed_value.shape[1] < sequence_length:
                            padding = torch.zeros(batch_size, sequence_length - processed_value.shape[1], device=device)
                            processed_value = torch.cat([processed_value, padding], dim=1)
                        else:
                            processed_value = processed_value[:, :sequence_length]
                    processed_observations[key] = processed_value
            except Exception as error:
                logger.error(f"Error processing observation key '{key}': {error}")
                processed_observations[key] = self._handle_none_value(key, batch_size, sequence_length, device)

        # Ensure all required keys are present
        required_keys = ["image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]
        for key in required_keys:
            if key not in processed_observations:
                processed_observations[key] = self._handle_none_value(key, batch_size, sequence_length, device)

        # Final adjustments
        processed_observations["is_terminal"] = processed_observations["is_terminal"].bool()
        processed_observations["continuation"] = (~processed_observations["is_terminal"]).float()
        discount_factor = getattr(self.configuration, "discount_factor", 0.997)
        processed_observations["discount"] = processed_observations["discount"] * discount_factor

        # Handle NaN values
        for key, value in processed_observations.items():
            if isinstance(value, torch.Tensor):
                processed_observations[key] = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.current_step == 0 and "is_first" in processed_observations:
            processed_observations["is_first"][:, 0] = True

        if self.enable_debugging:
            for key, value in processed_observations.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"Processed {key} shape: {value.shape}, dtype: {value.dtype}")

        return processed_observations

    def _create_default_processed_data(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        """Create default processed data when observations are None."""
        return {
            "image": torch.zeros((batch_size, 1, 3, 64, 64), device=device),
            "action": torch.zeros((batch_size, 1, self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0]), device=device),
            "is_first": torch.ones((batch_size, 1), dtype=torch.bool, device=device),
            "is_terminal": torch.zeros((batch_size, 1), dtype=torch.bool, device=device),
            "reward": torch.zeros((batch_size, 1), device=device),
            "discount": torch.ones((batch_size, 1), device=device),
            "continuation": torch.ones((batch_size, 1), device=device)
        }

    def _handle_none_value(self, key: str, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
        """Handle None values by providing defaults with proper dimensions."""
        if key == "image":
            return torch.zeros((batch_size, sequence_length, 3, 64, 64), device=device)
        elif key == "action":
            shape = (batch_size, sequence_length, self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0])
            action_tensor = torch.zeros(shape, device=device)
            if hasattr(self.action_space, 'n'):
                action_tensor[:, :, 0] = 1.0
            return action_tensor
        elif key in ["is_first", "is_terminal", "is_last"]:
            tensor = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
            if key == "is_first":
                tensor[:, 0] = True
            return tensor
        return torch.zeros((batch_size, sequence_length), device=device)

    def _process_action(self, action: Any, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
        """Process action data for the world model."""
        action_tensor = self._convert_to_tensor(action, device)
        if hasattr(self.action_space, 'n'):
            if action_tensor.dim() == 1:
                action_tensor = F.one_hot(action_tensor.long(), num_classes=self.action_space.n).float()
                action_tensor = action_tensor.reshape(batch_size, sequence_length, -1)
            elif action_tensor.dim() == 2:
                action_tensor = action_tensor.unsqueeze(1) if action_tensor.shape[1] != sequence_length else action_tensor
            if action_tensor.shape[1] != sequence_length:
                action_tensor = action_tensor[:, :sequence_length] if action_tensor.shape[1] > sequence_length else F.pad(action_tensor, (0, 0, 0, sequence_length - action_tensor.shape[1]))
        else:
            if action_tensor.dim() == 1:
                action_tensor = action_tensor.reshape(batch_size, sequence_length, -1)
            elif action_tensor.dim() == 2:
                action_tensor = action_tensor.unsqueeze(1) if action_tensor.shape[1] != sequence_length else action_tensor
            if action_tensor.shape[1] != sequence_length:
                action_tensor = action_tensor[:, :sequence_length] if action_tensor.shape[1] > sequence_length else F.pad(action_tensor, (0, 0, 0, sequence_length - action_tensor.shape[1]))
        return action_tensor

    def _convert_to_tensor(self, value: Any, device: str) -> torch.Tensor:
        """Convert value to a tensor on the specified device."""
        if isinstance(value, torch.Tensor):
            return value.to(device, dtype=torch.float32)
        elif isinstance(value, np.ndarray):
            return torch.tensor(value, device=device, dtype=torch.float32)
        return torch.tensor(value, device=device, dtype=torch.float32)

    def _process_image(self, image: Any, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
        """Process image data for the world model."""
        image_tensor = self._convert_to_tensor(image, device)
        if image_tensor.dim() == 3:  # [H, W, C]
            image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        elif image_tensor.dim() == 4:  # [B, H, W, C] or [B, C, H, W]
            if image_tensor.shape[-1] in [1, 3, 4]:
                image_tensor = image_tensor.permute(0, 3, 1, 2).unsqueeze(1)
            else:
                image_tensor = image_tensor.unsqueeze(1)
        elif image_tensor.dim() == 5 and image_tensor.shape[2] != sequence_length:
            image_tensor = image_tensor[:, :sequence_length]
        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        if self.training and getattr(self.configuration, "augmentation_enabled", True):
            image_tensor = augment_image(image_tensor, getattr(self.configuration, "augmentation_crop_size", 64))
        return image_tensor

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Perform a forward pass through the world model."""
        embeddings = self.encoder(data)
        actions, is_first = self._prepare_inputs(data, embeddings.shape[0])
        posterior, prior = self.dynamics.observe(embeddings.transpose(0, 1), actions.transpose(0, 1), is_first.transpose(0, 1))
        features = self.dynamics.get_features(posterior)
        predictions = self._generate_predictions(features)
        if "decoder" in predictions:
            predictions["decoder"] = {"image": predictions["decoder"]}
        return predictions

    def _prepare_inputs(self, data: Dict[str, torch.Tensor], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare actions and is_first flags for dynamics."""
        actions = data.get("action", torch.zeros((batch_size, 1, self.action_space.n if hasattr(self.action_space, 'n') else self.action_space.shape[0]), device=self.device))
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        is_first = data.get("is_first", torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device))
        if is_first.dim() == 1:
            is_first = is_first.unsqueeze(1)
        return actions, is_first

    def _generate_predictions(self, features: torch.Tensor) -> Dict[str, Any]:
        """Generate predictions from features using prediction heads."""
        predictions = {}
        for name, head in self.heads.items():
            try:
                predictions[name] = head(features)
            except Exception as error:
                logger.error(f"Error in {name} head: {error}")
                if name == "decoder":
                    predictions[name] = torch.zeros(features.shape[0], features.shape[1], self.image_channels, self.image_size, self.image_size, device=self.device)
                else:
                    predictions[name] = DistributionWrapper(
                        torch.zeros(features.shape[0], features.shape[1], 255 if name == "reward" else 1, device=self.device),
                        distribution_type="categorical" if name == "reward" else "bernoulli",
                        number_of_bins=255 if name == "reward" else 1
                    )
        return predictions

    def train_step(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Trains the world model on a batch of data, returning the state, predictions, and metrics.
        
        Args:
            batch_data: Dictionary containing batched data (e.g., image, action, reward, etc.).
        
        Returns:
            Tuple containing:
            - state: Dictionary with latent state tensors.
            - predictions: Dictionary with model predictions (e.g., features).
            - metrics: Dictionary with loss values and optimization metrics.
        """
        metrics = {}
        if not batch_data or not isinstance(batch_data, dict):
            logger.error(f"Invalid batch data type: {type(batch_data).__name__}")
            return None, None, {"batch_data_error": 1.0}

        try:
            # Preprocess batch data
            processed_batch = self.preprocess(batch_data)
            if processed_batch["image"].dim() == 4:
                processed_batch["image"] = processed_batch["image"].unsqueeze(1)  # [B, T=1, C, H, W]
            
            # Normalize images to [0, 1]
            processed_batch["image"] = processed_batch["image"].float() / 255.0
            
            for key in ["action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]:
                if processed_batch[key].dim() == 2:
                    processed_batch[key] = processed_batch[key].unsqueeze(1)  # [B, T=1, ...]

            if self.has_discrete_actions and processed_batch["action"].dtype != torch.long:
                processed_batch["action"] = processed_batch["action"].argmax(dim=-1).long()

            # Encode observations (no gradient)
            with torch.no_grad():
                observation_embedding = self.encoder(processed_batch)  # [B, T, embedding_dim]
                if observation_embedding.dim() == 2:
                    observation_embedding = observation_embedding.unsqueeze(1)  # [B, T=1, embedding_dim]

            # Run dynamics model
            state, posterior, prior = self.dynamics.observe_step(
                observation_embedding,
                processed_batch["action"],
                processed_batch["is_first"],
                None
            )
            if not state or "deter" not in state or state["deter"] is None:
                logger.error("Invalid state from dynamics")
                return None, None, {"invalid_state_error": 1.0}

            # Use automatic mixed precision
            with torch.amp.autocast('cuda', enabled=self.use_automatic_mixed_precision):
                # Get features
                features = self.dynamics.get_features(state)  # [B, T, feature_dim]
                B, T, feature_dim = features.shape
                features_flat = features.view(B * T, feature_dim)  # [B*T, feature_dim]

                # Decoder: reconstruct observations
                mean, logvar = self.heads["decoder"](features_flat)  # [B*T, C, H, W]
                mean = mean.view(B, T, *mean.shape[1:])  # [B, T, C, H, W]
                logvar = logvar.view(B, T, *logvar.shape[1:])  # [B, T, C, H, W]
                
                # Constrain decoder output
                mean = torch.clamp(mean, 0.0, 1.0)  # Ensure mean is in [0, 1]
                logvar = torch.clamp(logvar, -10.0, 10.0)  # Limit variance for stability

                # Define two-hot encoding function (unchanged)
                def create_two_hot_encoding(rewards, num_bins=255, lower_bound=-20.0, upper_bound=20.0):
                    symlog_rewards = torch.sign(rewards) * torch.log1p(torch.abs(rewards))
                    clamped_rewards = torch.clamp(symlog_rewards, lower_bound, upper_bound)
                    scaled_rewards = (clamped_rewards - lower_bound) / (upper_bound - lower_bound) * (num_bins - 1)
                    indices = scaled_rewards.floor().long()
                    weights = scaled_rewards - indices.float()
                    lower = torch.clamp(indices, 0, num_bins - 1)
                    upper = torch.clamp(indices + 1, 0, num_bins - 1)
                    target = torch.zeros(rewards.numel(), num_bins, device=rewards.device)
                    target.scatter_(1, lower.view(-1, 1), (1.0 - weights).view(-1, 1))
                    target.scatter_(1, upper.view(-1, 1), weights.view(-1, 1))
                    return target

                # Other prediction heads
                reward_distribution = self.heads["reward"](features_flat)  # [B*T, 255]
                reward_logits = reward_distribution.logits.view(B * T, 255)

                continuation_prediction = self.heads["continuation"](features_flat)  # [B*T, 1]
                continuation_logits = continuation_prediction.logits.view(B, T, -1)  # [B, T, 1]

                # Debug logs
                if self.enable_debugging:
                    logger.debug(f"Continuation logits shape: {continuation_logits.shape}")
                    logger.debug(f"Continuation target shape: {processed_batch['continuation'].shape}")
                    logger.debug(f"Observation target shape: {processed_batch['image'].shape}")
                    logger.debug(f"Mean shape: {mean.shape}")
                    logger.debug(f"Posterior keys: {list(posterior.keys())}")

                # Compute losses
                observation_target = processed_batch["image"]  # [B, T, C, H, W], already normalized to [0, 1]
                if observation_target.dim() == 5 and observation_target.shape[2] != 3:
                    observation_target = observation_target.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
                std = torch.exp(0.5 * logvar)
                observation_dist = distributions.Normal(mean, std)
                observation_loss = -observation_dist.log_prob(observation_target).mean()

                reward_target = processed_batch["reward"]  # [B, T, 1]
                target_twohot = create_two_hot_encoding(reward_target.view(-1))  # [B*T, 255]
                reward_loss = -torch.sum(target_twohot * F.log_softmax(reward_logits, dim=-1), dim=-1).mean()

                continuation_target = processed_batch["continuation"]  # [B, T, 1]
                if continuation_target.dim() == 3 and continuation_target.shape[1] != T:
                    continuation_target = continuation_target.transpose(1, 2)  # Fix [B, 1, T] -> [B, T, 1]
                continuation_loss = F.binary_cross_entropy_with_logits(continuation_logits, continuation_target)

                # KL loss
                if self.dynamics.use_discrete_latents:
                    posterior_dist = distributions.Categorical(logits=posterior["logits"])
                    prior_dist = distributions.Categorical(logits=prior["logits"])
                    kl_loss = distributions.kl_divergence(posterior_dist, prior_dist).mean()
                else:
                    posterior_dist = distributions.Normal(posterior["mean"], posterior["std"])
                    prior_dist = distributions.Normal(prior["mean"], prior["std"])
                    kl_loss = distributions.kl_divergence(posterior_dist, prior_dist).mean()

                # Combine losses with balanced scaling
                world_model_loss = (
                    self.dynamics_loss_scale * kl_loss +
                    1.0 * observation_loss +  # Reduced scale to 1.0
                    self.reward_head_loss_scale * reward_loss +
                    self.continuation_head_loss_scale * continuation_loss
                )

                # Optimize model parameters
                optimizer_metrics = self.model_optimizer(world_model_loss, self.parameters())
                metrics.update(optimizer_metrics)
                metrics.update({
                    "world_model_loss": world_model_loss.item(),
                    "decoder_loss": observation_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "reward_loss": reward_loss.item(),
                    "continuation_loss": continuation_loss.item()
                })

                # Log all metrics
                logger.info(f"World Model Train Step Metrics: {metrics}")

                return state, {"features": features}, metrics

        except Exception as error:
            logger.error(f"Exception in world_model.train_step: {error}")
            traceback.print_exc()
            return None, None, {"world_model_exception": 1.0, "error": str(error)}

    def _compute_prediction_losses(self, processed_data: Dict[str, torch.Tensor], predictions: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for decoder, reward, and continuation predictions."""
        losses = {}
        batch_size = processed_data["image"].shape[0]
        seq_len = processed_data["image"].shape[1] if processed_data["image"].dim() > 4 else 1

        # Decoder loss - MSE for image reconstruction
        try:
            target = processed_data["image"]
            if target.dim() == 4:
                target = target.unsqueeze(1)
            if target.dtype == torch.uint8:
                target = target.float() / 255.0

            observation_pred = predictions["decoder"]  # [B, T, C, H, W]
            losses["decoder"] = F.mse_loss(observation_pred, target, reduction='mean')

            if torch.isnan(losses["decoder"]):
                logger.warning(f"Decoder Loss is NaN: {losses['decoder'].item()}")
                losses["decoder"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception as error:
            logger.error(f"Error computing Decoder Loss: {error}")
            losses["decoder"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Reward loss
        try:
            target = processed_data["reward"]
            if target.dim() == 1:
                target = target.reshape(batch_size, seq_len)
            elif target.dim() == 3:
                target = target.squeeze(-1)

            reward_dist = predictions["reward"]
            reward_logits = reward_dist.logits
            if reward_logits.dim() == 3 and reward_logits.shape[0] != batch_size:
                reward_logits = reward_logits.permute(1, 0, 2)

            symlog_target = torch.clamp(symmetric_logarithm_transformation(target), -20.0, 20.0)
            scaled_target = (symlog_target + 20.0) / 40.0 * 254
            losses["reward"] = F.cross_entropy(reward_logits.view(-1, 255), scaled_target.view(-1).long(), reduction='mean')

            if torch.isnan(losses["reward"]):
                logger.warning(f"Reward Loss is NaN: {losses['reward'].item()}")
                losses["reward"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception as error:
            logger.error(f"Error computing Reward Loss: {error}")
            losses["reward"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Continuation loss
        try:
            target = processed_data["continuation"]
            if target.dim() == 1:
                target = target.reshape(batch_size, seq_len)
            elif target.dim() == 3:
                target = target.squeeze(-1)

            continuation_dist = predictions["continuation"]
            continuation_logits = continuation_dist.logits
            if continuation_logits.dim() == 3 and continuation_logits.shape[0] != batch_size:
                continuation_logits = continuation_logits.permute(1, 0, 2)

            losses["continuation"] = F.binary_cross_entropy_with_logits(continuation_logits.squeeze(-1), target, reduction='mean')

            if torch.isnan(losses["continuation"]):
                logger.warning(f"Continuation Loss is NaN: {losses['continuation'].item()}")
                losses["continuation"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        except Exception as error:
            logger.error(f"Error computing Continuation Loss: {error}")
            losses["continuation"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        return losses