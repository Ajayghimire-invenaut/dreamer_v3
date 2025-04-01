import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, List, Optional
from agent.networks import MultiLayerPerceptron, RewardObjective, lambda_return_target, create_two_hot_encoding, inverse_symmetric_logarithm_transformation
from utils.optimizer import Optimizer
from utils.helper_functions import imagine_trajectory
import logging
import traceback
import time
import os

# Setup logger for debugging and information
logger = logging.getLogger(__name__)

class ImaginedBehavior(nn.Module):
    """
    Implements imagination-based actor-critic policy optimization for DreamerV3.
    Uses imagined rollouts from the world model to train a policy (actor) and value function (critic).
    """
    def __init__(self, configuration: Any, world_model: Any) -> None:
        super(ImaginedBehavior, self).__init__()
        self.use_automatic_mixed_precision = (getattr(configuration, "precision", 16) == 16) and torch.cuda.is_available()
        self.configuration = configuration
        self.world_model = world_model
        self.enable_debugging = getattr(configuration, "debug", False)
        self.device = world_model.device
        
        # Determine if actions are discrete or continuous
        self.has_discrete_actions = hasattr(world_model.action_space, 'n')
        default_gradient_type = "reinforce" if self.has_discrete_actions else "dynamics"
        self.imagination_gradient_type = getattr(configuration, "actor_imag_gradient", default_gradient_type)
        
        if self.enable_debugging:
            logger.debug(f"[DEBUG ImaginedBehavior] Using {self.imagination_gradient_type} gradients for {'discrete' if self.has_discrete_actions else 'continuous'} actions")

        # Calculate feature dimension based on world model dynamics
        self.feature_dimension = (
            world_model.dynamics.discrete_latent_num * world_model.dynamics.discrete_latent_size +
            world_model.dynamics.hidden_units
        ) if world_model.dynamics.use_discrete_latents else (
            world_model.dynamics.stochastic_dimension + world_model.dynamics.hidden_units
        )

        # Set up action space properties
        if self.has_discrete_actions:
            action_shape = (world_model.action_space.n,)
            distribution_type = getattr(configuration, "actor_distribution_type", "onehot")
        else:
            action_dimension = world_model.action_space.shape[0]
            action_shape = (action_dimension * 2,)  # Mean and log_std for each dimension
            distribution_type = "normal"

        # Initialize actor network with 5 layers
        self.actor_network = MultiLayerPerceptron(
            input_dimension=self.feature_dimension,
            output_shape=action_shape,
            number_of_layers=getattr(configuration, "actor_number_of_layers", 5),
            units_per_layer=getattr(configuration, "units", 512),
            activation_function=getattr(configuration, "activation_function", "silu"),
            normalization_type=getattr(configuration, "normalization_type", "layer"),
            distribution_type=distribution_type,
            temperature=getattr(configuration, "actor_temperature", 1.0),
            uniform_mix_ratio=getattr(configuration, "actor_unimix_ratio", 0.01),
            device=self.device,
            name="Actor",
            use_orthogonal_initialization=getattr(configuration, "use_orthogonal_initialization", True)
        )

        # Initialize value (critic) network with 5 layers
        self.value_network = MultiLayerPerceptron(
            input_dimension=self.feature_dimension,
            output_shape=(255,),
            number_of_layers=getattr(configuration, "critic_number_of_layers", 5),
            units_per_layer=getattr(configuration, "units", 512),
            activation_function=getattr(configuration, "activation_function", "silu"),
            normalization_type=getattr(configuration, "normalization_type", "layer"),
            distribution_type="symlog_disc",
            device=self.device,
            name="Value",
            use_orthogonal_initialization=getattr(configuration, "use_orthogonal_initialization", True)
        )
        
        # Zero-initialize the last layer of the value network
        if hasattr(self.value_network, 'layers') and len(self.value_network.layers) > 0:
            last_layer = self.value_network.layers[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                nn.init.zeros_(last_layer.bias)

        # Initialize slow target value network with 5 layers
        self.use_slow_target_network = getattr(configuration, "critic_use_slow_target", True)
        if self.use_slow_target_network:
            self.slow_value_network = MultiLayerPerceptron(
                input_dimension=self.feature_dimension,
                output_shape=(255,),
                number_of_layers=getattr(configuration, "critic_number_of_layers", 5),
                units_per_layer=getattr(configuration, "units", 512),
                activation_function=getattr(configuration, "activation_function", "silu"),
                normalization_type=getattr(configuration, "normalization_type", "layer"),
                distribution_type="symlog_disc",
                device=self.device,
                name="SlowValue",
                use_orthogonal_initialization=getattr(configuration, "use_orthogonal_initialization", True)
            )
            if hasattr(self.slow_value_network, 'layers') and len(self.slow_value_network.layers) > 0:
                last_layer = self.slow_value_network.layers[-1]
                if isinstance(last_layer, nn.Linear):
                    nn.init.zeros_(last_layer.weight)
                    nn.init.zeros_(last_layer.bias)
            self.slow_value_network.load_state_dict(self.value_network.state_dict())
            self.update_counter = 0
        else:
            self.slow_value_network = None

        # Initialize optimizers
        optimizer_arguments = dict(
            weight_decay=getattr(configuration, "weight_decay_value", 0.0),
            optimizer_type=getattr(configuration, "optimizer_type", "adam"),
            use_automatic_mixed_precision=self.use_automatic_mixed_precision
        )
        self.actor_optimizer = Optimizer(
            "actor",
            self.actor_network.parameters(),
            learning_rate=getattr(configuration, "actor_lr", 3e-5),
            epsilon=getattr(configuration, "actor_eps", 1e-8),
            gradient_clip=getattr(configuration, "actor_grad_clip", 1000.0),
            **optimizer_arguments
        )
        self.value_optimizer = Optimizer(
            "value",
            self.value_network.parameters(),
            learning_rate=getattr(configuration, "critic_lr", 3e-5),
            epsilon=getattr(configuration, "critic_eps", 1e-8),
            gradient_clip=getattr(configuration, "critic_grad_clip", 1000.0),
            **optimizer_arguments
        )

        # Exponential moving average for return scaling
        self.return_ema_alpha = getattr(configuration, "return_ema_alpha", 0.01)
        self.return_scale = 1.0
        self.minimum_return_scale = 1.0

        # Reward processing objective
        self.reward_objective = RewardObjective(
            alpha=getattr(configuration, "reward_ema_alpha", 0.01)
        )

        # Imagination horizon as an adjustable parameter
        self.imag_horizon = getattr(configuration, "imag_horizon", 15)

    def train(self, mode: bool = True) -> 'ImaginedBehavior':
        """Set the module in training or evaluation mode."""
        super(ImaginedBehavior, self).train(mode)
        return self

    def update_slow_target_network(self) -> None:
        """Update the slow target value network using Polyak averaging."""
        if not self.use_slow_target_network or self.slow_value_network is None:
            return
        if self.update_counter % getattr(self.configuration, "critic_slow_target_update_interval", 1) == 0:
            mix_fraction = getattr(self.configuration, "critic_slow_target_update_fraction", 0.005)
            for slow_param, param in zip(self.slow_value_network.parameters(), self.value_network.parameters()):
                slow_param.data.lerp_(param.data, mix_fraction)
        self.update_counter += 1

    def collect_optimizer_states(self) -> Dict[str, Any]:
        """Collect optimizer states for saving checkpoints."""
        return {
            "actor": self.actor_optimizer.state_dict(),
            "value": self.value_optimizer.state_dict()
        }

    def forward(self, features: torch.Tensor) -> Any:
        """Forward pass through the actor network to get action distributions."""
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.actor_network(features)

    def check_gradients(self, loss_name: str, model: nn.Module) -> Dict[str, float]:
        """Check gradients for NaN or infinity values."""
        metrics = {}
        total_gradient_norm = 0.0
        maximum_gradient_norm = 0.0
        has_nan_gradients = False
        has_inf_gradients = False
        parameter_count = 0
        
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if torch.isnan(parameter.grad).any():
                    has_nan_gradients = True
                    if self.enable_debugging:
                        logger.debug(f"NaN gradient detected in {name}")
                if torch.isinf(parameter.grad).any():
                    has_inf_gradients = True
                    if self.enable_debugging:
                        logger.debug(f"Infinite gradient detected in {name}")
                
                parameter_count += 1
                gradient_norm = parameter.grad.norm().item()
                total_gradient_norm += gradient_norm
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
                
        if parameter_count > 0:
            metrics[f"{loss_name}_gradient_average"] = total_gradient_norm / parameter_count
            metrics[f"{loss_name}_gradient_maximum"] = maximum_gradient_norm
        if has_nan_gradients:
            metrics[f"{loss_name}_gradient_has_nan"] = 1.0
        if has_inf_gradients:
            metrics[f"{loss_name}_gradient_has_inf"] = 1.0
            
        return metrics

    def compute_return_scale(self, returns: torch.Tensor) -> float:
        """Compute a scale for normalizing returns using percentile statistics."""
        flat_returns = returns.flatten()
        
        if flat_returns.shape[0] < 10:
            return max(self.return_scale, self.minimum_return_scale)
            
        sorted_returns, _ = torch.sort(flat_returns)
        number_of_samples = sorted_returns.shape[0]
        
        fifth_percentile_index = max(0, min(number_of_samples - 1, int(0.05 * number_of_samples)))
        ninety_fifth_percentile_index = max(0, min(number_of_samples - 1, int(0.95 * number_of_samples)))
        
        fifth_percentile = sorted_returns[fifth_percentile_index].item()
        ninety_fifth_percentile = sorted_returns[ninety_fifth_percentile_index].item()
        
        current_range = max(ninety_fifth_percentile - fifth_percentile, 1e-8)
        new_scale = max(
            self.return_ema_alpha * current_range + (1 - self.return_ema_alpha) * self.return_scale,
            self.minimum_return_scale
        )
        
        self.return_scale = new_scale
        return new_scale

    def compute_expected_value(self, value_distribution: Any) -> torch.Tensor:
        """Compute the expected value from a categorical distribution over symlog bins."""
        if hasattr(value_distribution, 'distribution') and hasattr(value_distribution.distribution, 'probs'):
            probabilities = value_distribution.distribution.probs
            number_of_bins = probabilities.shape[-1]
            bin_values = torch.linspace(-20.0, 20.0, number_of_bins, device=probabilities.device)
            expected_symlog_value = torch.sum(probabilities * bin_values, dim=-1)
            return inverse_symmetric_logarithm_transformation(expected_symlog_value)
        return value_distribution.mode()

    def train_step(self, starting_state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, List[torch.Tensor], Dict[str, float]]:
        """Train the actor and critic using imagined trajectories from the world model."""
        metrics = {}
        
        import logging
        logger = logging.getLogger(__name__)
        log_file = os.path.join(os.getcwd(), "log.txt")
        with open(log_file, "a") as f:
            f.write(f"--- Starting train_step at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        
        if not starting_state or not isinstance(starting_state, dict) or "deter" not in starting_state:
            error_msg = f"Invalid starting state for behavior training: {starting_state}"
            if self.enable_debugging:
                logger.debug(error_msg)
            with open(log_file, "a") as f:
                f.write(f"ERROR: {error_msg}\n")
            return None, None, None, None, {"invalid_state_error": 1.0}
        
        if not isinstance(starting_state["deter"], torch.Tensor):
            error_msg = f"starting_state['deter'] is not a tensor: {type(starting_state['deter'])}, value: {starting_state['deter']}"
            if self.enable_debugging:
                logger.debug(error_msg)
            with open(log_file, "a") as f:
                f.write(f"ERROR: {error_msg}\n")
            raise ValueError(f"starting_state['deter'] must be a tensor, got {type(starting_state['deter'])}")
        
        try:
            batch_size = starting_state["deter"].shape[0] if starting_state["deter"].dim() == 3 else starting_state["deter"].shape[0]
            imagination_horizon = self.imag_horizon

            if self.enable_debugging:
                logger.debug(f"Full starting_state: {starting_state}")
                logger.debug(f"train_step: Starting state deter shape: {starting_state['deter'].shape}")
                logger.debug(f"train_step: Starting state stoch shape: {starting_state['stoch'].shape if 'stoch' in starting_state else 'N/A'}")
            with open(log_file, "a") as f:
                f.write(f"Full starting_state: {starting_state}\n")
                f.write(f"train_step: Starting state deter shape: {starting_state['deter'].shape}\n")
                f.write(f"train_step: Starting state stoch shape: {starting_state['stoch'].shape if 'stoch' in starting_state else 'N/A'}\n")

            if "logits" not in starting_state and self.world_model.dynamics.use_discrete_latents:
                starting_state["logits"] = torch.zeros(
                    batch_size, self.world_model.dynamics.discrete_latent_num, 
                    self.world_model.dynamics.discrete_latent_size, device=self.device
                )
            if "stoch" not in starting_state:
                if self.world_model.dynamics.use_discrete_latents:
                    logits = starting_state["logits"]
                    if logits.dim() == 3:
                        logits = logits[:, -1] if logits.shape[1] > 1 else logits.squeeze(1)
                    indices = torch.argmax(logits, dim=-1)
                    onehot = F.one_hot(indices, num_classes=self.world_model.dynamics.discrete_latent_size).float()
                    starting_state["stoch"] = onehot.view(batch_size, -1)
                else:
                    starting_state["stoch"] = torch.zeros(
                        batch_size, self.world_model.dynamics.stochastic_dimension, device=self.device
                    )
            
            if self.enable_debugging:
                logger.debug(f"After ensuring logits/stoch: {starting_state}")
            with open(log_file, "a") as f:
                f.write(f"After ensuring logits/stoch: {starting_state}\n")
            
            if starting_state["deter"].dim() == 3:  # [B, T, hidden_units]
                hidden_state = starting_state["deter"][:, -1, :].unsqueeze(0)  # [1, B, hidden_units]
            else:
                hidden_state = starting_state["deter"].unsqueeze(0) if starting_state["deter"].dim() == 2 else starting_state["deter"]
            
            starting_state = {
                "hidden_state": hidden_state,
                "stoch": starting_state["stoch"][:, -1] if starting_state["stoch"].dim() == 3 else starting_state["stoch"],
                "deter": starting_state["deter"][:, -1, :] if starting_state["deter"].dim() == 3 else starting_state["deter"]
            }
            
            if self.enable_debugging:
                logger.debug(f"train_step: Normalized hidden_state shape: {starting_state['hidden_state'].shape}")
                logger.debug(f"train_step: Normalized stoch shape: {starting_state['stoch'].shape}")
                logger.debug(f"train_step: Normalized deter shape: {starting_state['deter'].shape}")
            with open(log_file, "a") as f:
                f.write(f"train_step: Normalized hidden_state shape: {starting_state['hidden_state'].shape}\n")
                f.write(f"train_step: Normalized stoch shape: {starting_state['stoch'].shape}\n")
                f.write(f"train_step: Normalized deter shape: {starting_state['deter'].shape}\n")
            
            # Imagine trajectory once, detaching for value computation later
            imagined_features, imagined_state, imagined_actions, imagined_action_indices = imagine_trajectory(
                self.world_model, batch_size, imagination_horizon, 
                self.world_model.action_space.n if self.has_discrete_actions else self.world_model.action_space.shape[0],
                starting_state, self.actor_network,
                use_gradients=(self.imagination_gradient_type == "dynamics" and not self.has_discrete_actions)
            )

            if self.enable_debugging:
                logger.debug(f"train_step: Post-imagine_trajectory: Imagined features shape: {imagined_features.shape}")
                logger.debug(f"train_step: Post-imagine_trajectory: Imagined actions shape: {imagined_actions.shape}")
                logger.debug(f"train_step: Post-imagine_trajectory: Imagined action indices shape: {imagined_action_indices.shape}")
                logger.debug(f"train_step: Post-imagine_trajectory: Imagined state deter shape: {imagined_state['deter'].shape}")
                logger.debug(f"train_step: Post-imagine_trajectory: Imagined state stoch shape: {imagined_state['stoch'].shape}")
            with open(log_file, "a") as f:
                f.write(f"train_step: Post-imagine_trajectory: Imagined features shape: {imagined_features.shape}\n")
                f.write(f"train_step: Post-imagine_trajectory: Imagined actions shape: {imagined_actions.shape}\n")
                f.write(f"train_step: Post-imagine_trajectory: Imagined action indices shape: {imagined_action_indices.shape}\n")
                f.write(f"train_step: Post-imagine_trajectory: Imagined state deter shape: {imagined_state['deter'].shape}\n")
                f.write(f"train_step: Post-imagine_trajectory: Imagined state stoch shape: {imagined_state['stoch'].shape}\n")

            with torch.amp.autocast('cuda', enabled=self.use_automatic_mixed_precision):
                # Compute shared predictions from world model
                reward_distribution = self.world_model.heads["reward"](imagined_features)
                reward = self.compute_expected_value(reward_distribution)
                continuation_distribution = self.world_model.heads["continuation"](imagined_features)
                discount = continuation_distribution.mode() * getattr(self.configuration, "discount_factor", 0.997)
                
                # Compute world model losses
                reward_bins = torch.linspace(-20.0, 20.0, 255, device=self.device)
                reward_target = torch.bucketize(reward.detach(), reward_bins)
                reward_loss = F.cross_entropy(reward_distribution.logits.view(-1, 255), reward_target.view(-1))
                
                continuation_logits = continuation_distribution.logits  # [B*T, 1]
                continuation_target = (discount > 0).float()  # [B*T, 1], binary target
                continuation_loss = F.binary_cross_entropy_with_logits(continuation_logits, continuation_target)
                
                metrics.update({
                    "reward_loss": reward_loss.item(),
                    "continuation_loss": continuation_loss.item()
                })
                logger.info(f"World Model Losses - Reward: {reward_loss.item():.4f}, Continuation: {continuation_loss.item():.4f}")

                # Value computation (detach imagined_features to isolate graph)
                value_features = imagined_features.detach()
                value_distribution = self.value_network(value_features)
                value = self.compute_expected_value(value_distribution)
                
                if self.use_slow_target_network:
                    slow_value_distribution = self.slow_value_network(value_features)
                    slow_value = self.compute_expected_value(slow_value_distribution)
                else:
                    slow_value = value
                
                reward = reward.unsqueeze(-1)
                discount = discount.unsqueeze(-1)
                value = value.unsqueeze(-1)
                slow_value = slow_value.unsqueeze(-1)

                if self.enable_debugging:
                    logger.debug(f"Reward shape: {reward.shape}, Value shape: {value.shape}, Slow Value shape: {slow_value.shape}, Discount shape: {discount.shape}")
                with open(log_file, "a") as f:
                    f.write(f"Reward shape: {reward.shape}, Value shape: {value.shape}, Slow Value shape: {slow_value.shape}, Discount shape: {discount.shape}\n")

                reward_t = reward.transpose(0, 1)
                slow_value_t = slow_value.transpose(0, 1)
                discount_t = discount.transpose(0, 1)

                lambda_discount_factor = getattr(self.configuration, "discount_lambda", 0.95)
                return_targets, weights, value_baseline = lambda_return_target(
                    reward_t, slow_value_t, discount_t, lambda_discount_factor
                )
                return_targets = [target.transpose(0, 1) for target in return_targets]

                if self.enable_debugging:
                    logger.debug(f"Return targets length: {len(return_targets)}, First target shape: {return_targets[0].shape}")
                with open(log_file, "a") as f:
                    f.write(f"Return targets length: {len(return_targets)}, First target shape: {return_targets[0].shape}\n")

                # Actor loss computation
                imagined_features_for_actor = imagined_features if self.imagination_gradient_type == "dynamics" else imagined_features.detach()
                actor_distribution = self.actor_network(imagined_features_for_actor)

                if self.enable_debugging:
                    logger.debug(f"train_step: Actor distribution shape: {actor_distribution.distribution.logits.shape}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: Actor distribution shape: {actor_distribution.distribution.logits.shape}\n")

                if self.has_discrete_actions:
                    action_log_probabilities = actor_distribution.log_probability(imagined_action_indices)
                else:
                    action_log_probabilities = actor_distribution.log_probability(imagined_actions)

                if self.enable_debugging:
                    logger.debug(f"train_step: Action log probabilities shape: {action_log_probabilities.shape}")
                    logger.debug(f"train_step: Action log probabilities sample: {action_log_probabilities[:2, :2]}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: Action log probabilities shape: {action_log_probabilities.shape}\n")
                    f.write(f"train_step: Action log probabilities sample: {action_log_probabilities[:2, :2]}\n")

                advantages = torch.stack([target - slow_val.squeeze(-1) for target, slow_val in zip(return_targets, slow_value.split(1, dim=0))], dim=0)
                advantages = advantages.transpose(0, 1)

                if self.enable_debugging:
                    logger.debug(f"train_step: Advantages shape: {advantages.shape}")
                    logger.debug(f"train_step: Advantages sample: {advantages[:2, :2]}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: Advantages shape: {advantages.shape}\n")
                    f.write(f"train_step: Advantages sample: {advantages[:2, :2]}\n")

                if advantages.numel() > 1:
                    return_scale = self.compute_return_scale(advantages)
                    large_advantage_mask = torch.abs(advantages) > self.minimum_return_scale
                    advantages[large_advantage_mask] = advantages[large_advantage_mask] / return_scale
                    metrics["return_scale"] = return_scale
                    if self.enable_debugging:
                        logger.debug(f"train_step: Return scale: {return_scale}")
                        logger.debug(f"train_step: Scaled advantages shape: {advantages.shape}")
                        logger.debug(f"train_step: Scaled advantages sample: {advantages[:2, :2]}")
                    with open(log_file, "a") as f:
                        f.write(f"train_step: Return scale: {return_scale}\n")
                        f.write(f"train_step: Scaled advantages shape: {advantages.shape}\n")
                        f.write(f"train_step: Scaled advantages sample: {advantages[:2, :2]}\n")

                entropy = actor_distribution.entropy()
                entropy_coefficient = getattr(self.configuration, "actor_entropy", 0.01)

                if self.enable_debugging:
                    logger.debug(f"train_step: Entropy shape: {entropy.shape}")
                    logger.debug(f"train_step: Entropy sample: {entropy[:2, :2]}")
                    logger.debug(f"train_step: Entropy coefficient: {entropy_coefficient}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: Entropy shape: {entropy.shape}\n")
                    f.write(f"train_step: Entropy sample: {entropy[:2, :2]}\n")
                    f.write(f"train_step: Entropy coefficient: {entropy_coefficient}\n")

                actor_loss = -(action_log_probabilities * advantages.detach()).mean() - entropy_coefficient * entropy.mean()
                
                if self.enable_debugging:
                    logger.debug(f"train_step: Actor loss: {actor_loss.item()}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: Actor loss: {actor_loss.item()}\n")

                if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                    logger.warning(f"Invalid actor loss: {actor_loss.item()}")
                    with open(log_file, "a") as f:
                        f.write(f"WARNING: Invalid actor loss: {actor_loss.item()}\n")
                    actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                # Optimize actor
                actor_metrics = self.actor_optimizer(actor_loss, self.actor_network.parameters())
                metrics.update(actor_metrics)
                metrics.update(self.check_gradients("actor", self.actor_network))
                metrics.update({
                    "actor_loss": actor_loss.item(),
                    "policy_entropy": entropy.mean().item(),
                    "advantage_mean": advantages.mean().item(),
                    "advantage_std": advantages.std().item()
                })
                logger.info(f"Actor Metrics - Loss: {actor_loss.item():.4f}, Entropy: {entropy.mean().item():.4f}, Advantage Mean: {advantages.mean().item():.4f}")

                # Value loss computation
                value_logits = value_distribution.logits  # [B, T, 255]
                num_values = value_logits.shape[-1]  # 255
                returns = torch.stack([target.squeeze(-1) for target in return_targets], dim=1)  # [B, T, K]
                returns = returns.mean(dim=2)  # [B, T], average over K targets
                min_val, max_val = -20.0, 20.0
                bins = torch.linspace(min_val, max_val, num_values, device=self.device)
                target_indices = torch.bucketize(returns, bins)  # [B, T]
                target_twohot = F.one_hot(target_indices, num_classes=num_values).float()  # [B, T, 255]
                
                if self.enable_debugging:
                    logger.debug(f"train_step: returns shape after mean: {returns.shape}")
                    logger.debug(f"train_step: target_indices shape: {target_indices.shape}")
                    logger.debug(f"train_step: value_logits shape: {value_logits.shape}")
                    logger.debug(f"train_step: target_twohot shape: {target_twohot.shape}")
                with open(log_file, "a") as f:
                    f.write(f"train_step: returns shape after mean: {returns.shape}\n")
                    f.write(f"train_step: target_indices shape: {target_indices.shape}\n")
                    f.write(f"train_step: value_logits shape: {value_logits.shape}\n")
                    f.write(f"train_step: target_twohot shape: {target_twohot.shape}\n")
                
                value_loss = -torch.sum(target_twohot * F.log_softmax(value_logits, dim=-1), dim=-1).mean()
                
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    logger.warning(f"Invalid value loss: {value_loss.item()}")
                    with open(log_file, "a") as f:
                        f.write(f"WARNING: Invalid value loss: {value_loss.item()}\n")
                    value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                value_metrics = self.value_optimizer(value_loss, self.value_network.parameters())
                metrics.update(value_metrics)
                metrics.update(self.check_gradients("value", self.value_network))
                metrics.update({"value_loss": value_loss.item()})
                logger.info(f"Value Metrics - Loss: {value_loss.item():.4f}")
                
                self.update_slow_target_network()
                
                # Log all metrics at the end of the step
                logger.info(f"Train Step Combined Metrics: {metrics}")
                
                return imagined_features, imagined_state, imagined_actions, weights, metrics
            
        except Exception as error:
            error_msg = f"Exception in behavior training step: {str(error)}\nTraceback: {traceback.format_exc()}"
            logger.error(error_msg)
            with open(log_file, "a") as f:
                f.write(f"ERROR: {error_msg}\n")
            return None, None, None, None, {"behavior_exception": 1.0, "error": str(error)}