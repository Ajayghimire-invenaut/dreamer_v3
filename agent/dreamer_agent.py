import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import logging
from agent.behavior import ImaginedBehavior
from agent.random_explorer import RandomExplorer
from agent.world_model import WorldModel

# Set up logging for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

class DreamerAgent(nn.Module):
    """
    Main agent class for DreamerV3, coordinating the world model, policy learning, and environment interaction.
    Learns a world model and policy through imagination-based training.
    """
    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        configuration: Any,
        logger_object: Any,
        dataset: Any
    ) -> None:
        super(DreamerAgent, self).__init__()
        self.configuration = configuration
        self.logger = logger_object
        self.dataset = dataset
        self.action_space = action_space
        self.enable_debugging = getattr(configuration, "debug", False)
        self.device = torch.device(configuration.computation_device if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.logging_interval = getattr(configuration, "logging_interval", 100)
        self.training_updates_per_step = getattr(configuration, "training_updates_per_step", 8)
        self.train_ratio = getattr(configuration, "train_ratio", 512)
        self.exploration_termination_step = getattr(configuration, "exploration_termination_step", 50000) // getattr(configuration, "action_repeat", 1)

        # Tracking metrics and steps
        self.metrics: Dict[str, List] = {}
        self.current_step = logger_object.global_step // getattr(configuration, "action_repeat", 1)
        self.latest_losses = {}

        # Initialize core components
        self.world_model = WorldModel(observation_space, action_space, configuration)
        self.task_behavior = ImaginedBehavior(configuration, self.world_model)
        
        # Set feature_dimension as a regular attribute
        self.feature_dimension = self.world_model.dynamics.feature_dimension

        # Move models to the appropriate device
        if getattr(configuration, "compile_models", False) and torch.cuda.is_available() and getattr(configuration, "os_name", "posix") != "nt":
            try:
                self.world_model = torch.compile(self.world_model)
                self.task_behavior = torch.compile(self.task_behavior)
                if self.enable_debugging:
                    self.logger.info("[INFO] Successfully compiled world model and task behavior")
            except Exception as error:
                self.logger.warning(f"[WARNING] Failed to compile models: {error}")
        self.to(self.device)

        # Set up exploration policy
        self.explorer = RandomExplorer(configuration, action_space).to(self.device)
        exploration_behavior = getattr(configuration, "actor_exploration_behavior", "random")
        if exploration_behavior != "greedy":
            behavior_options = {
                "greedy": lambda: self.task_behavior,
                "random": lambda: RandomExplorer(configuration, action_space),
                "plan2explore": lambda: self.task_behavior  # Placeholder for Plan2Explore
            }
            self.explorer = behavior_options.get(exploration_behavior, lambda: RandomExplorer(configuration, action_space))().to(self.device)
        if self.enable_debugging:
            self.logger.debug(f"[DEBUG DreamerAgent] Initialized explorer with type: {type(self.explorer).__name__}")

    def forward(
        self,
        observation: Dict[str, Any],
        state: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute policy actions and update state, performing training updates if in training mode."""
        policy_output, updated_state = self.compute_policy(observation, state, training)

        # Perform training updates during training if dataset is available
        if training and self.dataset:
            metrics_sum = {}
            try:
                total_updates = self.train_ratio
                for _ in range(total_updates):
                    batch = next(self.dataset)
                    metrics = self.train(batch)
                    if metrics is None:
                        continue
                    for key, value in metrics.items():
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                
                if total_updates > 0 and metrics_sum:
                    metrics_average = {key: value / total_updates for key, value in metrics_sum.items()}
                    if self.enable_debugging:
                        self.latest_losses.update(metrics_average)
                        self.logger.debug(f"Training metrics at step {self.current_step}: {metrics_average}")
            except Exception as error:
                if self.enable_debugging:
                    self.logger.debug(f"Error during training updates: {error}")
            self.current_step += 1

        return policy_output, updated_state

    def compute_policy(
        self,
        observation: Dict[str, Any],
        state: Optional[Dict[str, torch.Tensor]],
        training: bool
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        """Compute policy actions from observations and update the latent state."""
        if observation is None:
            if self.enable_debugging:
                self.logger.debug("Received None observation in compute_policy")
            observation = {
                "image": np.zeros((64, 64, 3), dtype=np.uint8),
                "is_first": np.array(True),
                "is_terminal": np.array(False),
                "reward": np.array(0.0)
            }
        
        # Preprocess observations with symlog for non-image data
        try:
            processed_observation = self.world_model.preprocess(observation)
            # Apply symlog transformation to non-image data if present (e.g., state vectors)
            if "state" in processed_observation:  # Assuming state vector might be included
                processed_observation["state"] = torch.sign(processed_observation["state"]) * torch.log1p(torch.abs(processed_observation["state"]))
        except Exception as error:
            if self.enable_debugging:
                self.logger.debug(f"Error in preprocess: {error}")
            device = next(self.parameters()).device
            processed_observation = {
                "image": torch.zeros((1, 1, 3, 64, 64), device=device),
                "is_first": torch.tensor([True], dtype=torch.bool, device=device).unsqueeze(0),
                "is_terminal": torch.tensor([False], dtype=torch.bool, device=device).unsqueeze(0),
                "reward": torch.tensor([0.0], device=device).unsqueeze(0),
                "discount": torch.tensor([1.0], device=device).unsqueeze(0),
                "continuation": torch.tensor([1.0], device=device).unsqueeze(0)
            }
        
        # Removed torch.no_grad() to enable encoder training
        try:
            embedding = self.world_model.encoder(processed_observation)
            if embedding is None:
                if self.enable_debugging:
                    logger.debug("Encoder returned None embedding")
                device = next(self.parameters()).device
                embedding = torch.zeros((1, self.world_model.embedding_dimension), device=device)
            
            if embedding.dim() == 3:
                embedding = embedding.squeeze(1)  # [B, 1, D] -> [B, D]
            elif embedding.dim() == 2:
                embedding = embedding.unsqueeze(0)  # [D] -> [1, D]

            is_first = processed_observation["is_first"]
            if is_first.dim() == 1:
                is_first = is_first.unsqueeze(0).unsqueeze(-1)  # [B] -> [1, B, 1]
            elif is_first.dim() == 2:
                is_first = is_first.unsqueeze(-1)  # [B, T] -> [B, T, 1]

            previous_action = state.get("action") if state else None
            if previous_action is not None and previous_action.dim() == 2:
                previous_action = previous_action.unsqueeze(0)  # [B, A] -> [1, B, A]

            latent_state, posterior, _ = self.world_model.dynamics.observe_step(embedding, previous_action, is_first, state)
            features = self.world_model.dynamics.get_features(latent_state)
            if features.dim() == 3:
                features = features[-1]  # [T, B, D] -> [B, D]
            
            features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            if self.enable_debugging:
                logger.debug(f"Features shape after dynamics: {features.shape}")
        except Exception as error:
            if self.enable_debugging:
                self.logger.debug(f"Error in encoder or dynamics: {error}")
                import traceback
                self.logger.debug(traceback.format_exc())
            device = next(self.parameters()).device
            features = torch.zeros((1, self.feature_dimension), device=device)
            latent_state = {
                "deter": torch.zeros((1, self.world_model.dynamics.hidden_units), device=device),
                "stoch": torch.zeros((1, self.world_model.dynamics.stochastic_dimension), device=device)
            } if state is None else state

        # Select action based on training mode
        try:
            actor = self.explorer if training and self.current_step < self.exploration_termination_step else self.task_behavior
            actor_distribution = actor(features)
            action = actor_distribution.sample() if training else actor_distribution.mode()

            if hasattr(self.action_space, 'n'):
                action_output = action if action.dim() == 2 else F.one_hot(action.long(), num_classes=self.action_space.n).float()
                action_indices_for_log_prob = torch.argmax(action_output, dim=-1) if action_output.dim() > 1 else action
            else:
                action_output = action
                action_indices_for_log_prob = action
            
            log_probability = actor_distribution.log_probability(action_indices_for_log_prob)
        except Exception as error:
            if self.enable_debugging:
                self.logger.debug(f"Error in action selection: {error}")
                import traceback
                self.logger.debug(traceback.format_exc())
            device = next(self.parameters()).device
            if hasattr(self.action_space, 'n'):
                action_output = F.one_hot(torch.randint(0, self.action_space.n, (1,), device=device), num_classes=self.action_space.n).float()
            else:
                action_output = torch.zeros(self.action_space.shape, device=device)
            log_probability = torch.tensor(-float('inf'), device=device)

        # Detach outputs for inference
        latent_state = {key: value.detach() for key, value in latent_state.items()}
        action_output = action_output.detach()
        log_probability = log_probability.detach() if log_probability is not None else None
        
        policy_output = {"action": action_output, "log_probability": log_probability}
        latent_state["action"] = action_output

        if self.enable_debugging:
            self.logger.debug(f"Observation keys: {list(observation.keys())}")
            self.logger.debug(f"State: {state if state is None else {k: v.shape for k, v in state.items()}}")
            self.logger.debug(f"Policy output action shape: {policy_output['action'].shape}")
            if hasattr(actor_distribution, 'entropy'):
                try:
                    entropy = actor_distribution.entropy().mean().item()
                    self.logger.debug(f"Policy entropy: {entropy:.4f}")
                except:
                    pass

        return policy_output, latent_state

    def train(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Train the world model and behavior using a batch of data."""
        if batch_data is None or not isinstance(batch_data, dict):
            logger.error(f"Batch data is invalid type: {type(batch_data).__name__}")
            return {"batch_data_error": 1.0}
                    
        try:
            if self.enable_debugging:
                logger.debug(f"Batch keys: {list(batch_data.keys())}")
                if "image" in batch_data:
                    image_data = batch_data["image"]
                    if isinstance(image_data, np.ndarray):
                        logger.debug(f"Image shape: {image_data.shape}, min: {image_data.min()}, max: {image_data.max()}")
                    elif isinstance(image_data, torch.Tensor):
                        logger.debug(f"Image shape: {image_data.shape}, min: {image_data.min().item()}, max: {image_data.max().item()}")

            device = next(self.parameters()).device
            tensor_batch = {}
            for key, value in batch_data.items():
                if isinstance(value, np.ndarray):
                    tensor_batch[key] = torch.tensor(value, device=device)
                elif isinstance(value, torch.Tensor):
                    tensor_batch[key] = value.to(device)
                else:
                    tensor_batch[key] = value
            
            # Adjust batch and sequence dimensions with enhanced logging
            if "image" in tensor_batch:
                batch_size = tensor_batch["image"].shape[0]
                seq_len = tensor_batch["image"].shape[1] if tensor_batch["image"].dim() > 4 else 1
                if tensor_batch["image"].dim() == 4:
                    tensor_batch["image"] = tensor_batch["image"].unsqueeze(1)
                    seq_len = 1
                for key in tensor_batch:
                    if isinstance(tensor_batch[key], torch.Tensor):
                        if tensor_batch[key].dim() >= 3 and tensor_batch[key].shape[0] != batch_size:
                            tensor_batch[key] = tensor_batch[key].view(batch_size, seq_len, *tensor_batch[key].shape[1:])
                        elif tensor_batch[key].dim() == 2 and tensor_batch[key].shape[0] == batch_size:
                            tensor_batch[key] = tensor_batch[key].unsqueeze(1)
                    if self.enable_debugging:
                        logger.debug(f"Adjusted {key} shape: {tensor_batch[key].shape if isinstance(tensor_batch[key], torch.Tensor) else 'N/A'}")
            batch_data = tensor_batch
            
            # Train world model
            try:
                posterior_state, context, world_model_metrics = self.world_model.train_step(batch_data)
                logger.info(f"World Model Metrics: {world_model_metrics}")
                
                if all(value == 0.0 for key, value in world_model_metrics.items() if key.endswith('_loss')):
                    logger.warning("All world model losses are zero! Possible training issue.")
                    
                if not posterior_state or "deter" not in posterior_state or posterior_state["deter"] is None:
                    logger.error("Invalid state from world model")
                    return {"invalid_posterior_error": 1.0}
                            
                # Train behavior
                try:
                    imagined_features, imagined_state, imagined_actions, weights, behavior_metrics = self.task_behavior.train_step(posterior_state)
                    logger.info(f"Behavior Metrics: {behavior_metrics}")
                    
                    if all(value == 0.0 for key, value in behavior_metrics.items() if key.endswith('_loss')):
                        logger.warning("All behavior losses are zero! Possible training issue.")
                    
                    # Combine metrics
                    metrics = {**world_model_metrics, **behavior_metrics}
                    
                    # Add gradient norms for key components every 50 steps
                    if self.current_step % 50 == 0:
                        for name, module in [('world_model', self.world_model), ('task_behavior.actor_network', self.task_behavior.actor_network), ('task_behavior.value_network', self.task_behavior.value_network)]:
                            for param_name, param in module.named_parameters():
                                if param.grad is not None:
                                    grad_norm = param.grad.norm().item()
                                    metrics[f"{name}.{param_name}.gradient_norm"] = grad_norm
                                    logger.debug(f"Gradient norm for {name}.{param_name}: {grad_norm:.4f}")
                                else:
                                    metrics[f"{name}.{param_name}.gradient_none"] = 1.0
                                    logger.debug(f"No gradient for {name}.{param_name}")
                    
                    # Log combined metrics
                    logger.info(f"Combined Training Metrics at Step {self.current_step}: {metrics}")
                    
                    return metrics
                except Exception as error:
                    logger.error(f"Error in behavior training step: {error}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"behavior_exception": 1.0, "error": str(error)}
                            
            except Exception as error:
                logger.error(f"Error in world model training step: {error}")
                import traceback
                logger.error(traceback.format_exc())
                return {"world_model_exception": 1.0, "error": str(error)}
                        
        except Exception as error:
            logger.error(f"Error in training: {error}")
            import traceback
            logger.error(traceback.format_exc())
            return {"training_exception": 1.0, "error": str(error)}

    def train_step(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """Public method to train the agent on a batch of data."""
        metrics = self.train(batch_data)
        
        if not metrics or not isinstance(metrics, dict):
            return {"train_error": 1.0}
            
        numeric_metrics = {}
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)):
                    numeric_metrics[key] = float(value)
                elif isinstance(value, torch.Tensor):
                    numeric_metrics[key] = value.item()
                elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                    numeric_metrics[key] = float(value)
            except:
                pass
                
        if not numeric_metrics:
            numeric_metrics = {
                "actor_loss": 0.0,
                "value_loss": 0.0,
                "world_model_loss": 0.0,
                "kl_loss": 0.0
            }
            
        return numeric_metrics

    def collect_optimizer_states(self) -> Dict[str, Any]:
        """Collect optimizer states for checkpointing."""
        optimizer_states = {}
        
        if hasattr(self.world_model, 'model_optimizer') and self.world_model.model_optimizer is not None:
            optimizer_states['world_model'] = self.world_model.model_optimizer.state_dict()
        
        if hasattr(self.task_behavior, 'actor_optimizer') and self.task_behavior.actor_optimizer is not None:
            optimizer_states['actor'] = self.task_behavior.actor_optimizer.state_dict()
        
        if hasattr(self.task_behavior, 'value_optimizer') and self.task_behavior.value_optimizer is not None:
            optimizer_states['value'] = self.task_behavior.value_optimizer.state_dict()
        
        return optimizer_states