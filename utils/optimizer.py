import torch
import torch.nn.utils as torch_utils
from typing import Any, Dict, Iterator, Optional, List
import logging

# Setup logger for debugging and information
logger = logging.getLogger(__name__)

class Optimizer:
    """
    Optimizer wrapper for DreamerV3 that provides a unified interface.
    Supports gradient clipping, mixed precision, warmup, and learning rate scheduling.
    """
    def __init__(
        self,
        name: str,
        parameters: Iterator[torch.nn.Parameter],
        learning_rate: float,
        epsilon: float = 1e-8,
        gradient_clip: float = 1000.0,  # Updated to 1000 per DreamerV3 recommendation
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        use_automatic_mixed_precision: bool = False,
        warmup_steps: int = 0
    ) -> None:
        """
        Initialize the optimizer wrapper with specified settings.
        
        Args:
            name: Identifier for the optimizer (e.g., 'actor', 'value', 'world_model')
            parameters: Iterable of model parameters to optimize
            learning_rate: Initial learning rate
            epsilon: Small value for numerical stability
            gradient_clip: Maximum gradient norm for clipping (default 1000 per DreamerV3)
            weight_decay: Weight decay coefficient for regularization
            optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
            use_automatic_mixed_precision: Enable mixed precision training if True
            warmup_steps: Number of steps for learning rate warmup
        """
        self.name = name
        self.parameters = list(parameters)
        if not self.parameters:
            logger.warning(f"{self.name} optimizer initialized with no parameters")
        self.gradient_clip = float(gradient_clip)
        self.weight_decay = float(weight_decay)
        self.use_automatic_mixed_precision = use_automatic_mixed_precision and torch.cuda.is_available()
        
        self.current_learning_rate = float(learning_rate)
        self.base_learning_rate = float(learning_rate)
        
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Initialize the underlying PyTorch optimizer
        if optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters,
                lr=self.current_learning_rate,
                eps=float(epsilon),
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters,
                lr=self.current_learning_rate,
                eps=float(epsilon),
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters,
                lr=self.current_learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Initialize gradient scaler for mixed precision
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_automatic_mixed_precision) if self.use_automatic_mixed_precision else None
        
        logger.debug(f"Created {name} optimizer with learning_rate={self.current_learning_rate}, "
                     f"epsilon={epsilon}, gradient_clip={self.gradient_clip}, "
                     f"mixed_precision={self.use_automatic_mixed_precision}")

    def __call__(self, loss: torch.Tensor, parameters_to_clip: Iterator[torch.nn.Parameter], retain_graph: bool = False) -> Dict[str, float]:
        metrics = {f"{self.name}_loss": loss.detach().cpu().item()}
        
        if torch.isnan(loss) or torch.isinf(loss):
            metrics[f"{self.name}_invalid_loss"] = 1.0
            logger.warning(f"{self.name} optimizer received invalid loss: {loss.item()}")
            return metrics
        
        if loss.item() == 0.0:
            metrics[f"{self.name}_zero_loss"] = 1.0
            logger.debug(f"{self.name} optimizer received zero loss")
        
        parameter_list = list(parameters_to_clip)
        debug_parameters = {}
        if parameter_list and self.step_count % 50 == 0:
            for i, param in enumerate(parameter_list[:3]):
                if param.requires_grad:
                    debug_parameters[f"param_{i}"] = param.detach().clone()
        
        if self.warmup_steps > 0:
            self._update_learning_rate()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.use_automatic_mixed_precision and self.gradient_scaler is not None:
            self.gradient_scaler.scale(loss).backward(retain_graph=retain_graph)
            self.gradient_scaler.unscale_(self.optimizer)
        else:
            loss.backward(retain_graph=retain_graph)
        
        parameters_with_gradients = [param for param in parameter_list if param.grad is not None]
        if parameters_with_gradients:
            gradient_norms = [torch.norm(param.grad.detach()) for param in parameters_with_gradients]
            total_gradient_norm = torch.norm(torch.stack(gradient_norms))
            metrics[f"{self.name}_gradient_norm"] = total_gradient_norm.item()
            
            if any(torch.isnan(param.grad).any() or torch.isinf(param.grad).any() for param in parameters_with_gradients):
                metrics[f"{self.name}_bad_gradients"] = 1.0
                logger.warning(f"{self.name} optimizer detected NaN/Inf gradients")
            
            zero_gradient_count = sum(1 for param in parameters_with_gradients if param.grad.abs().sum().item() == 0)
            if zero_gradient_count > 0:
                metrics[f"{self.name}_zero_gradients"] = zero_gradient_count
            
            if self.gradient_clip > 0:
                torch_utils.clip_grad_norm_(parameters_with_gradients, max_norm=self.gradient_clip)
                post_clip_norm = torch.norm(torch.stack([torch.norm(param.grad.detach()) for param in parameters_with_gradients]))
                metrics[f"{self.name}_clipped_gradient_norm"] = post_clip_norm.item()
                if total_gradient_norm > self.gradient_clip:
                    metrics[f"{self.name}_clip_ratio"] = (post_clip_norm / total_gradient_norm).item()
        else:
            metrics[f"{self.name}_no_gradients"] = 1.0
            logger.debug(f"{self.name} optimizer: No gradients computed")
        
        if self.use_automatic_mixed_precision and self.gradient_scaler is not None:
            self.gradient_scaler.step(self.optimizer)
            self.gradient_scaler.update()
        else:
            self.optimizer.step()
        
        if debug_parameters:
            for i, param_key in enumerate(debug_parameters.keys()):
                if i < len(parameter_list) and parameter_list[i].requires_grad:
                    param_after = parameter_list[i].detach()
                    param_before = debug_parameters[param_key]
                    if torch.allclose(param_before, param_after, rtol=1e-5, atol=1e-8):
                        metrics[f"{self.name}_no_update_{i}"] = 1.0
                    else:
                        change_magnitude = torch.norm(param_after - param_before) / torch.norm(param_before) if torch.norm(param_before) > 0 else 0
                        if isinstance(change_magnitude, torch.Tensor):
                            metrics[f"{self.name}_parameter_change_{i}"] = change_magnitude.item()
                        else:
                            metrics[f"{self.name}_parameter_change_{i}"] = float(change_magnitude)
        
        self.step_count += 1
        return metrics
    
    def _update_learning_rate(self) -> None:
        """Update the learning rate during the warmup period."""
        if self.step_count >= self.warmup_steps:
            new_learning_rate = self.base_learning_rate
        else:
            progress = self.step_count / self.warmup_steps
            new_learning_rate = self.base_learning_rate * progress
        
        if abs(new_learning_rate - self.current_learning_rate) > 1e-6:
            self.current_learning_rate = new_learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_learning_rate
            logger.debug(f"{self.name} optimizer updated learning rate to {self.current_learning_rate}")

    def state_dict(self) -> Dict:
        """
        Retrieve the optimizer state for checkpointing.
        
        Returns:
            Dictionary containing optimizer and scaler states
        """
        state_dictionary = {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "learning_rate": self.current_learning_rate
        }
        if self.use_automatic_mixed_precision and self.gradient_scaler is not None:
            state_dictionary["scaler"] = self.gradient_scaler.state_dict()
        return state_dictionary

    def load_state_dict(self, state_dictionary: Dict) -> None:
        """
        Load the optimizer state from a checkpoint.
        
        Args:
            state_dictionary: Dictionary with saved optimizer and scaler states
        """
        self.optimizer.load_state_dict(state_dictionary["optimizer"])
        self.step_count = state_dictionary.get("step_count", 0)
        self.current_learning_rate = state_dictionary.get("learning_rate", self.base_learning_rate)
        
        if self.use_automatic_mixed_precision and self.gradient_scaler is not None and "scaler" in state_dictionary:
            self.gradient_scaler.load_state_dict(state_dictionary["scaler"])
        logger.debug(f"{self.name} optimizer loaded state with step_count={self.step_count}")

    def get_learning_rate(self) -> float:
        """
        Get the current learning rate.
        
        Returns:
            Current learning rate value
        """
        return self.current_learning_rate

    def add_learning_rate_scheduler(self, scheduler_type: str = 'cosine', **kwargs: Any) -> None:
        """
        Add a learning rate scheduler to the optimizer.
        
        Args:
            scheduler_type: Type of scheduler ('cosine', 'step', or 'plateau')
            **kwargs: Additional arguments for the scheduler
        """
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('max_steps', 10000),
                eta_min=kwargs.get('minimum_learning_rate', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 1000),
                gamma=kwargs.get('decay_factor', 0.5)
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('reduction_factor', 0.5),
                patience=kwargs.get('patience', 5),
                min_lr=kwargs.get('minimum_learning_rate', 1e-6)
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            self.scheduler = None
            return
            
        logger.info(f"Added {scheduler_type} scheduler to {self.name} optimizer")

    def step_scheduler(self, validation_loss: Optional[float] = None) -> None:
        """
        Step the learning rate scheduler, if present.
        
        Args:
            validation_loss: Validation loss value (required for ReduceLROnPlateau scheduler)
        """
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if validation_loss is not None:
                    self.scheduler.step(validation_loss)
                else:
                    logger.debug(f"{self.name} optimizer: Validation loss required for ReduceLROnPlateau but not provided")
            else:
                self.scheduler.step()
                
            for i, param_group in enumerate(self.optimizer.param_groups):
                logger.debug(f"{self.name} optimizer group {i} learning_rate: {param_group['lr']}")