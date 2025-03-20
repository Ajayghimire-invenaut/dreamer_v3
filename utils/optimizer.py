import torch
import torch.nn.utils as torch_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, Optional

class Optimizer:
    def __init__(self,
                 name: str,
                 parameters: Any,
                 learning_rate: float,
                 eps: float = 1e-8,  # Default for Adam/AdamW stability
                 clip: Optional[float] = None,
                 weight_decay: Optional[float] = 0.0,
                 opt: str = "adamw",
                 use_amp: bool = False,
                 total_steps: Optional[int] = None,
                 debug: bool = False) -> None:
        """
        Initialize the Optimizer with specified parameters.
        
        :param name: Identifier for the optimizer (e.g., "world_model").
        :param parameters: Model parameters to optimize.
        :param learning_rate: Initial learning rate.
        :param eps: Epsilon for numerical stability in Adam/AdamW.
        :param clip: Gradient clipping norm (None to disable).
        :param weight_decay: Weight decay coefficient (0.0 to disable).
        :param opt: Optimizer type ("adamw", "adam", or "sgd").
        :param use_amp: Enable automatic mixed precision.
        :param total_steps: Total steps for cosine annealing scheduler (None to disable).
        :param debug: Enable debug logging.
        """
        self.name = name
        self.parameters = parameters
        self.clip = clip
        self.weight_decay = weight_decay if weight_decay is not None else 0.0
        self.use_amp = use_amp
        self.debug = debug

        # Initialize optimizer
        if opt.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                parameters,
                lr=learning_rate,
                eps=eps,
                weight_decay=self.weight_decay
            )
        elif opt.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                parameters,
                lr=learning_rate,
                eps=eps,
                weight_decay=self.weight_decay
            )
        elif opt.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters,
                lr=learning_rate,
                momentum=0.9,  # Common default for SGD
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt}")

        # Initialize GradScaler for mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Initialize scheduler if total_steps is provided
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps) if total_steps else None
        
        if self.debug:
            print(f"[DEBUG Optimizer] Initialized {self.name} with type {opt}, lr={learning_rate}, "
                  f"clip={clip}, use_amp={use_amp}", flush=True)

    def __call__(self, loss: torch.Tensor, parameters_to_clip: Any, retain_graph: bool = False) -> Dict[str, float]:
        """
        Perform an optimization step.
        
        :param loss: Loss tensor to optimize (scalar).
        :param parameters_to_clip: Parameters for gradient clipping.
        :param retain_graph: Whether to retain the computation graph after backward.
        :return: Dictionary of metrics (loss and gradient norm).
        """
        metrics: Dict[str, float] = {f"{self.name}_loss": loss.detach().cpu().item()}

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)  # More memory-efficient
        
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} loss before backward: {metrics[f'{self.name}_loss']:.4f}", flush=True)

        # Backward pass with AMP
        self.scaler.scale(loss).backward(retain_graph=retain_graph)

        # Unscale gradients before clipping
        self.scaler.unscale_(self.optimizer)

        # Clip gradients if specified
        grad_norm = 0.0
        if self.clip is not None:
            grad_norm = torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=self.clip)
            metrics[f"{self.name}_grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if self.debug:
                print(f"[DEBUG Optimizer] {self.name} gradient norm: {metrics[f'{self.name}_grad_norm']:.4f}", flush=True)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
            metrics[f"{self.name}_learning_rate"] = self.optimizer.param_groups[0]["lr"]
            if self.debug:
                print(f"[DEBUG Optimizer] {self.name} updated learning rate: {metrics[f'{self.name}_learning_rate']:.6f}", flush=True)

        return metrics

    def state_dict(self) -> Dict:
        """
        Return the optimizer's state dictionary.
        
        :return: State dictionary including optimizer, scaler, and scheduler states (if applicable).
        """
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict()
        }
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} state_dict keys: {list(state.keys())}", flush=True)
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        """
        Load the optimizer's state from a dictionary.
        
        :param state_dict: Dictionary with optimizer, scaler, and optionally scheduler states.
        """
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} loaded state_dict with keys: {list(state_dict.keys())}", flush=True)

    def get_lr(self) -> float:
        """
        Get the current learning rate.
        
        :return: Current learning rate from the first parameter group.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} current learning rate: {lr:.6f}", flush=True)
        return lr