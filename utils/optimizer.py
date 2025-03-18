import torch
import torch.nn.utils as torch_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, Optional

class Optimizer:
    def __init__(self,
                 name: str,
                 parameters: Any,
                 learning_rate: float,
                 eps: float = 1e-4,
                 clip: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 opt: str = "adamw",
                 use_amp: bool = False,
                 total_steps: Optional[int] = None) -> None:
        self.name = name
        self.parameters = parameters
        self.clip = clip
        self.weight_decay = weight_decay
        if opt.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, eps=eps, weight_decay=weight_decay)
        elif opt.lower() == "adam":
            self.optimizer = torch.optim.Adam(parameters, lr=learning_rate, eps=eps)
        elif opt.lower() == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {opt}")
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps) if total_steps is not None else None

    def __call__(self, loss: torch.Tensor, parameters_to_clip: Any, retain_graph: bool = True) -> Dict[str, float]:
        metrics: Dict[str, float] = {f"{self.name}_loss": loss.detach().cpu().item()}
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward(retain_graph=retain_graph)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch_utils.clip_grad_norm_(parameters_to_clip, self.clip) if self.clip else 0.0
        if self.weight_decay:
            self._apply_weight_decay(parameters_to_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        metrics[f"{self.name}_grad_norm"] = grad_norm
        return metrics

    def _apply_weight_decay(self, parameters: Any) -> None:
        # Ensure not to double-apply weight decay if using AdamW.
        for parameter in parameters:
            parameter.data = (1 - self.weight_decay) * parameter.data

    def state_dict(self) -> Dict:
        return self.optimizer.state_dict()
