import torch
import torch.nn.utils as torch_utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, Optional
import sys

class Optimizer:
    def __init__(self,
                 name: str,
                 parameters: Any,
                 learning_rate: float,
                 eps: float = 1e-8,
                 clip: Optional[float] = None,
                 weight_decay: Optional[float] = 0.0,
                 opt: str = "adamw",
                 use_amp: bool = False,
                 total_steps: Optional[int] = None,
                 debug: bool = False) -> None:
        self.name = name
        self.parameters = parameters
        self.clip = clip
        self.weight_decay = weight_decay if weight_decay is not None else 0.0
        self.use_amp = use_amp and torch.cuda.is_available()
        self.debug = debug

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
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt}")

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps) if total_steps else None
        
        if self.debug:
            print(f"[DEBUG Optimizer] Initialized {self.name} with type {opt}, lr={learning_rate}, "
                  f"clip={self.clip}, use_amp={self.use_amp}", flush=True, file=sys.stderr)

    def __call__(self, loss: torch.Tensor, parameters_to_clip: Any, retain_graph: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {f"{self.name}_loss": loss.detach().cpu().item()}
        if torch.isinf(loss) or torch.isnan(loss):
            if self.debug:
                print(f"[DEBUG Optimizer] {self.name} loss is inf/NaN: {metrics[f'{self.name}_loss']}, skipping update", 
                      flush=True, file=sys.stderr)
            return metrics
        
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} loss before backward: {metrics[f'{self.name}_loss']:.4f}", 
                  flush=True, file=sys.stderr)

        # Backward pass with AMP
        if self.use_amp and self.scaler is not None:
            with torch.cuda.amp.autocast():
                self.scaler.scale(loss).backward(retain_graph=retain_graph)
            # Clip gradients before unscaling to cap them early and prevent AMP-driven explosion
            if self.clip is not None:
                grad_norm_before = torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=float('inf'))
                if self.debug:
                    print(f"[DEBUG Optimizer] {self.name} raw gradient norm before clipping: {grad_norm_before.item():.4f}", 
                          flush=True, file=sys.stderr)
                torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=self.clip)
                grad_norm = torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=float('inf'))
                metrics[f"{self.name}_grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if self.debug:
                    print(f"[DEBUG Optimizer] {self.name} gradient norm after clipping (max {self.clip}): {metrics[f'{self.name}_grad_norm']:.4f}", 
                          flush=True, file=sys.stderr)
                if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                    print(f"[DEBUG Optimizer] {self.name} clipped gradients are inf/NaN, skipping update", 
                          flush=True, file=sys.stderr)
                    self.optimizer.zero_grad(set_to_none=True)
                    return metrics
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward(retain_graph=retain_graph)
            if self.clip is not None:
                grad_norm_before = torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=float('inf'))
                if self.debug:
                    print(f"[DEBUG Optimizer] {self.name} raw gradient norm before clipping: {grad_norm_before.item():.4f}", 
                          flush=True, file=sys.stderr)
                torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=self.clip)
                grad_norm = torch_utils.clip_grad_norm_(parameters_to_clip, max_norm=float('inf'))
                metrics[f"{self.name}_grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                if self.debug:
                    print(f"[DEBUG Optimizer] {self.name} gradient norm after clipping (max {self.clip}): {metrics[f'{self.name}_grad_norm']:.4f}", 
                          flush=True, file=sys.stderr)
                if torch.isinf(grad_norm) or torch.isnan(grad_norm):
                    print(f"[DEBUG Optimizer] {self.name} clipped gradients are inf/NaN, skipping update", 
                          flush=True, file=sys.stderr)
                    self.optimizer.zero_grad(set_to_none=True)
                    return metrics
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
            metrics[f"{self.name}_learning_rate"] = self.optimizer.param_groups[0]["lr"]
            if self.debug:
                print(f"[DEBUG Optimizer] {self.name} updated learning rate: {metrics[f'{self.name}_learning_rate']:.6f}", 
                      flush=True, file=sys.stderr)

        return metrics

    def state_dict(self) -> Dict:
        state = {"optimizer": self.optimizer.state_dict()}
        if self.use_amp and self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} state_dict keys: {list(state.keys())}", flush=True, file=sys.stderr)
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.use_amp and self.scaler is not None and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} loaded state_dict with keys: {list(state_dict.keys())}", flush=True, file=sys.stderr)

    def get_lr(self) -> float:
        lr = self.optimizer.param_groups[0]["lr"]
        if self.debug:
            print(f"[DEBUG Optimizer] {self.name} current learning rate: {lr:.6f}", flush=True, file=sys.stderr)
        return lr