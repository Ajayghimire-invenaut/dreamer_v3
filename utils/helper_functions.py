"""
Helper functions and scheduling utilities for Dreamer-V3.

This module includes:
  - Scheduling classes: Every, Until, Once.
  - Deterministic utilities: set_random_seed, enable_deterministic_mode.
  - Data metrics: count_episode_steps, tensor_stats.
  - Loss and target computations: static_scan_imagine, lambda_return_target, compute_actor_loss.
  - Advanced data augmentation: augment_image.
  - OneHotDistribution helper.
  - RewardEMA for reward normalization.
  - tensor_to_numpy utility.
"""

import random
import time
import torch
import numpy as np
from typing import Any, Dict, List, Tuple

# Scheduling Utilities
class Every:
    def __init__(self, interval: int) -> None:
        self.interval = interval
    def __call__(self, current_step: int) -> bool:
        return current_step % self.interval == 0

class Until:
    def __init__(self, end_step: int) -> None:
        self.end_step = end_step
    def __call__(self, current_step: int) -> bool:
        return current_step < self.end_step

class Once:
    def __init__(self) -> None:
        self.done = False
    def __call__(self) -> bool:
        if not self.done:
            self.done = True
            return True
        return False
    def reset_flag(self) -> None:
        self.done = False

# Deterministic Setup
def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enable_deterministic_mode() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data Metrics
def count_episode_steps(directory: Any) -> int:
    from pathlib import Path
    directory = Path(directory)
    total = 0
    for file in directory.glob("*.npz"):
        try:
            data = np.load(file)
            total += len(data["reward"]) - 1
        except Exception:
            continue
    return total

def tensor_stats(tensor: torch.Tensor, name: str) -> Dict[str, float]:
    return {f"{name}_mean": tensor.mean().item(), f"{name}_std": tensor.std().item()}

# Imagined Trajectory Rollout
def static_scan_imagine(start_state: Any, horizon: int, dynamics_model: Any, action_dim: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Roll out the dynamics model for imagined trajectories.
    
    (In the official implementation, the model is rolled out by iteratively sampling actions.)
    Here we simulate this by returning zeroed tensors with the correct shapes.
    
    Args:
      start_state: The starting state (unused in this simplified version).
      horizon: Number of time steps to simulate.
      dynamics_model: An instance with attributes stoch_dimension, deter_dimension, and device.
      action_dim: The dimension of the actions (number of possible actions).
    
    Returns:
      imagined_features: Tensor [horizon, batch, (stoch_dimension + deter_dimension)].
      imagined_state: Dict with "mean", "std", and "deter" each of shape [horizon, batch, dimension].
      imagined_actions: Dummy tensor of shape [horizon, batch, action_dim].
    """
    batch_size = 32  # Make sure this matches your training batch size.
    total_dimension = dynamics_model.stoch_dimension + dynamics_model.deter_dimension
    imagined_features = torch.zeros(horizon, batch_size, total_dimension, device=dynamics_model.device)
    imagined_state = {
        "mean": torch.zeros(horizon, batch_size, dynamics_model.stoch_dimension, device=dynamics_model.device),
        "std": torch.ones(horizon, batch_size, dynamics_model.stoch_dimension, device=dynamics_model.device),
        "deter": torch.zeros(horizon, batch_size, dynamics_model.deter_dimension, device=dynamics_model.device)
    }
    imagined_actions = torch.zeros(horizon, batch_size, action_dim, device=dynamics_model.device)
    return imagined_features, imagined_state, imagined_actions

# Loss and Target Computations
def lambda_return_target(reward: torch.Tensor,
                         value: torch.Tensor,
                         discount: float,
                         lambda_value: float,
                         normalize: bool = True) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Compute lambda-return targets and optionally normalize them.
    
    Instead of clamping the return using free nats, this implementation computes the recursive return and then
    normalizes it by subtracting the mean and dividing by max(std, 1.0).
    
    Args:
      reward: Tensor of shape [T, B, 1].
      value: Tensor of shape [T, B, 1].
      discount: Discount factor (e.g. 0.997).
      lambda_value: Lambda parameter (e.g. 0.95).
      normalize: Whether to normalize the returns.
      
    Returns:
      target: List of T tensors.
      weights: Tensor of shape [T, B] (all ones).
      baseline: The original value tensor.
    """
    T, B, _ = reward.shape
    returns: List[torch.Tensor] = [None] * T
    next_return = value[-1]
    for t in reversed(range(T)):
        next_return = reward[t] + discount * ((1 - lambda_value) * value[t] + lambda_value * next_return)
        returns[t] = next_return
    returns_tensor = torch.stack(returns, dim=0)  # Shape [T, B, 1]
    if normalize:
        mean = returns_tensor.mean()
        std = returns_tensor.std()
        divisor = std if std > 1.0 else 1.0
        returns_tensor = (returns_tensor - mean) / divisor
        returns = [returns_tensor[t] for t in range(T)]
    weights = torch.ones(T, B, device=value.device)
    return returns, weights, value

def compute_actor_loss(actor: Any,
                       features: torch.Tensor,
                       actions: torch.Tensor,
                       target: List[torch.Tensor],
                       weights: torch.Tensor,
                       baseline: torch.Tensor,
                       value_network: Any,
                       configuration: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute actor loss with advantage estimation and entropy bonus.
    
    Args:
      actor: Actor network.
      features: Features from world model.
      actions: Imagined actions.
      target: List of lambda-return targets.
      weights: Weight tensor.
      baseline: Baseline value estimates.
      value_network: Critic network.
      configuration: Configuration with hyperparameters.
    
    Returns:
      actor_loss: Scalar tensor.
      metrics: Dict with metrics.
    """
    target_stack = torch.stack(target, dim=0)  # Shape [T, B, 1]
    advantage = target_stack - baseline
    dist = actor(features)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    actor_loss = -(log_prob * advantage.detach() * weights.unsqueeze(-1)).mean()
    actor_loss -= configuration.actor["entropy"] * entropy
    metrics = {"actor_loss": actor_loss.item(), "actor_entropy": entropy.item()}
    return actor_loss, metrics

# Advanced Data Augmentation
def augment_image(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    """
    Apply advanced data augmentation: random crop, horizontal flip, brightness/contrast.
    
    Args:
      image: Tensor [B, H, W, C].
      crop_size: Crop size.
    
    Returns:
      Augmented image tensor.
    """
    if image.dim() != 4:
        return image
    B, H, W, C = image.shape
    if H <= crop_size or W <= crop_size:
        return image
    top = torch.randint(0, H - crop_size + 1, (1,)).item()
    left = torch.randint(0, W - crop_size + 1, (1,)).item()
    cropped = image[:, top:top+crop_size, left:left+crop_size, :]
    if torch.rand(1).item() < 0.5:
        cropped = torch.flip(cropped, dims=[2])
    brightness = 0.8 + 0.4 * torch.rand(1).item()
    contrast = 0.8 + 0.4 * torch.rand(1).item()
    mean = cropped.mean(dim=(1,2), keepdim=True)
    cropped = (cropped - mean) * contrast + mean * brightness
    return torch.clamp(cropped, 0.0, 1.0)

# OneHotDistribution Helper
class OneHotDistribution:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -((self.logits - value) ** 2).mean()
    def mode(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(torch.argmax(self.logits, dim=-1),
                                             num_classes=self.logits.shape[-1]).float()
    def sample(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        sample_indices = torch.multinomial(probabilities, num_samples=1)
        return torch.nn.functional.one_hot(sample_indices.squeeze(-1),
                                             num_classes=self.logits.shape[-1]).float()

# Updated DistributionWrapper with an entropy method
class DistributionWrapper:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == self.logits.dim() - 1:
            target = target.unsqueeze(-1)
        return -((self.logits - target) ** 2).mean()
    def mode(self) -> torch.Tensor:
        return self.logits
    def sample(self) -> torch.Tensor:
        noise = torch.randn_like(self.logits) * 0.1
        return self.logits + noise
    def entropy(self) -> torch.Tensor:
        sigma = 0.1
        entropy_value = 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e * (sigma ** 2)))
        return torch.full_like(self.logits, entropy_value)

# RewardEMA for reward normalization
class RewardEMA:
    def __init__(self, device: Any, alpha: float = 1e-2) -> None:
        self.device = device
        self.alpha = alpha
        self.quantile_range = torch.tensor([0.05, 0.95], device=device)
    def __call__(self, input_tensor: torch.Tensor, ema_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flattened = input_tensor.detach().flatten()
        quantiles = torch.quantile(flattened, self.quantile_range)
        ema_values[:] = self.alpha * quantiles + (1 - self.alpha) * ema_values
        scale = torch.clamp(ema_values[1] - ema_values[0], min=1.0)
        offset = ema_values[0]
        return offset.detach(), scale.detach()

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
