import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any, Dict, List, Tuple, Optional
import math

#########################################
# Symlog Transformation Utilities
#########################################
def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inv_symlog(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def discretize_symlog(x: torch.Tensor, num_bins: int = 255, low: float = -10.0, high: float = 10.0) -> torch.Tensor:
    """
    Applies symlog to x, then discretizes to integer bins.
    Assumes x is in a scale where most values lie between low and high.
    """
    y = symlog(x)
    y_clamped = torch.clamp(y, low, high)
    bins = ((y_clamped - low) / (high - low) * (num_bins - 1)).long()
    return bins

def undisc_symlog(bins: torch.Tensor, num_bins: int = 255, low: float = -10.0, high: float = 10.0) -> torch.Tensor:
    """
    Converts discretized symlog bin indices back to a real value.
    """
    y = bins.float() / (num_bins - 1) * (high - low) + low
    return inv_symlog(y)

#########################################
# Orthogonal Initialization
#########################################
def orthogonal_initialize(module: nn.Module, gain: Optional[float] = None) -> None:
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

#########################################
# Reward Objective Module with Debug
#########################################
class RewardObjective(nn.Module):
    def __init__(self, clip_value: float = 5.0, alpha: float = 0.01):
        """
        Compares the predicted reward against ground-truth reward (if provided) using an EMA baseline.
        If no ground-truth is provided, returns clamped predicted reward as intrinsic reward.
        Assumes batch-first input: [B, H, feature_dim].
        """
        super(RewardObjective, self).__init__()
        self.clip_value = clip_value
        self.alpha = alpha
        self.register_buffer("baseline", torch.tensor(0.0))
    
    def forward(self, imagined_features: torch.Tensor, world_model: Any, ground_truth_reward: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input: [B, H, feature_dim]
        reward_dist = world_model.heads["reward"](imagined_features)  # [B, H, ...]
        predicted_reward = reward_dist.mode()  # [B, H, 1]
        if getattr(world_model.configuration, "debug", False):
            print(f"[DEBUG RewardObjective] predicted_reward shape: {predicted_reward.shape}, "
                  f"mean: {predicted_reward.mean().item():.4f}, std: {predicted_reward.std().item():.4f}", flush=True)
        
        if ground_truth_reward is not None:
            if world_model.configuration.reward_head["distribution_type"] == "symlog_disc":
                # Keep predicted reward in symlog space for comparison
                predicted_symlog = reward_dist.logits.max(dim=-1)[1]  # [B, H]
                predicted_symlog_value = undisc_symlog(predicted_symlog, num_bins=255)  # [B, H]
                ground_truth_symlog = symlog(ground_truth_reward)  # [B, H]
                if getattr(world_model.configuration, "debug", False):
                    print(f"[DEBUG RewardObjective] ground_truth_symlog shape: {ground_truth_symlog.shape}, "
                          f"mean: {ground_truth_symlog.mean().item():.4f}, std: {ground_truth_symlog.std().item():.4f}", flush=True)
                error = predicted_symlog_value - ground_truth_symlog
            else:
                error = predicted_reward - ground_truth_reward  # [B, H, 1]
            
            if getattr(world_model.configuration, "debug", False):
                print(f"[DEBUG RewardObjective] error shape: {error.shape}, "
                      f"mean: {error.mean().item():.4f}, std: {error.std().item():.4f}", flush=True)
            self.baseline = self.alpha * error.mean() + (1 - self.alpha) * self.baseline
            if getattr(world_model.configuration, "debug", False):
                print(f"[DEBUG RewardObjective] updated baseline: {self.baseline.item():.4f}", flush=True)
            error = error - self.baseline
            error = torch.clamp(error, -self.clip_value, self.clip_value)
            intrinsic_reward = torch.abs(error)  # [B, H, 1] or [B, H]
        else:
            intrinsic_reward = torch.clamp(predicted_reward, -self.clip_value, self.clip_value)  # [B, H, 1]
        
        if intrinsic_reward.dim() == 2:
            intrinsic_reward = intrinsic_reward.unsqueeze(-1)  # Ensure [B, H, 1]
        
        if getattr(world_model.configuration, "debug", False):
            print(f"[DEBUG RewardObjective] intrinsic_reward shape: {intrinsic_reward.shape}, "
                  f"mean: {intrinsic_reward.mean().item():.4f}, std: {intrinsic_reward.std().item():.4f}", flush=True)
        return intrinsic_reward

#########################################
# Prediction Step: Decoder + Encoder
#########################################
def predict_next_embedding(state: Dict[str, torch.Tensor],
                          action: torch.Tensor,
                          world_model: Any,
                          encoder: Any) -> torch.Tensor:
    """
    Predicts next observation via decoder and encodes it.
    Input state: Dict[B, ...], action: [B, action_dim]
    Output: [B, embed_dim]
    """
    features = world_model.dynamics.get_features(state)  # [B, feature_dim] or [B, T, feature_dim]
    if features.dim() == 3:
        features = features[:, -1, :]  # Take last timestep if multi-step: [B, feature_dim]
    predicted_obs_dist = world_model.heads["decoder"](features)  # [B, C, H, W]
    predicted_obs = predicted_obs_dist["image"].mode()  # [B, C, H, W]
    if getattr(world_model.configuration, "debug", False):
        print(f"[DEBUG predict_next_embedding] predicted_obs shape: {predicted_obs.shape}", flush=True)
    predicted_embedding = encoder({"image": predicted_obs})  # [B, embed_dim]
    if getattr(world_model.configuration, "debug", False):
        print(f"[DEBUG predict_next_embedding] predicted_embedding shape: {predicted_embedding.shape}, "
              f"mean: {predicted_embedding.mean().item():.4f}, std: {predicted_embedding.std().item():.4f}", flush=True)
    return predicted_embedding

#########################################
# Rollout (Imagination) Step with Debug
#########################################
def imagine_trajectory(start_state: Dict[str, torch.Tensor],
                      horizon: int,
                      dynamics_model: Any,
                      actor: Any,
                      world_model: Any,
                      encoder: Any,
                      configuration: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Rolls out the dynamics model for a given horizon.
    Outputs are batch-first: [B, H, ...].
    Returns features, state, actions, and action indices.
    """
    batch_size = start_state["deter"].shape[0]
    feature_list = []
    state_history = {key: [] for key in start_state.keys()}
    actions_list = []
    action_indices_list = []
    state = {k: v.clone() for k, v in start_state.items()}

    for t in range(horizon):
        features = dynamics_model.get_features(state)  # [B, feature_dim] or [B, T, feature_dim]
        if getattr(configuration, "debug", False):
            print(f"[DEBUG imagine_trajectory] Step {t}: raw features shape: {features.shape}", flush=True)
        if features.dim() == 3:
            features = features[:, -1, :]  # [B, feature_dim]
            if getattr(configuration, "debug", False):
                print(f"[DEBUG imagine_trajectory] Step {t}: reduced features shape: {features.shape}", flush=True)
        feature_list.append(features)
        
        actor_dist = actor(features.detach())
        action_indices = actor_dist.sample()  # [B] or [B, action_dim]
        if action_indices.dim() == 1:  # Discrete actions
            action = F.one_hot(action_indices.long(), num_classes=configuration.number_of_possible_actions).float()  # [B, num_actions]
            action_indices_list.append(action_indices)
        else:  # Continuous actions
            action = action_indices
            action_indices_list.append(action)
        actions_list.append(action)
        if getattr(configuration, "debug", False):
            print(f"[DEBUG imagine_trajectory] Step {t}: action shape: {action.shape}, "
                  f"action_indices shape: {action_indices.shape}", flush=True)
        
        predicted_embedding = predict_next_embedding(state, action, world_model, encoder)
        if predicted_embedding.dim() == 3:
            predicted_embedding = predicted_embedding.squeeze(0)
            if getattr(configuration, "debug", False):
                print(f"[DEBUG imagine_trajectory] Step {t}: squeezed predicted_embedding shape: {predicted_embedding.shape}", flush=True)
        
        state, _ = dynamics_model.observe_step(
            state, action, predicted_embedding, torch.ones(batch_size, device=dynamics_model.device)
        )
        for key, value in state.items():
            state_history[key].append(value)

    imagined_features = torch.stack(feature_list, dim=1)  # [B, H, feature_dim]
    imagined_state = {key: torch.stack(vals, dim=1) for key, vals in state_history.items()}  # Dict[B, H, ...]
    imagined_actions = torch.stack(actions_list, dim=1)  # [B, H, action_dim]
    imagined_action_indices = torch.stack(action_indices_list, dim=1)  # [B, H] or [B, H, action_dim]
    
    if getattr(configuration, "debug", False):
        print(f"[DEBUG imagine_trajectory] Final imagined_features shape: {imagined_features.shape}", flush=True)
        print(f"[DEBUG imagine_trajectory] Final imagined_actions shape: {imagined_actions.shape}", flush=True)
        print(f"[DEBUG imagine_trajectory] Final imagined_action_indices shape: {imagined_action_indices.shape}", flush=True)
    
    return imagined_features, imagined_state, imagined_actions, imagined_action_indices

#########################################
# Lambda-Return Target and Other Utilities
#########################################
def lambda_return_target(reward: torch.Tensor, value: torch.Tensor, discount: float, lambda_: float) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    # Input: [H, B, 1] for reward and value
    if reward.dim() == 2:
        reward = reward.unsqueeze(-1)
    if value.dim() == 2:
        value = value.unsqueeze(-1)
    T = reward.shape[0]
    if value.shape[0] == T:
        value = torch.cat([value, value[-1:].clone()], dim=0)
    assert value.shape[0] == T + 1, "Value tensor must have T+1 timesteps"
    target_list = [value[-1]]  # [B, 1]
    weight_list = [torch.ones_like(reward[0])]  # [B, 1]
    for t in reversed(range(T)):
        current_target = reward[t] + discount * ((1 - lambda_) * value[t+1] + lambda_ * target_list[0])
        target_list.insert(0, current_target)
        current_weight = discount * lambda_ * weight_list[0]
        weight_list.insert(0, current_weight)
    baseline = value[:-1].transpose(0, 1)  # [B, T, 1]
    final_targets = target_list[:-1]  # List[T] of [B, 1]
    final_weights = weight_list[:-1]  # List[T] of [B, 1]
    return final_targets, final_weights, baseline

def compute_actor_loss(actor: Any, features: torch.Tensor, actions: torch.Tensor, 
                      action_indices: torch.Tensor, target: List[torch.Tensor], 
                      weights: List[torch.Tensor], baseline: torch.Tensor, 
                      value_network: Any, configuration: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    features_detached = features.detach()  # [B, H, feature_dim]
    dist = actor(features_detached)  # [B, H, action_dim]
    if action_indices.dim() == 2:  # Discrete: [B, H]
        log_prob = dist.log_prob(action_indices)  # [B, H]
    else:  # Continuous: [B, H, action_dim]
        log_prob = dist.log_prob(actions)  # [B, H]
    
    target_stack = torch.stack(target, dim=1)  # [B, T, 1]
    advantage = (target_stack - baseline).detach()  # [B, T, 1]
    entropy = dist.entropy()  # [B, H]
    weights_stack = torch.stack(weights, dim=1)  # [B, T, 1]
    actor_loss = - (log_prob * advantage.squeeze(-1) * weights_stack.squeeze(-1)).mean() - configuration.actor.get("entropy", 0.0) * entropy.mean()
    loss_metrics = {"actor_loss": actor_loss.item(), "policy_entropy": entropy.mean().item()}
    return actor_loss, loss_metrics

def tensor_stats(tensor: torch.Tensor, name: str) -> Dict[str, float]:
    return {f"{name}_mean": tensor.mean().item(), f"{name}_std": tensor.std().item()}

def augment_image(image: torch.Tensor, crop_size: int) -> torch.Tensor:
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

def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enable_deterministic_mode() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def tensor_to_numpy(tensor: Any) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        return np.array(tensor)

class OneHotDistribution:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return -((self.logits - value) ** 2).mean()

    def mode(self) -> torch.Tensor:
        return F.one_hot(torch.argmax(self.logits, dim=-1), num_classes=self.logits.shape[-1]).float()

    def sample(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        sample_indices = torch.multinomial(probabilities, num_samples=1)
        return F.one_hot(sample_indices.squeeze(-1), num_classes=self.logits.shape[-1]).float()

    def entropy(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        return -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=-1)  # [B] or [B, H]

class DistributionWrapper:
    def __init__(self, logits: torch.Tensor, dist_type: str = "gaussian") -> None:
        self.logits = logits
        self.dist_type = dist_type
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if self.dist_type == "gaussian":
            if target.dim() == self.logits.dim() - 1:
                target = target.unsqueeze(-1)
            if self.logits.shape[0] != target.shape[0]:
                target = target.transpose(0, 1)
            return -((self.logits - target) ** 2).mean()
        elif self.dist_type == "symlog_disc":
            target = target.long().squeeze(-1)
            logits_flat = self.logits.reshape(-1, self.logits.shape[-1])
            return -nn.functional.cross_entropy(logits_flat, target.reshape(-1), reduction='mean')
        elif self.dist_type == "binary":
            return -nn.functional.binary_cross_entropy_with_logits(self.logits, target, reduction='mean')
        else:
            raise ValueError("Unsupported distribution type")
    def mode(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            return self.logits
        elif self.dist_type == "symlog_disc":
            mode = torch.argmax(self.logits, dim=-1)
            return undisc_symlog(mode, num_bins=self.logits.shape[-1])
        elif self.dist_type == "binary":
            return (self.logits >= 0).float()
        else:
            raise ValueError("Unsupported distribution type")
    def sample(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            noise = torch.randn_like(self.logits) * 0.1
            return self.logits + noise
        elif self.dist_type == "symlog_disc":
            probs = torch.softmax(self.logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            sample_idx = distribution.sample()
            return undisc_symlog(sample_idx, num_bins=self.logits.shape[-1])
        elif self.dist_type == "binary":
            probs = torch.sigmoid(self.logits)
            return torch.bernoulli(probs)
        else:
            raise ValueError("Unsupported distribution type")
    def entropy(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            sigma = 0.1
            entropy_value = 0.5 * math.log(2 * math.pi * math.e * (sigma ** 2))
            return torch.full_like(self.logits, entropy_value)
        elif self.dist_type == "symlog_disc":
            probs = torch.softmax(self.logits, dim=-1)
            return torch.distributions.Categorical(probs=probs).entropy()
        elif self.dist_type == "binary":
            p = torch.sigmoid(self.logits)
            entropy = - p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
            return entropy
        else:
            raise ValueError("Unsupported distribution type")

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