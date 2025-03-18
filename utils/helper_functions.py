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
    # Map clamped values to bins in 0..(num_bins-1)
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
        Compares the predicted reward (obtained via the reward head’s mode)
        against the ground-truth reward (if provided) using an EMA baseline.
        If ground_truth_reward is not provided, the intrinsic reward is simply
        the clamped predicted reward.

        NOTE: If using symlog discretization, this module applies symlog to the
        ground-truth reward so that it is in the same space as the predictions.
        """
        super(RewardObjective, self).__init__()
        self.clip_value = clip_value
        self.alpha = alpha
        self.register_buffer("baseline", torch.tensor(0.0))
    
    def forward(self, imagined_features: torch.Tensor, world_model: Any, ground_truth_reward: Optional[torch.Tensor] = None) -> torch.Tensor:
        predicted_reward = world_model.heads["reward"](imagined_features).mode()
        print(f"[DEBUG RewardObjective] predicted_reward mean: {predicted_reward.mean().item():.4f}, std: {predicted_reward.std().item():.4f}", flush=True)
        
        # If using symlog discretization, transform the ground-truth reward into symlog space.
        if ground_truth_reward is not None and world_model.configuration.reward_head["distribution_type"] == "symlog_disc":
            ground_truth_reward = symlog(ground_truth_reward)
            print(f"[DEBUG RewardObjective] ground_truth_reward after symlog transform mean: {ground_truth_reward.mean().item():.4f}, std: {ground_truth_reward.std().item():.4f}", flush=True)
        
        if ground_truth_reward is not None:
            error = predicted_reward - ground_truth_reward
            print(f"[DEBUG RewardObjective] error mean: {error.mean().item():.4f}, std: {error.std().item():.4f}", flush=True)
            # Update baseline with EMA.
            self.baseline = self.alpha * error.mean() + (1 - self.alpha) * self.baseline
            print(f"[DEBUG RewardObjective] updated baseline: {self.baseline.item():.4f}", flush=True)
            error = error - self.baseline
            error = torch.clamp(error, -self.clip_value, self.clip_value)
            intrinsic_reward = torch.abs(error)
        else:
            intrinsic_reward = torch.clamp(predicted_reward, -self.clip_value, self.clip_value)
        
        print(f"[DEBUG RewardObjective] intrinsic_reward mean: {intrinsic_reward.mean().item():.4f}, std: {intrinsic_reward.std().item():.4f}", flush=True)
        return intrinsic_reward

#########################################
# Prediction Step: Decoder + Encoder
#########################################
def predict_next_embedding(state: Dict[str, torch.Tensor],
                           action: torch.Tensor,
                           world_model: Any,
                           encoder: Any) -> torch.Tensor:
    """
    Uses the world model’s decoder to predict the next observation distribution,
    then takes its mode and feeds the resulting image through the encoder to obtain
    the next latent embedding.
    """
    features = world_model.dynamics.get_features(state)
    predicted_obs_distribution = world_model.heads["decoder"](features)
    predicted_obs = predicted_obs_distribution["image"].mode()  # alternatively, use .sample()
    print(f"[DEBUG predict_next_embedding] predicted_obs shape: {predicted_obs.shape}", flush=True)
    predicted_embedding = encoder({"image": predicted_obs})
    print(f"[DEBUG predict_next_embedding] predicted_embedding mean: {predicted_embedding.mean().item():.4f}, std: {predicted_embedding.std().item():.4f}", flush=True)
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
                       configuration: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    """
    Rolls out the dynamics model for a given horizon.
    For each time step:
      1. Extracts features from the current state.
      2. Samples an action using the actor.
      3. Predicts the next latent embedding via the decoder-encoder loop.
      4. Updates the state with a one-step transition.
    
    The outputs are stacked in time-first order and then transposed to batch-first.
    """
    batch_size = start_state["deter"].shape[0]
    feature_list = []
    state_history = {key: [] for key in start_state.keys()}
    actions_list = []
    state = start_state

    for t in range(horizon):
        features = dynamics_model.get_features(state)
        print(f"[DEBUG imagine_trajectory] Step {t}: raw features shape: {features.shape}", flush=True)
        # Ensure features have an explicit time dimension.
        if features.dim() == 2:
            features = features.unsqueeze(0)  # [1, B, f]
        elif features.dim() == 3 and features.size(0) != 1:
            # If more than one time step exists, use only the last step.
            features = features[-1:].clone()
            print(f"[DEBUG imagine_trajectory] Step {t}: reduced features shape: {features.shape}", flush=True)
        feature_list.append(features)
        actor_dist = actor(features.detach())  # Detach features to avoid leaking gradients.
        action = actor_dist.sample()
        actions_list.append(action)
        print(f"[DEBUG imagine_trajectory] Step {t}: action shape: {action.shape}", flush=True)
        predicted_embedding = predict_next_embedding(state, action, world_model, encoder)
        if predicted_embedding.dim() == 2:
            predicted_embedding = predicted_embedding.unsqueeze(0)
            print(f"[DEBUG imagine_trajectory] Step {t}: expanded predicted_embedding shape: {predicted_embedding.shape}", flush=True)
        # Update the state using a one-step transition. Squeeze the predicted embedding’s time dimension.
        state, _ = dynamics_model.observe_step(
            state, action, predicted_embedding.squeeze(0), torch.ones(batch_size, device=dynamics_model.device)
        )
        for key, value in state.items():
            state_history[key].append(value)
    
    # Stack results along the time dimension (results are [T, B, ...]).
    imagined_features = torch.stack(feature_list, dim=0)
    imagined_actions = torch.stack(actions_list, dim=0)
    imagined_state = {key: torch.stack(vals, dim=0) for key, vals in state_history.items()}
    
    # Transpose from time-first [T, B, ...] to batch-first [B, T, ...].
    imagined_features = imagined_features.transpose(0, 1)
    imagined_actions = imagined_actions.transpose(0, 1)
    imagined_state = {key: value.transpose(0, 1) for key, value in imagined_state.items()}
    
    print(f"[DEBUG imagine_trajectory] Final imagined_features shape: {imagined_features.shape}", flush=True)
    print(f"[DEBUG imagine_trajectory] Final imagined_actions shape: {imagined_actions.shape}", flush=True)
    
    return imagined_features, imagined_state, imagined_actions

#########################################
# Lambda-Return Target and Other Utilities
#########################################
def lambda_return_target(reward, value, discount, lambda_):
    # Ensure reward and value have shape [T, B, 1]
    if reward.dim() == 2:
        reward = reward.unsqueeze(-1)
    if value.dim() == 2:
        value = value.unsqueeze(-1)
    T = reward.shape[0]
    if value.shape[0] == T:
        value = torch.cat([value, value[-1:].clone()], dim=0)
    assert value.shape[0] == T + 1, "Value tensor must have T+1 timesteps"
    target_list = [value[-1]]  # bootstrap
    weight_list = [torch.ones_like(reward[0])]
    for t in reversed(range(T)):
        current_target = reward[t] + discount * ((1 - lambda_) * value[t+1] + lambda_ * target_list[0])
        target_list.insert(0, current_target)
        current_weight = discount * lambda_ * weight_list[0]
        weight_list.insert(0, current_weight)
    baseline = value[:-1].transpose(0, 1)  # [B, T, 1]
    final_targets = target_list[:-1]  # list of T tensors, each [B, 1]
    final_weights = weight_list[:-1]
    return final_targets, final_weights, baseline

def compute_actor_loss(actor, features, actions, target, weights, baseline, value_network, configuration):
    # Detach features so that actor loss does not backprop into the world model.
    features_detached = features.detach()
    dist = actor(features_detached)
    log_prob = dist.log_prob(actions)
    target_stack = torch.stack(target, dim=1)  # shape [B, T, 1]
    advantage = (target_stack - baseline).detach()
    entropy = dist.entropy()
    weights_stack = torch.stack(weights, dim=1)  # shape [B, T, 1]
    actor_loss = - (log_prob * advantage * weights_stack).mean() - configuration.actor.get("entropy", 0.0) * entropy.mean()
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

def tensor_to_numpy(tensor):
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
        return F.one_hot(torch.argmax(self.logits, dim=-1),
                         num_classes=self.logits.shape[-1]).float()
    def sample(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        sample_indices = torch.multinomial(probabilities, num_samples=1)
        return F.one_hot(sample_indices.squeeze(-1),
                         num_classes=self.logits.shape[-1]).float()

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
            # Use reshape instead of view.
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
