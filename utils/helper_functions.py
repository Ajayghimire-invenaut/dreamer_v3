import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any, Dict, List, Tuple, Optional
import math

#########################################
# Reward Objective Module with Debug
#########################################
class RewardObjective(nn.Module):
    def __init__(self, clip_value: float = 5.0, alpha: float = 0.01):
        """
        Reward objective that compares the predicted reward against the ground-truth,
        uses an exponential moving average (EMA) as a running baseline, and clips the error.
        """
        super(RewardObjective, self).__init__()
        self.clip_value = clip_value
        self.alpha = alpha
        self.register_buffer("baseline", torch.tensor(0.0))
    
    def forward(self, imagined_features: torch.Tensor, world_model: Any, ground_truth_reward: torch.Tensor = None) -> torch.Tensor:
        # Predict reward from the imagined features using the reward head.
        predicted_reward = world_model.heads["reward"](imagined_features).mode()
        print(f"[DEBUG RewardObjective] predicted_reward mean: {predicted_reward.mean().item():.4f}, std: {predicted_reward.std().item():.4f}", flush=True)
        
        if ground_truth_reward is not None:
            error = predicted_reward - ground_truth_reward
            print(f"[DEBUG RewardObjective] ground_truth_reward mean: {ground_truth_reward.mean().item():.4f}, std: {ground_truth_reward.std().item():.4f}", flush=True)
            print(f"[DEBUG RewardObjective] error mean: {error.mean().item():.4f}, std: {error.std().item():.4f}", flush=True)
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
    Uses the world model's decoder head to predict the next observation,
    then passes it through the encoder to obtain the next latent embedding.
    """
    # Use the dynamics (RSSM) to get features from the current state.
    features = world_model.dynamics.get_features(state)
    predicted_obs_distribution = world_model.heads["decoder"](features)
    predicted_obs = predicted_obs_distribution["image"].mode()  # or .sample() if desired
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
      Rolls out the dynamics model for a given horizon using the actor to sample actions.
      Instead of reusing current features as the next embedding, this function uses
      predict_next_embedding to generate the next latent embedding.
      The updated version forces the time dimension to be 1 at each step.
      """
      batch_size = start_state["deter"].shape[0]
      feature_list = []
      state_history = {key: [] for key in start_state.keys()}
      actions_list = []
      state = start_state

      for t in range(horizon):
          # Get features from the current state.
          features = dynamics_model.get_features(state)
          print(f"[DEBUG imagine_trajectory] Step {t}: raw features shape: {features.shape}", flush=True)
          # Force features to have a time dimension of 1.
          if features.dim() == 3 and features.size(1) != 1:
              features = features[:, -1:, :]  # take only the last time step
              print(f"[DEBUG imagine_trajectory] Step {t}: reduced features shape: {features.shape}", flush=True)
          feature_list.append(features)

          # Sample action from the actor.
          actor_dist = actor(features)
          action = actor_dist.sample()
          actions_list.append(action)
          print(f"[DEBUG imagine_trajectory] Step {t}: action shape: {action.shape}", flush=True)
          
          # Predict the next latent embedding using the decoder->encoder loop.
          predicted_embedding = predict_next_embedding(state, action, world_model, encoder)
          # If predicted_embedding has a time dimension > 1, reduce it to the last step.
          if predicted_embedding.dim() == 3 and predicted_embedding.size(1) != 1:
              predicted_embedding = predicted_embedding[:, -1, :]
              print(f"[DEBUG imagine_trajectory] Step {t}: reduced predicted_embedding shape: {predicted_embedding.shape}", flush=True)
          
          # Update the state using observe_step. The 'previous_action' here is action.
          state, _ = dynamics_model.observe_step(
              state, action, predicted_embedding, torch.ones(batch_size, device=dynamics_model.device)
          )
          for key, value in state.items():
              state_history[key].append(value)
      
      # Stack along the new time dimension.
      try:
          imagined_features = torch.stack(feature_list, dim=0)  # Expected shape: (horizon, B, 1, feature_dim)
      except Exception as e:
          print(f"[ERROR imagine_trajectory] Error stacking feature_list: {e}", flush=True)
          raise

      imagined_actions = torch.stack(actions_list, dim=0)  # shape: (horizon, B, action_dim)
      imagined_state = {key: torch.stack(vals, dim=0) for key, vals in state_history.items()}

      print(f"[DEBUG imagine_trajectory] Final imagined_features shape: {imagined_features.shape}", flush=True)
      print(f"[DEBUG imagine_trajectory] Final imagined_actions shape: {imagined_actions.shape}", flush=True)
      
      return imagined_features, imagined_state, imagined_actions

#########################################
# Lambda-Return Target and Other Utilities
#########################################
def lambda_return_target(reward: torch.Tensor,
                         value: torch.Tensor,
                         discount: float,
                         lambda_value: float,
                         normalize: bool = True) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    T, B, _ = reward.shape
    returns: List[torch.Tensor] = [None] * T
    next_return = value[-1]
    for t in reversed(range(T)):
        next_return = reward[t] + discount * ((1 - lambda_value) * value[t] + lambda_value * next_return)
        returns[t] = next_return
    returns_tensor = torch.stack(returns, dim=0)
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
    target_stack = torch.stack(target, dim=0)
    advantage = target_stack - baseline
    dist = actor(features)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy().mean()
    actor_loss = -(log_prob * advantage.detach() * weights.unsqueeze(-1)).mean()
    actor_loss -= configuration.actor["entropy"] * entropy
    metrics = {"actor_loss": actor_loss.item(), "actor_entropy": entropy.item()}
    return actor_loss, metrics

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
        return torch.nn.functional.one_hot(torch.argmax(self.logits, dim=-1),
                                             num_classes=self.logits.shape[-1]).float()
    def sample(self) -> torch.Tensor:
        probabilities = torch.softmax(self.logits, dim=-1)
        sample_indices = torch.multinomial(probabilities, num_samples=1)
        return torch.nn.functional.one_hot(sample_indices.squeeze(-1),
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
            logits = self.logits.reshape(-1, self.logits.shape[-1])
            return -nn.functional.cross_entropy(logits, target.reshape(-1), reduction='mean')
        elif self.dist_type == "binary":
            # For binary distributions, assume target values are in [0,1]
            return -nn.functional.binary_cross_entropy_with_logits(self.logits, target, reduction='mean')
        else:
            raise ValueError("Unsupported distribution type")
    def mode(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            return self.logits
        elif self.dist_type == "symlog_disc":
            mode = torch.argmax(self.logits, dim=-1)
            return mode.float()
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
            return distribution.sample().float()
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
