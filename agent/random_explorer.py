import torch
import torch.nn as nn
import torch.distributions as td
from typing import Any, Tuple, Dict
from utils.helper_functions import OneHotDistribution

class RandomExplorer(nn.Module):
    def __init__(self, configuration: Any, action_space: Any) -> None:
        super(RandomExplorer, self).__init__()
        self.configuration = configuration
        self.action_space = action_space
        self.device = configuration.computation_device if torch.cuda.is_available() else "cpu"
        self.debug = configuration.debug  # Propagate debug flag
        self.to(self.device)

    def actor(self, features: torch.Tensor) -> td.Distribution:
        """
        Returns a random action distribution based on the action space.
        Input: features [B, feature_dim]
        Output: Distribution object
        """
        if self.debug:
            print("[DEBUG RandomExplorer.actor] features shape:", features.shape, flush=True)
        batch_size = features.shape[0]  # Infer batch size from features: [B, feature_dim]
        if hasattr(self.action_space, 'n'):  # Discrete action space
            logits = torch.zeros(batch_size, self.configuration.number_of_possible_actions, device=self.device)  # [B, num_actions]
            if self.configuration.actor["distribution_type"] in ["onehot", "onehot_gumble"]:
                if self.debug:
                    print("[DEBUG RandomExplorer.actor] Using OneHotDistribution with logits shape:", logits.shape, flush=True)
                return OneHotDistribution(logits)
            else:
                if self.debug:
                    print("[DEBUG RandomExplorer.actor] Using Categorical distribution with logits shape:", logits.shape, flush=True)
                return td.Categorical(logits=logits)
        else:  # Continuous action space
            action_dim = self.action_space.shape[0]
            low_tensor = torch.tensor(self.action_space.low, device=self.device).repeat(batch_size, 1)  # [B, action_dim]
            high_tensor = torch.tensor(self.action_space.high, device=self.device).repeat(batch_size, 1)  # [B, action_dim]
            if self.debug:
                print("[DEBUG RandomExplorer.actor] Using Uniform distribution with low shape:", low_tensor.shape, flush=True)
            return td.Independent(td.Uniform(low_tensor, high_tensor), 1)

    def forward(self,
                observation: Any,
                reset_flags: Any,
                state: Any = None,
                training: bool = True) -> Tuple[Dict[str, torch.Tensor], None]:
        """
        Generates random actions for exploration.
        Output: policy_output with [B, ...] tensors, None state
        """
        batch_size = self.configuration.number_of_environments
        dummy_features = torch.zeros(batch_size, 1, device=self.device)  # [B, 1]
        action_dist = self.actor(dummy_features)  # [B, num_actions] or [B, action_dim]
        
        if hasattr(self.action_space, 'n'):  # Discrete action space
            action_indices = torch.randint(low=0, high=self.action_space.n, size=(batch_size,), device=self.device)  # [B]
            if self.configuration.actor["distribution_type"] in ["onehot", "onehot_gumble"]:
                action = torch.nn.functional.one_hot(action_indices, num_classes=self.configuration.number_of_possible_actions).float()  # [B, num_actions]
                log_probability = action_dist.log_prob(action)  # [B]
            else:
                action = action_indices  # [B]
                log_probability = action_dist.log_prob(action_indices)  # [B]
        else:  # Continuous action space
            action = action_dist.sample()  # [B, action_dim]
            log_probability = action_dist.log_prob(action)  # [B]

        policy_output = {
            "action": action,
            "log_probability": log_probability
        }
        new_state = None
        if self.debug:
            print("[DEBUG RandomExplorer.forward] Action shape:", action.shape, flush=True)
            print("[DEBUG RandomExplorer.forward] Log prob shape:", log_probability.shape, flush=True)
        return policy_output, new_state

    def train(self, *args: Any, **kwargs: Any) -> Tuple[None, Dict[str, float]]:
        """
        RandomExplorer does not train; returns empty metrics.
        """
        return None, {}