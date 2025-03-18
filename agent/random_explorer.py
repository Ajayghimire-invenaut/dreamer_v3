import torch
import torch.nn as nn
import torch.distributions as td
from typing import Any
from utils.helper_functions import OneHotDistribution

class RandomExplorer(nn.Module):
    def __init__(self, configuration: Any, action_space: Any) -> None:
        super(RandomExplorer, self).__init__()
        self.configuration = configuration
        self.action_space = action_space

    def actor(self, features: torch.Tensor) -> td.Distribution:
        # This actor ignores the input features and returns a random distribution.
        # For discrete action spaces, it returns either a one-hot distribution or a categorical distribution.
        if getattr(self.configuration, "debug", False):
            print("[DEBUG RandomExplorer] features shape:", features.shape)
        if hasattr(self.action_space, 'n'):
            # For discrete action spaces.
            if self.configuration.actor["distribution_type"] in ["onehot", "onehot_gumble"]:
                # Create zero logits and repeat them for each environment.
                logits = torch.zeros(self.configuration.number_of_possible_actions,
                                     device=self.configuration.computation_device)
                # Create a batch of logits.
                repeated_logits = logits.repeat(self.configuration.number_of_environments, 1)
                if getattr(self.configuration, "debug", False):
                    print("[DEBUG RandomExplorer] Using onehot distribution with logits shape:", repeated_logits.shape)
                return OneHotDistribution(repeated_logits)
            else:
                # For categorical distribution.
                logits = torch.zeros(self.configuration.number_of_possible_actions,
                                     device=self.configuration.computation_device)
                repeated_logits = logits.repeat(self.configuration.number_of_environments, 1)
                if getattr(self.configuration, "debug", False):
                    print("[DEBUG RandomExplorer] Using Categorical distribution with logits shape:", repeated_logits.shape)
                return td.Categorical(logits=repeated_logits)
        else:
            # For continuous action spaces, return a uniform distribution.
            low_tensor = torch.tensor(self.action_space.low, device=self.configuration.computation_device)\
                          .repeat(self.configuration.number_of_environments, 1)
            high_tensor = torch.tensor(self.action_space.high, device=self.configuration.computation_device)\
                           .repeat(self.configuration.number_of_environments, 1)
            if getattr(self.configuration, "debug", False):
                print("[DEBUG RandomExplorer] Using Uniform distribution with low shape:", low_tensor.shape)
            return td.Independent(td.Uniform(low_tensor, high_tensor), 1)

    def forward(self, observation: Any, reset_flags: Any, state: Any = None, training: bool = True) -> tuple:
        # Generate a random action.
        if hasattr(self.action_space, 'n'):
            # For discrete actions, sample a random integer for each environment.
            batch_size = self.configuration.number_of_environments
            # Sample a random integer per environment.
            action_value = torch.randint(low=0, high=self.action_space.n, size=(batch_size,))
            if self.configuration.actor["distribution_type"] in ["onehot", "onehot_gumble"]:
                # Convert to one-hot vectors.
                action = torch.nn.functional.one_hot(action_value, num_classes=self.configuration.number_of_possible_actions).float()
            else:
                action = action_value.float()
        else:
            # For continuous actions, sample uniformly between low and high.
            low = torch.tensor(self.action_space.low, device=self.configuration.computation_device)
            high = torch.tensor(self.action_space.high, device=self.configuration.computation_device)
            action = torch.rand_like(low) * (high - low) + low

        policy_output = {
            "action": action,
            "log_probability": torch.tensor(0.0, device=self.configuration.computation_device)
        }
        new_state = None
        return policy_output, new_state

    def train(self, *args: Any, **kwargs: Any) -> tuple:
        # No training is needed for a random policy.
        return None, {}
