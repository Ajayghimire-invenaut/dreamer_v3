import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from typing import Any, Tuple, Dict, Optional
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)

from agent.networks import OneHotDistribution, DistributionWrapper

class RandomExplorer(nn.Module):
    """
    Random action policy for exploration in DreamerV3.
    Generates random actions based on the action space, supporting both discrete and continuous spaces.
    Used during initial environment interactions to collect diverse experience.
    """
    def __init__(self, configuration: Any, action_space: Any) -> None:
        super(RandomExplorer, self).__init__()
        self.configuration = configuration
        self.action_space = action_space
        self.device = configuration.computation_device if torch.cuda.is_available() else "cpu"
        self.enable_debugging = getattr(configuration, "debug", False)
        
        # Determine action space type
        self.has_discrete_actions = hasattr(self.action_space, 'n')
        
        if self.has_discrete_actions:
            self.action_dimension = self.action_space.n  # e.g., 2 for CartPole-v1
            self.distribution_type = getattr(configuration, "actor_distribution_type", "onehot")
            self.logits_shape = (self.action_dimension,)
        else:
            self.action_dimension = np.prod(self.action_space.shape)
            self.distribution_type = "normal"
            self.action_lower_bound = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self.action_upper_bound = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)
        
        # Distribution parameters
        self.temperature = getattr(configuration, "actor_temperature", 1.0)
        self.uniform_mix_ratio = getattr(configuration, "actor_unimix_ratio", 0.01)
        
        if self.enable_debugging:
            logger.debug(f"[DEBUG RandomExplorer] Initialized with action_dimension={self.action_dimension}, "
                         f"has_discrete_actions={self.has_discrete_actions}, distribution_type={self.distribution_type}")
        
        self.to(self.device)

    def create_actor_distribution(self, features: torch.Tensor) -> Any:
        """Create a random action distribution based on the action space type."""
        batch_size = features.shape[0] if features is not None and features.nelement() > 0 else 1
        
        if self.has_discrete_actions:
            # Uniform distribution over discrete actions
            logits = torch.zeros(batch_size, self.action_dimension, device=self.device)
            dist = OneHotDistribution(logits, temperature=self.temperature, uniform_mix_ratio=self.uniform_mix_ratio)
            if self.enable_debugging:
                logger.debug(f"Created OneHotDistribution with logits shape: {logits.shape}")
            return dist
        else:
            # Uniform-like normal distribution for continuous actions
            logits = torch.zeros(batch_size, self.action_dimension * 2, device=self.device)
            logits[:, :self.action_dimension] = 0.0  # Mean centered at 0
            action_range = (self.action_upper_bound - self.action_lower_bound) / 2.0
            logits[:, self.action_dimension:] = torch.log(action_range + 1e-6)  # Log std to cover action range
            dist = DistributionWrapper(
                logits=logits,
                distribution_type=self.distribution_type,
                temperature=self.temperature,
                uniform_mix_ratio=0.0,  # No mixing for continuous actions
                number_of_bins=255,
                lower_bound=-20.0,
                upper_bound=20.0
            )
            if self.enable_debugging:
                logger.debug(f"Created DistributionWrapper with logits shape: {logits.shape}")
            return dist

    def forward(
        self,
        observation: Any,
        state: Optional[Dict[str, torch.Tensor]] = None,
        training: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        """Generate a single random action based on the action space, ignoring observation details."""
        # Force batch_size to 1 for single-environment exploration as per DreamerAgent
        batch_size = 1
        
        # Create dummy features with batch_size=1
        dummy_features = torch.zeros(batch_size, self.feature_dimension, device=self.device)
        
        # Get action distribution and sample
        action_distribution = self.create_actor_distribution(dummy_features)
        
        with torch.no_grad():
            try:
                if self.has_discrete_actions:
                    action = action_distribution.sample()  # Shape: [1, action_dimension]
                    action_indices = torch.argmax(action, dim=-1)  # Shape: [1]
                    action_indices = action_indices.clamp(0, self.action_dimension - 1)
                    action_output = F.one_hot(action_indices, num_classes=self.action_dimension).float()  # Shape: [1, action_dimension]
                    log_probability = action_distribution.log_probability(action_indices)  # Shape: [1]
                else:
                    action = action_distribution.sample()  # Shape: [1, action_dimension]
                    action_output = torch.clamp(action, self.action_lower_bound, self.action_upper_bound)
                    log_probability = action_distribution.log_probability(action).sum(dim=-1)  # Shape: [1]
                
                if self.enable_debugging:
                    logger.debug(f"Action: {action_output}, Log probability: {log_probability}")
            except Exception as error:
                if self.enable_debugging:
                    logger.debug(f"Error in action sampling: {error}")
                if self.has_discrete_actions:
                    action_output = F.one_hot(torch.randint(0, self.action_dimension, (1,), device=self.device), num_classes=self.action_dimension).float()
                else:
                    action_output = torch.zeros((1, self.action_dimension), device=self.device)
                log_probability = torch.tensor(-float('inf'), device=self.device)
        
        policy_output = {"action": action_output, "log_probability": log_probability}
        
        if self.enable_debugging:
            logger.debug(f"RandomExplorer output: action_shape={action_output.shape}, log_prob_shape={log_probability.shape}")
        
        return policy_output, None  # No state update

    def train(
        self,
        posterior: Optional[Dict[str, torch.Tensor]] = None,
        context: Optional[Dict[str, torch.Tensor]] = None,
        data: Optional[Dict[str, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any
    ) -> Tuple[None, None, None, None, Dict[str, float]]:
        """Placeholder training method; random explorer does not learn."""
        return None, None, None, None, {"random_exploration_loss": 0.0}

    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract dummy features from state (returns zeros as placeholder)."""
        batch_size = state["deter"].shape[0] if "deter" in state else 1
        feature_dimension = state["deter"].shape[-1] + state["stoch"].shape[-1] if "deter" in state and "stoch" in state else self.feature_dimension
        device = state["deter"].device if "deter" in state else self.device
        return torch.zeros(batch_size, feature_dimension, device=device)

    @property
    def feature_dimension(self) -> int:
        """Dynamic feature dimension based on configuration or default."""
        return getattr(self.configuration, "feature_dimension", 32)  # Default to 32 if not specified