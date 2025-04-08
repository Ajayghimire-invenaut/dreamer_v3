# Implements the actor network for policy learning
import torch
import torch.nn as nn
import torch.distributions as dist

class PolicyActor(nn.Module):
    def __init__(self, latent_size=32, action_size=2, hidden_size=1024):
        super(PolicyActor, self).__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        
        # Actor network to predict action distribution
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size * 2)  # Mean and log_std for action distribution
        )

    def forward(self, latent_state_batch):
        """
        Predict action distribution from latent states.
        
        Args:
            latent_state_batch (torch.Tensor): [batch, seq, latent_size]
        
        Returns:
            action_distribution (torch.distributions.Normal): Action distribution
            sampled_actions (torch.Tensor): Sampled actions [batch, seq, action_size]
        """
        # Predict action parameters
        action_params = self.network(latent_state_batch)
        mean, log_std = action_params.chunk(2, dim=-1)
        std = torch.exp(log_std.clamp(-4, 4))  # Stabilize variance
        
        # Create normal distribution for actions
        action_distribution = dist.Normal(mean, std)
        sampled_actions = action_distribution.rsample()  # Reparameterized sampling
        
        return action_distribution, sampled_actions

    def act(self, latent_state, deterministic=False):
        """
        Select an action for inference.
        
        Args:
            latent_state (torch.Tensor): [batch, latent_size]
            deterministic (bool): If True, use mean action
        
        Returns:
            action (torch.Tensor): [batch, action_size]
        """
        latent_state_batch = latent_state.unsqueeze(1)  # Add seq dim
        action_dist, actions = self.forward(latent_state_batch)
        return action_dist.mean.squeeze(1) if deterministic else actions.squeeze(1)