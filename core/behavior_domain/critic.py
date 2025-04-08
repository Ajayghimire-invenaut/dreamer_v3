# Implements the critic network for value estimation
import torch
import torch.nn as nn

class ValueCritic(nn.Module):
    def __init__(self, latent_size=32, hidden_size=1024):
        super(ValueCritic, self).__init__()
        self.latent_size = latent_size
        
        # Critic network to estimate value function
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Value estimate
        )

    def forward(self, latent_state_batch):
        """
        Estimate value from latent states.
        
        Args:
            latent_state_batch (torch.Tensor): [batch, seq, latent_size]
        
        Returns:
            value_estimate (torch.Tensor): [batch, seq, 1]
        """
        return self.network(latent_state_batch)