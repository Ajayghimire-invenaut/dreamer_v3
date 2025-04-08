# Predicts rewards from latent states
import torch
import torch.nn as nn

class RewardPredictor(nn.Module):
    def __init__(self, latent_size=32, hidden_size=1024):
        super(RewardPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Single reward value
        )

    def forward(self, latent_state_batch):
        """
        Predict rewards from latent states.
        
        Args:
            latent_state_batch (torch.Tensor): [batch, seq, latent_size]
        
        Returns:
            predicted_rewards (torch.Tensor): [batch, seq, 1]
        """
        return self.network(latent_state_batch)