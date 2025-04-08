# Implements the recurrent dynamics model for predicting next latent states
import torch
import torch.nn as nn

class WorldDynamics(nn.Module):
    def __init__(self, latent_size=32, action_size=2, hidden_size=1024, stoch_size=32, deter_size=1024):
        super(WorldDynamics, self).__init__()
        self.latent_size = latent_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.action_size = action_size
        
        # GRU for deterministic state transitions
        self.recurrent_network = nn.GRUCell(
            input_size=stoch_size + action_size,
            hidden_size=deter_size
        )
        
        # Prior network
        self.prior_network = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stoch_size * 2)
        )
        
        # Posterior network
        self.posterior_network = nn.Sequential(
            nn.Linear(deter_size + latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, stoch_size * 2)
        )

    def forward(self, previous_stochastic_state, previous_deterministic_state, action, encoded_observation=None):
        """
        Predict the next latent state (stochastic + deterministic).
        
        Args:
            previous_stochastic_state (torch.Tensor): [batch, stoch_size]
            previous_deterministic_state (torch.Tensor): [batch, deter_size]
            action (torch.Tensor): [batch, action_size] or [batch, seq, action_size]
            encoded_observation (torch.Tensor, optional): [batch, latent_size] for posterior
        
        Returns:
            stochastic_state (torch.Tensor): [batch, stoch_size]
            deterministic_state (torch.Tensor): [batch, deter_size]
            posterior_dist (torch.Tensor): [batch, stoch_size * 2]
            prior_dist (torch.Tensor): [batch, stoch_size * 2]
        """
        # Ensure action has 2 dimensions (batch, action_size)
        if action.dim() == 3:  # [batch, seq, action_size]
            action = action.squeeze(1)  # Take first seq dim if needed, or adjust caller
        elif action.dim() != 2:
            raise ValueError(f"Action must have 2 or 3 dimensions, got {action.shape}")
        
        # Combine previous stochastic state and action
        recurrent_input = torch.cat([previous_stochastic_state, action], dim=-1)
        
        # Update deterministic state
        deterministic_state = self.recurrent_network(recurrent_input, previous_deterministic_state)
        
        # Predict prior distribution
        prior_distribution = self.prior_network(deterministic_state)
        prior_mean, prior_log_std = prior_distribution.chunk(2, dim=-1)
        prior_std = torch.exp(prior_log_std.clamp(-4, 4))
        prior_sample = prior_mean + prior_std * torch.randn_like(prior_mean)
        
        # Compute posterior if observation provided
        if encoded_observation is not None:
            posterior_input = torch.cat([deterministic_state, encoded_observation], dim=-1)
            posterior_distribution = self.posterior_network(posterior_input)
            posterior_mean, posterior_log_std = posterior_distribution.chunk(2, dim=-1)
            posterior_std = torch.exp(posterior_log_std.clamp(-4, 4))
            stochastic_state = posterior_mean + posterior_std * torch.randn_like(posterior_mean)
        else:
            posterior_distribution = prior_distribution
            stochastic_state = prior_sample
        
        return stochastic_state, deterministic_state, posterior_distribution, prior_distribution

    def initial_state(self, batch_size, device):
        """Initialize zero states for a batch."""
        return (torch.zeros(batch_size, self.stoch_size, device=device),
                torch.zeros(batch_size, self.deter_size, device=device))