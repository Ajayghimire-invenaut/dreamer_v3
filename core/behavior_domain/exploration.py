# Implements exploration strategy (optional, as DreamerV3 relies on imagination)
import torch

class ExplorationStrategy:
    def __init__(self, action_size=2, noise_scale=0.1):
        self.action_size = action_size
        self.noise_scale = noise_scale

    def add_exploration_noise(self, actions, device):
        """
        Add Gaussian noise to actions for exploration.
        
        Args:
            actions (torch.Tensor): [batch, seq, action_size]
            device (torch.device): Device for tensor operations
        
        Returns:
            noisy_actions (torch.Tensor): [batch, seq, action_size]
        """
        noise = torch.randn_like(actions, device=device) * self.noise_scale
        return (actions + noise).clamp(-1, 1)  # Assuming normalized action space [-1, 1]

    def intrinsic_reward(self, latent_state_prior, latent_state_posterior):
        """
        Compute intrinsic reward based on KL divergence (optional exploration boost).
        
        Args:
            latent_state_prior (torch.Tensor): [batch, seq, stoch_size * 2]
            latent_state_posterior (torch.Tensor): [batch, seq, stoch_size * 2]
        
        Returns:
            intrinsic_reward (torch.Tensor): [batch, seq, 1]
        """
        prior_mean, prior_log_std = latent_state_prior.chunk(2, dim=-1)
        post_mean, post_log_std = latent_state_posterior.chunk(2, dim=-1)
        kl_div = 0.5 * (prior_log_std - post_log_std + 
                        (torch.exp(post_log_std) ** 2 + (post_mean - prior_mean) ** 2) / 
                        (2 * torch.exp(prior_log_std) ** 2) - 1)
        return kl_div.sum(dim=-1, keepdim=True) * 0.1  # Scale for balance