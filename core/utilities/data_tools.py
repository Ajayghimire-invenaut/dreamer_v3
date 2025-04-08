# Provides utility functions for data processing and loss computation
import torch
import torch.nn.functional as F

class DataTools:
    @staticmethod
    def symlog(x):
        """
        Apply symmetric logarithmic transformation to stabilize reward distributions.
        Matches DreamerV3's handling of rewards (from tools.py in NM512/dreamerv3-torch).
        
        Args:
            x (torch.Tensor): Input tensor (e.g., rewards)
        
        Returns:
            torch.Tensor: Symlog-transformed tensor
        """
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x):
        """
        Inverse of symlog for decoding predictions.
        
        Args:
            x (torch.Tensor): Symlog-transformed tensor
        
        Returns:
            torch.Tensor: Original-scale tensor
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    @staticmethod
    def preprocess_observation(observation, device, image_range=(-0.5, 0.5)):
        """
        Preprocess image observations (e.g., DMC 64x64x3) to normalized range.
        
        Args:
            observation (torch.Tensor): [batch, seq, channels, height, width]
            device (torch.device): Target device
            image_range (tuple): Target range (min, max)
        
        Returns:
            torch.Tensor: Normalized observation
        """
        observation = observation.to(device, dtype=torch.float32)
        min_val, max_val = image_range
        return (observation / 255.0) * (max_val - min_val) + min_val

    @staticmethod
    def compute_kl_loss(posterior_distribution, prior_distribution, free_nats=1.0):
        """
        Compute KL divergence loss between posterior and prior distributions.
        Matches DreamerV3's KL balancing (from NM512/dreamerv3-torch).
        
        Args:
            posterior_distribution (torch.Tensor): [batch, seq, stoch_size * 2]
            prior_distribution (torch.Tensor): [batch, seq, stoch_size * 2]
            free_nats (float): Free nats for KL regularization
        
        Returns:
            torch.Tensor: KL loss [batch, seq]
        """
        post_mean, post_log_std = posterior_distribution.chunk(2, dim=-1)
        prior_mean, prior_log_std = prior_distribution.chunk(2, dim=-1)
        post_std = torch.exp(post_log_std)
        prior_std = torch.exp(prior_log_std)
        
        kl_div = 0.5 * (
            prior_log_std - post_log_std +
            (post_std ** 2 + (post_mean - prior_mean) ** 2) / (2 * prior_std ** 2) - 1
        ).sum(dim=-1)
        
        # Apply free nats regularization
        return torch.maximum(kl_div, torch.tensor(free_nats, device=kl_div.device))

    @staticmethod
    def compute_lambda_returns(rewards, values, discount=0.99, lambda_=0.95):
        """
        Compute λ-returns for critic training, as per DreamerV3 paper.
        
        Args:
            rewards (torch.Tensor): [batch, seq, 1]
            values (torch.Tensor): [batch, seq, 1]
            discount (float): Discount factor
            lambda_ (float): Lambda for GAE-like returns
        
        Returns:
            torch.Tensor: λ-returns [batch, seq, 1]
        """
        batch_size, seq_length = rewards.shape[:2]
        returns = torch.zeros_like(rewards)
        last_return = values[:, -1]  # Bootstrap from last value
        
        for t in reversed(range(seq_length)):
            returns[:, t] = rewards[:, t] + discount * (
                (1 - lambda_) * values[:, t] + lambda_ * last_return
            )
            last_return = returns[:, t]
        
        return returns

    @staticmethod
    def sequence_to_batch(tensor):
        """
        Convert sequence tensor to flat batch for processing.
        
        Args:
            tensor (torch.Tensor): [batch, seq, ...]
        
        Returns:
            torch.Tensor: [batch * seq, ...]
        """
        batch_size, seq_length = tensor.shape[:2]
        return tensor.view(batch_size * seq_length, *tensor.shape[2:])

    @staticmethod
    def batch_to_sequence(tensor, batch_size, seq_length):
        """
        Convert flat batch back to sequence format.
        
        Args:
            tensor (torch.Tensor): [batch * seq, ...]
            batch_size (int): Original batch size
            seq_length (int): Original sequence length
        
        Returns:
            torch.Tensor: [batch, seq, ...]
        """
        return tensor.view(batch_size, seq_length, *tensor.shape[1:])