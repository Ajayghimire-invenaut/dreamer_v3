# Encodes observations into a latent space for the posterior
import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    def __init__(self, input_channels=3, latent_size=1024):
        super(ObservationEncoder, self).__init__()
        self.latent_size = latent_size
        
        # Convolutional encoder for image observations (e.g., DMC 64x64x3)
        self.conv_network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dense layer to project to latent size
        self.dense_network = nn.Linear(256 * 4 * 4, latent_size)  # 64x64 -> 4x4 after convs

    def forward(self, observation_batch):
        """
        Encode observations into a latent representation.
        
        Args:
            observation_batch (torch.Tensor): [batch, seq, channels, height, width]
        
        Returns:
            encoded_representation (torch.Tensor): [batch, seq, latent_size]
        """
        batch_size, sequence_length = observation_batch.shape[:2]
        flat_batch = observation_batch.view(batch_size * sequence_length, *observation_batch.shape[2:])
        
        # Process through conv layers
        conv_features = self.conv_network(flat_batch)
        
        # Project to latent space
        encoded_representation = self.dense_network(conv_features)
        
        # Reshape back to sequence format
        return encoded_representation.view(batch_size, sequence_length, self.latent_size)