# Decodes latent states back into observations for reconstruction loss
import torch
import torch.nn as nn

class ObservationDecoder(nn.Module):
    def __init__(self, latent_size=32, output_channels=3):
        super(ObservationDecoder, self).__init__()
        self.latent_size = latent_size
        
        # Project latent to dense features
        self.dense_network = nn.Sequential(
            nn.Linear(latent_size, 256 * 4 * 4),
            nn.ReLU()
        )
        
        # Deconvolutional layers to reconstruct 64x64 images
        self.deconv_network = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, latent_state_batch):
        """
        Decode latent states into reconstructed observations.
        
        Args:
            latent_state_batch (torch.Tensor): [batch, seq, latent_size]
        
        Returns:
            reconstructed_observations (torch.Tensor): [batch, seq, channels, height, width]
        """
        batch_size, sequence_length = latent_state_batch.shape[:2]
        flat_batch = latent_state_batch.view(batch_size * sequence_length, -1)
        
        # Project to dense features
        dense_features = self.dense_network(flat_batch)
        dense_features = dense_features.view(-1, 256, 4, 4)
        
        # Decode to observation space
        reconstructed_observations = self.deconv_network(dense_features)
        
        # Reshape to sequence format
        return reconstructed_observations.view(batch_size, sequence_length, *reconstructed_observations.shape[1:])