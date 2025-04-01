import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging
from gym.spaces import Discrete

# Setup logger for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

# ---------- Symlog Transformation Utilities ----------
def symmetric_logarithm_transformation(input_tensor: torch.Tensor) -> torch.Tensor:
    """Apply symmetric logarithm transformation to handle positive and negative values."""
    return torch.sign(input_tensor) * torch.log1p(torch.abs(input_tensor))

def inverse_symmetric_logarithm_transformation(transformed_tensor: torch.Tensor) -> torch.Tensor:
    """Reverse the symmetric logarithm transformation to recover original values."""
    return torch.sign(transformed_tensor) * torch.expm1(torch.abs(transformed_tensor))

def discretize_symmetric_logarithm(
    input_tensor: torch.Tensor,
    number_of_bins: int = 255,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0
) -> torch.Tensor:
    """Discretize a tensor using symlog transformation into bins."""
    transformed_values = symmetric_logarithm_transformation(input_tensor)
    clamped_values = torch.clamp(transformed_values, lower_bound, upper_bound)
    scaled_values = ((clamped_values - lower_bound) / (upper_bound - lower_bound) * (number_of_bins - 1))
    return scaled_values.long()

def create_two_hot_encoding(
    input_tensor: torch.Tensor,
    number_of_bins: int = 255,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0
) -> torch.Tensor:
    """Create a two-hot encoding of input values."""
    transformed_values = symmetric_logarithm_transformation(input_tensor)
    clamped_values = torch.clamp(transformed_values, lower_bound, upper_bound)
    scaled_values = ((clamped_values - lower_bound) / (upper_bound - lower_bound) * (number_of_bins - 1))
    
    lower_indices = torch.floor(scaled_values).long().clamp(0, number_of_bins - 1)
    upper_indices = torch.ceil(scaled_values).long().clamp(0, number_of_bins - 1)
    upper_weights = scaled_values - lower_indices.float()
    lower_weights = 1.0 - upper_weights
    
    shape = list(input_tensor.shape) + [number_of_bins]
    two_hot_tensor = torch.zeros(shape, device=input_tensor.device)
    
    flat_size = int(np.prod(input_tensor.shape))
    flat_lower_indices = lower_indices.reshape(-1)
    flat_upper_indices = upper_indices.reshape(-1)
    flat_lower_weights = lower_weights.reshape(-1)
    flat_upper_weights = upper_weights.reshape(-1)
    flat_two_hot = two_hot_tensor.reshape(flat_size, number_of_bins)
    
    batch_indices = torch.arange(flat_size, device=input_tensor.device)
    flat_two_hot[batch_indices, flat_lower_indices] = flat_lower_weights
    flat_two_hot[batch_indices, flat_upper_indices] += flat_upper_weights
    
    same_indices_mask = (flat_lower_indices == flat_upper_indices)
    flat_two_hot[batch_indices[same_indices_mask], flat_lower_indices[same_indices_mask]] = 1.0
    
    return two_hot_tensor

def undo_discretized_symmetric_logarithm(
    bin_indices: torch.Tensor,
    number_of_bins: int = 255,
    lower_bound: float = -20.0,
    upper_bound: float = 20.0
) -> torch.Tensor:
    """Convert discretized symlog bins back to real values."""
    if isinstance(bin_indices, int):
        bin_indices = torch.tensor(bin_indices, dtype=torch.float32)
    
    bin_values = bin_indices.float()
    transformed_values = bin_values / (number_of_bins - 1) * (upper_bound - lower_bound) + lower_bound
    return inverse_symmetric_logarithm_transformation(transformed_values)

# ---------- Distribution Classes ----------
class OneHotDistribution:
    """Distribution for one-hot categorical variables with uniform mixing."""
    def __init__(self, logits: torch.Tensor, temperature: float = 1.0, uniform_mix_ratio: float = 0.01) -> None:
        self.logits = logits
        self.temperature = temperature
        self.uniform_mix_ratio = uniform_mix_ratio
        self.num_classes = logits.shape[-1]
        self.distribution = distributions.Categorical(logits=self.apply_uniform_mixing(logits))

    def apply_uniform_mixing(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply uniform mixing for exploration."""
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        uniform_distribution = torch.ones_like(probabilities) / self.num_classes
        mixed_probabilities = (1 - self.uniform_mix_ratio) * probabilities + self.uniform_mix_ratio * uniform_distribution
        return torch.log(mixed_probabilities + 1e-8)

    def log_probability(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        if value.dim() == self.logits.dim() - 1:
            return self.distribution.log_prob(value)
        return self.distribution.log_prob(torch.argmax(value, dim=-1))

    def mode(self) -> torch.Tensor:
        """Return the most likely action."""
        return F.one_hot(torch.argmax(self.logits, dim=-1), num_classes=self.num_classes).float()

    def sample(self) -> torch.Tensor:
        """Sample an action."""
        sample_indices = self.distribution.sample()
        return F.one_hot(sample_indices, num_classes=self.num_classes).float()

    def reparameterized_sample(self) -> torch.Tensor:
        """Sample with Gumbel-Softmax reparameterization."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-10) + 1e-10)
        gumbel_logits = (self.logits + gumbel_noise) / self.temperature
        soft_probabilities = F.softmax(gumbel_logits, dim=-1)
        hard_sample = F.one_hot(torch.argmax(soft_probabilities, dim=-1), num_classes=self.num_classes).float()
        return hard_sample - soft_probabilities.detach() + soft_probabilities

    def entropy(self) -> torch.Tensor:
        """Compute entropy."""
        return self.distribution.entropy()

class DistributionWrapper:
    def __init__(self, logits: torch.Tensor, distribution_type: str, temperature: float = 1.0, uniform_mix_ratio: float = 0.01, number_of_bins: int = 256, lower_bound: float = -20.0, upper_bound: float = 20.0):
        self.logits = logits
        self.distribution_type = distribution_type.lower()
        self.temperature = temperature
        self.uniform_mix_ratio = uniform_mix_ratio
        self.number_of_bins = number_of_bins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        logger.debug(f"DistributionWrapper init - logits shape: {logits.shape}, type: {self.distribution_type}")

        if self.distribution_type == "normal":
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(log_std) + 1e-6
            self.distribution = distributions.Normal(mean, std)
        elif self.distribution_type == "categorical":
            scaled_logits = logits / temperature
            if uniform_mix_ratio > 0:
                probs = F.softmax(scaled_logits, dim=-1)
                uniform = torch.ones_like(probs) / logits.shape[-1]
                probs = (1 - uniform_mix_ratio) * probs + uniform_mix_ratio * uniform
                scaled_logits = torch.log(probs + 1e-8)
            self.distribution = distributions.Categorical(logits=scaled_logits)
        elif self.distribution_type == "symlog_disc":
            scaled_logits = logits / temperature
            if uniform_mix_ratio > 0:
                probs = F.softmax(scaled_logits, dim=-1)
                uniform = torch.ones_like(probs) / number_of_bins
                probs = (1 - uniform_mix_ratio) * probs + uniform_mix_ratio * uniform
                scaled_logits = torch.log(probs + 1e-8)
            self.distribution = distributions.Categorical(logits=scaled_logits)
        elif self.distribution_type == "onehot":
            self.distribution = distributions.Categorical(logits=logits / temperature)
        elif self.distribution_type in ["binary", "bernoulli"]:
            self.distribution = distributions.Bernoulli(logits=logits)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    def log_probability(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability."""
        if self.distribution_type == "categorical":
            if value.dim() != self.logits.dim():
                value = value.view(self.logits.shape[:-1])  # Remove class dimension if needed
            return self.distribution.log_prob(value)
        elif self.distribution_type == "symlog_disc":
            symlog_value = symmetric_logarithm_transformation(value)
            scaled_value = ((symlog_value - self.lower_bound) / (self.upper_bound - self.lower_bound) * (self.number_of_bins - 1)).long()
            return self.distribution.log_prob(scaled_value)
        elif self.distribution_type == "binary":
            return torch.where(value > 0.5, torch.log(torch.sigmoid(self.logits) + 1e-8), torch.log(1 - torch.sigmoid(self.logits) + 1e-8)).squeeze(-1)
        elif self.distribution_type == "normal":
            return self.distribution.log_prob(value).sum(dim=-1)
        try:
            return self.distribution.log_prob(value)
        except Exception as error:
            logger.error(f"Error in log_probability: {error}")
            return torch.zeros(value.shape[:-1], device=value.device)

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""
        sample = self.distribution.sample()
        if self.distribution_type == "symlog_disc":
            return undo_discretized_symmetric_logarithm(sample, self.number_of_bins, self.lower_bound, self.upper_bound)
        elif self.distribution_type in ["categorical", "onehot"]:
            return F.one_hot(sample, num_classes=self.logits.shape[-1]).float()
        return sample

    def reparameterized_sample(self) -> torch.Tensor:
        """Sample with reparameterization."""
        if self.distribution_type == "normal" and hasattr(self.distribution, 'rsample'):
            return self.distribution.rsample()
        elif self.distribution_type in ["categorical", "onehot", "symlog_disc"]:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-10) + 1e-10)
            gumbel_logits = (self.logits + gumbel_noise) / self.temperature
            soft_probabilities = F.softmax(gumbel_logits, dim=-1)
            if self.distribution_type == "symlog_disc":
                bin_values = torch.linspace(self.lower_bound, self.upper_bound, self.number_of_bins, device=self.logits.device)
                expected_symlog = torch.sum(soft_probabilities * bin_values, dim=-1)
                return inverse_symmetric_logarithm_transformation(expected_symlog)
            hard_sample = F.one_hot(torch.argmax(soft_probabilities, dim=-1), num_classes=self.logits.shape[-1]).float()
            return hard_sample - soft_probabilities.detach() + soft_probabilities
        return self.sample()

    def mode(self) -> torch.Tensor:
        """Return the most likely value."""
        if self.distribution_type == "normal":
            return self.distribution.mean
        elif self.distribution_type == "symlog_disc":
            mode = torch.argmax(self.logits, dim=-1)
            return undo_discretized_symmetric_logarithm(mode, self.number_of_bins, self.lower_bound, self.upper_bound)
        elif self.distribution_type in ["categorical", "onehot"]:
            return F.one_hot(torch.argmax(self.logits, dim=-1), num_classes=self.logits.shape[-1]).float()
        return (self.logits > 0).float()

    def mean(self) -> torch.Tensor:
        """Compute the expected value."""
        if self.distribution_type == "normal":
            return self.distribution.mean
        elif self.distribution_type == "symlog_disc":
            probs = F.softmax(self.logits, dim=-1)
            bin_values = torch.linspace(self.lower_bound, self.upper_bound, self.number_of_bins, device=self.logits.device)
            expected_symlog = torch.sum(probs * bin_values, dim=-1)
            return inverse_symmetric_logarithm_transformation(expected_symlog)
        elif self.distribution_type in ["categorical", "onehot"]:
            return F.softmax(self.logits, dim=-1)
        return torch.sigmoid(self.logits)

    def entropy(self) -> torch.Tensor:
        """Compute entropy."""
        return self.distribution.entropy()

# ---------- Orthogonal Initialization ----------
def apply_orthogonal_initialization(module: nn.Module, gain: float = 1.0) -> None:
    """Initialize weights using orthogonal initialization."""
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.GRUCell):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter, gain=gain)
            elif "bias" in name:
                nn.init.constant_(parameter, 0)

# ---------- Normalization Layers ----------
class RootMeanSquareNormalization(nn.Module):
    """Root Mean Square normalization layer."""
    def __init__(self, dimension: int, epsilon: float = 1e-8):
        super(RootMeanSquareNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor using RMS normalization."""
        rms = torch.sqrt(torch.mean(input_tensor ** 2, dim=-1, keepdim=True) + self.epsilon)
        return self.scale * input_tensor / rms

# ---------- MultiLayerPerceptron (MLP) ----------
class MultiLayerPerceptron(nn.Module):
    """Multi-layer perceptron network for DreamerV3 components."""
    def __init__(
        self,
        input_dimension: int,
        output_shape: Tuple[int, ...],
        number_of_layers: int = 5,
        units_per_layer: int = 512,
        activation_function: str = "silu",
        normalization_type: Optional[str] = "layer",
        distribution_type: Optional[str] = None,
        temperature: float = 1.0,
        uniform_mix_ratio: float = 0.01,
        device: str = "cpu",
        name: str = "Network",
        use_orthogonal_initialization: bool = True
    ) -> None:
        super(MultiLayerPerceptron, self).__init__()
        self.input_dimension = input_dimension
        self.output_shape = output_shape if isinstance(output_shape, tuple) else (output_shape,)
        self.number_of_layers = number_of_layers
        self.units_per_layer = units_per_layer
        self.activation_function = activation_function
        self.normalization_type = normalization_type
        self.distribution_type = distribution_type
        self.temperature = temperature
        self.uniform_mix_ratio = uniform_mix_ratio
        self.device = device
        self.name = name
        self.use_orthogonal_initialization = use_orthogonal_initialization

        # Activation function
        activation_map = {"silu": nn.SiLU(), "relu": nn.ReLU(), "elu": nn.ELU(), "tanh": nn.Tanh()}
        self.activation = activation_map.get(activation_function, nn.SiLU())

        # Normalization layer
        normalization_map = {"layer": nn.LayerNorm, "rms": RootMeanSquareNormalization, None: lambda x: nn.Identity()}
        normalization_class = normalization_map.get(normalization_type, nn.LayerNorm)

        # Build layers
        layers = []
        current_dimension = input_dimension
        for _ in range(number_of_layers - 1):  # Exclude output layer
            linear_layer = nn.Linear(current_dimension, units_per_layer)
            if use_orthogonal_initialization:
                apply_orthogonal_initialization(linear_layer)
            layers.append(linear_layer)
            layers.append(normalization_class(units_per_layer))
            layers.append(self.activation)
            current_dimension = units_per_layer
        
        # Output layer
        output_dimension = int(np.prod(self.output_shape))
        output_layer = nn.Linear(current_dimension, output_dimension)
        if use_orthogonal_initialization:
            apply_orthogonal_initialization(output_layer)
        layers.append(output_layer)

        self.layers = nn.Sequential(*layers)
        self.to(device)

    def forward(self, input_tensor: torch.Tensor) -> Any:
        output = self.layers(input_tensor)
        if self.distribution_type:
            batch_dims = input_tensor.shape[:-1]
            output = output.reshape(*batch_dims, *self.output_shape)  # Use reshape
            return DistributionWrapper(
                logits=output,
                distribution_type=self.distribution_type,
                temperature=self.temperature,
                uniform_mix_ratio=self.uniform_mix_ratio,
                number_of_bins=256 if self.distribution_type == "categorical" else 255,
                lower_bound=-20.0,
                upper_bound=20.0
            )
        return output

# ---------- MultiEncoder ----------
class MultiEncoder(nn.Module):
    """
    Multi-input encoder for processing observations in DreamerV3.
    Handles image-based observations with a convolutional architecture.
    """
    def __init__(
        self,
        input_shape: tuple,  # Expecting (H, W, C)
        configuration: Any,
    ) -> None:
        super(MultiEncoder, self).__init__()
        self.input_shape = input_shape  # (H, W, C), e.g., (64, 64, 3)
        self.channels = input_shape[2]  # Number of channels (e.g., 3)
        self.height = input_shape[0]
        self.width = input_shape[1]
        
        # Convolutional layers expecting [B, C, H, W]
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, configuration.units // 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(configuration.units // 4, configuration.units // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(configuration.units // 2, configuration.units, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(configuration.units, configuration.units * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_output_size = self._calculate_conv_output_size()
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, configuration.encoder_output_dimension),
            nn.ReLU()
        )

    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size of the convolutional layers."""
        x = torch.zeros(1, self.height, self.width, self.channels)  # [1, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [1, C, H, W]
        x = self.conv(x)
        return x.numel()

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the encoder."""
        x = observations["image"]
        if x.dim() == 5:  # [B, T, C, H, W]
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
        if x.dim() == 4 and x.shape[-1] == self.channels:  # [B, H, W, C]
            x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        conv_out = self.conv(x)
        flat = conv_out.reshape(conv_out.size(0), -1)
        embedding = self.fc(flat)
        if "T" in locals():
            embedding = embedding.reshape(B, T, -1)
        return embedding

# ---------- Decoder ----------
class Decoder(nn.Module):
    """
    Convolutional decoder for reconstructing observations in DreamerV3.
    Outputs mean and log-variance for a normal distribution over images.
    """
    def __init__(
        self,
        input_dimension: int,  # Latent feature dimension from dynamics
        output_shape: tuple,   # (C, H, W), e.g., (3, 64, 64)
        hidden_dimension: int = 128,  # Depth of conv layers, matching units
        number_of_layers: int = 4,
    ) -> None:
        super(Decoder, self).__init__()
        self.output_shape = output_shape  # (C, H, W)
        self.hidden_dimension = hidden_dimension
        
        # Initial dense layer to spatial grid (4x4 starting point)
        self.dense = nn.Linear(input_dimension, hidden_dimension * 4 * 4)
        
        # Common convolutional transpose layers
        common_layers = [
            nn.ReLU(),
            nn.Unflatten(-1, (hidden_dimension, 4, 4)),  # Reshape to [B, hidden_dim, 4, 4]
            nn.ConvTranspose2d(hidden_dimension, hidden_dimension // 2, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dimension // 2, hidden_dimension // 4, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dimension // 4, hidden_dimension // 8, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
        ]
        
        # Mean output
        self.mean_conv = nn.Sequential(
            *common_layers,
            nn.ConvTranspose2d(hidden_dimension // 8, output_shape[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        )
        
        # Log-variance output (clamped between -10 and 10)
        self.logvar_conv = nn.Sequential(
            *common_layers,
            nn.ConvTranspose2d(hidden_dimension // 8, output_shape[0], kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Output mean and log-variance for a normal distribution."""
        x = self.dense(x)
        mean = self.mean_conv(x)  # [B, C, H, W]
        logvar = self.logvar_conv(x)  # [B, C, H, W]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # Stabilize variance
        return mean, logvar
    
# ---------- RecurrentStateSpaceModel (RSSM) ----------
class RecurrentStateSpaceModel(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        recurrent_depth: int,
        use_discrete_latents: bool,
        activation_function: str,
        normalization_type: str,
        uniform_mix_ratio: float,
        action_dimension: int,
        observation_embedding_dimension: int,
        device: torch.device,
        use_orthogonal_initialization: bool,
        discrete_latent_num: int,
        discrete_latent_size: int,
        has_discrete_actions: bool,  # Added parameter
    ) -> None:
        super(RecurrentStateSpaceModel, self).__init__()
        self.hidden_units = hidden_units
        self.recurrent_depth = recurrent_depth
        self.use_discrete_latents = use_discrete_latents
        self.action_dimension = action_dimension
        self.observation_embedding_dimension = observation_embedding_dimension
        self.device = device
        self.discrete_latent_num = discrete_latent_num
        self.discrete_latent_size = discrete_latent_size
        self.has_discrete_actions = has_discrete_actions  # Set the attribute

        # Rest of your initialization code (e.g., GRU, prior_net, posterior_net) goes here
        # Example:
        self.stochastic_dimension = discrete_latent_num * discrete_latent_size if use_discrete_latents else hidden_units
        self.feature_dimension = hidden_units + self.stochastic_dimension
        self.gru = nn.GRU(
            input_size=action_dimension + self.stochastic_dimension,
            hidden_size=hidden_units,
            num_layers=recurrent_depth,
            batch_first=True
        )

        if use_orthogonal_initialization:
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)

        # Prior and posterior networks (assuming MultiLayerPerceptron is defined elsewhere)
        if use_discrete_latents:
            self.prior_net = MultiLayerPerceptron(
                input_dimension=hidden_units,
                output_shape=(discrete_latent_num * discrete_latent_size,),
                number_of_layers=2,
                units_per_layer=hidden_units,
                activation_function=activation_function,
                normalization_type=normalization_type,
                device=device,
                use_orthogonal_initialization=use_orthogonal_initialization
            )
            self.posterior_net = MultiLayerPerceptron(
                input_dimension=hidden_units + observation_embedding_dimension,
                output_shape=(discrete_latent_num * discrete_latent_size,),
                number_of_layers=2,
                units_per_layer=hidden_units,
                activation_function=activation_function,
                normalization_type=normalization_type,
                device=device,
                use_orthogonal_initialization=use_orthogonal_initialization
            )

        self.to(device)

    def observe_step(self, observation_embedding: torch.Tensor, action: torch.Tensor,
                     is_first: torch.Tensor, state: dict = None) -> tuple:
        # Adjust dimensions if necessary
        if observation_embedding.dim() == 2:
            observation_embedding = observation_embedding.unsqueeze(1)  # [B, D] -> [B, 1, D]
        if action.dim() == 2 and self.has_discrete_actions:
            action = action.unsqueeze(-1)  # [B, T] -> [B, T, 1] for discrete actions
        if is_first.dim() == 2:
            is_first = is_first.unsqueeze(1)  # [B, T] -> [B, 1, T]

        seq_len, batch_size = observation_embedding.shape[:2]

        # Initialize state if None
        if state is None:
            state = {
                "deter": torch.zeros(self.recurrent_depth, batch_size, self.hidden_units, device=self.device),
                "stoch": torch.zeros(batch_size, self.stochastic_dimension, device=self.device)
            }

        deter = state["deter"]
        stoch = state["stoch"]

        new_deters, new_stochs, posteriors, priors = [], [], [], []

        for t in range(seq_len):
            if is_first[t].any():
                deter = torch.zeros_like(deter)
                stoch = torch.zeros_like(stoch)

            # Prior distribution
            prior_logits = self.prior_net(deter[-1])
            prior_logits = prior_logits.view(batch_size, self.discrete_latent_num, self.discrete_latent_size)
            prior_dist = torch.distributions.Categorical(logits=prior_logits)

            # Posterior distribution
            post_input = torch.cat([deter[-1], observation_embedding[t]], dim=-1)
            post_logits = self.posterior_net(post_input)
            post_logits = post_logits.view(batch_size, self.discrete_latent_num, self.discrete_latent_size)
            post_dist = torch.distributions.Categorical(logits=post_logits)

            # Sample stochastic state
            stoch_sample = post_dist.sample()  # [B, num]
            stoch = F.one_hot(stoch_sample, num_classes=self.discrete_latent_size).float().view(batch_size, -1)  # [B, stoch_dim]

            # Process action based on action space type
            action_t = action[:, t] if action.shape[0] == batch_size else action[t]  # [B, action_dim]
            if self.has_discrete_actions:
                action_t = action_t.squeeze(-1)  # [B, 1] -> [B]
                action_t = F.one_hot(action_t.long(), num_classes=self.action_dimension).float()  # [B, action_dimension]

            # Construct GRU input
            gru_input = torch.cat([stoch, action_t], dim=-1).unsqueeze(1)  # [B, 1, stoch_dim + action_dim]
            _, deter = self.gru(gru_input, deter)

            # Store results
            new_deters.append(deter[-1])
            new_stochs.append(stoch)
            posteriors.append({"logits": post_logits})
            priors.append({"logits": prior_logits})

        # Prepare return values
        new_state = {
            "deter": torch.stack(new_deters),
            "stoch": torch.stack(new_stochs)
        }
        posterior = {"logits": torch.stack([p["logits"] for p in posteriors])}
        prior = {"logits": torch.stack([p["logits"] for p in priors])}

        return new_state, posterior, prior
    
    def get_features(self, state):
        deter = state['deter']  # Expected: [T, B, hidden_units]
        stoch = state['stoch']  # Expected: [T, B, stochastic_dimension]
        print(f"deter shape: {deter.shape}")
        print(f"stoch shape: {stoch.shape}")
        return torch.cat([deter, stoch], dim=-1)  # [T, B, feature_dim]
    
    def imagine_step(self, action: torch.Tensor, state: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], None]:
        """Perform one step of imagination in the latent space."""
        deter = state["hidden_state"]  # [num_layers, B, hidden_units]
        stoch = state["stoch"]  # [B, stochastic_dimension]
        
        logger.debug(f"imagine_step: Initial deter shape: {deter.shape}")
        logger.debug(f"imagine_step: Initial stoch shape: {stoch.shape}")
        logger.debug(f"imagine_step: Initial action shape: {action.shape}")
        
        # Predict prior distribution from deterministic state
        prior_logits = self.prior_net(deter[-1])  # [B, discrete_latent_num * discrete_latent_size]
        prior_logits = prior_logits.view(-1, self.discrete_latent_num, self.discrete_latent_size)  # [B, num, size]
        prior_dist = torch.distributions.Categorical(logits=prior_logits)
        
        # Sample next stochastic state
        stoch_sample = prior_dist.sample()  # [B, num]
        next_stoch = F.one_hot(stoch_sample, num_classes=self.discrete_latent_size).float().view(-1, self.stochastic_dimension)  # [B, stoch_dim]
        logger.debug(f"imagine_step: next_stoch shape: {next_stoch.shape}")
        
        # Process action
        if action.dim() > 2:
            action = action.squeeze(1)  # [B, T, A] -> [B, A] or [B, 1, A] -> [B, A]
            logger.debug(f"imagine_step: Action shape after squeeze: {action.shape}")
        if action.dim() != 2:
            logger.error(f"imagine_step: Action still not 2D after squeeze: {action.shape}")
            action = action.view(-1, self.action_dimension)  # Force [B, action_dim]
        logger.debug(f"imagine_step: Action shape final: {action.shape}")
        
        # Ensure batch size consistency
        if next_stoch.shape[0] != action.shape[0]:
            if next_stoch.shape[0] == 1:
                next_stoch = next_stoch.expand(action.shape[0], -1)
            elif action.shape[0] == 1:
                action = action.expand(next_stoch.shape[0], -1)
            else:
                raise ValueError(f"Batch size mismatch: next_stoch {next_stoch.shape[0]} vs action {action.shape[0]}")
        
        # GRU input
        gru_input = torch.cat([next_stoch, action], dim=-1).unsqueeze(1)  # [B, 1, stoch_dim + action_dim]
        logger.debug(f"imagine_step: gru_input shape: {gru_input.shape}")
        _, next_deter = self.gru(gru_input, deter)  # [num_layers, B, hidden_units]
        
        # Return next state
        next_state = {
            "hidden_state": next_deter,
            "stoch": next_stoch,
            "deter": next_deter[-1]
        }
        return next_state, None

# ---------- Reward Objective ----------
class RewardObjective(nn.Module):
    """Handles reward processing with symlog."""
    def __init__(self, alpha: float = 0.01):
        super(RewardObjective, self).__init__()
        self.alpha = alpha
        self.register_buffer("mean", torch.tensor(0.0))
        self.register_buffer("variance", torch.tensor(1.0))

    def forward(self, reward: torch.Tensor) -> torch.Tensor:
        """Process rewards."""
        with torch.no_grad():
            new_mean = self.alpha * reward.mean() + (1 - self.alpha) * self.mean
            new_variance = self.alpha * ((reward - new_mean) ** 2).mean() + (1 - self.alpha) * self.variance
            self.mean.copy_(new_mean)
            self.variance.copy_(new_variance.clamp(min=1e-8))
        return reward

# ---------- Lambda Return Target ----------
def lambda_return_target(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    lambda_discount_factor: float
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute lambda return targets."""
    if reward.dim() == 2:
        reward = reward.unsqueeze(-1)
    if value.dim() == 2:
        value = value.unsqueeze(-1)
    if discount.dim() == 2:
        discount = discount.unsqueeze(-1)
    
    reward = torch.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6)
    value = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
    discount = torch.nan_to_num(discount, nan=0.0, posinf=1.0, neginf=0.0)

    time_steps, batch_size = reward.shape[:2]
    targets, weights = [], torch.ones_like(reward)
    target = value[-1]

    for t in reversed(range(time_steps)):
        target = reward[t] + discount[t] * ((1 - lambda_discount_factor) * value[t] + lambda_discount_factor * target)
        target = torch.clamp(target, min=-100.0, max=100.0)
        targets.insert(0, target)

    return targets, weights, value