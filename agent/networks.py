"""
Neural network modules for Dreamer-V3.
Includes MultiEncoder, RSSM, MultiDecoder, MLP, and DistributionWrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import math  # Needed for entropy calculation

def orthogonal_initialize(module: nn.Module, gain: Optional[float] = None) -> None:
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class MultiEncoder(nn.Module):
    def __init__(self,
                 input_shapes: Dict[str, Tuple[int, ...]],
                 output_dimension: int,
                 use_orthogonal: bool = False) -> None:
        super(MultiEncoder, self).__init__()
        if "image" in input_shapes and len(input_shapes["image"]) == 3:
            self.is_image = True
            in_channels = input_shapes["image"][2]  # (H, W, C) → C channels
            self.convolutional_layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
                nn.ReLU()
            )
            # Expected output shape: (256, 2, 2) → flattened = 256*2*2 = 1024.
            self.fc = nn.Linear(256 * 2 * 2, output_dimension)
            if use_orthogonal:
                self.convolutional_layers.apply(orthogonal_initialize)
                orthogonal_initialize(self.fc)
        else:
            self.is_image = False
            if "image" in input_shapes:
                input_dim = input_shapes["image"][0]
            else:
                first_key = list(input_shapes.keys())[0]
                input_dim = input_shapes[first_key][0]
            self.fc = nn.Linear(input_dim, output_dimension)
            if use_orthogonal:
                orthogonal_initialize(self.fc)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.is_image:
            image = observations["image"]
            if getattr(observations, "debug", False):
                print("[DEBUG MultiEncoder] received image shape:", image.shape, flush=True)
            if image.dim() == 5:
                if image.shape[-1] in [1, 3]:
                    image = image.permute(0, 1, 4, 2, 3)
                    print("[DEBUG MultiEncoder] permuted 5D image shape:", image.shape, flush=True)
                B, T, C, H, W = image.shape
                image = image.reshape(B * T, C, H, W)
                features = self.convolutional_layers(image)
                flattened = features.reshape(features.size(0), -1)
                out = self.fc(flattened)
                out = out.reshape(B, T, -1)
                print("[DEBUG MultiEncoder] output shape (batched sequence):", out.shape, flush=True)
                return out
            elif image.dim() == 3:
                if image.shape[-1] in [1, 3]:
                    image = image.permute(2, 0, 1)
                image = image.unsqueeze(0)
            elif image.dim() == 4 and image.shape[-1] in [1, 3]:
                image = image.permute(0, 3, 1, 2)
            features = self.convolutional_layers(image)
            flattened = features.reshape(features.size(0), -1)
            out = self.fc(flattened)
            print("[DEBUG MultiEncoder] output shape (batched):", out.shape, flush=True)
            return out
        else:
            return self.fc(observations["image"])

class RSSM(nn.Module):
    def __init__(self,
                 stoch_dimension: int,
                 deter_dimension: int,
                 hidden_units: int,
                 rec_depth: int,
                 use_discrete: bool,
                 activation_function: str,
                 normalization_type: str,
                 mean_activation: str,
                 std_activation: str,
                 min_std: float,
                 unimix_ratio: float,
                 initial_state_type: str,
                 number_of_actions: int,
                 embedding_dimension: int,
                 device: str,
                 use_orthogonal: bool = False,
                 discrete_latent_num: int = 32,
                 discrete_latent_size: int = 32) -> None:
        super(RSSM, self).__init__()
        self.use_discrete = use_discrete
        self.device = device
        self.number_of_actions = number_of_actions
        self.deter_dimension = deter_dimension
        self.hidden_units = hidden_units
        self.rec_depth = rec_depth
        self.gru = nn.GRU(input_size=embedding_dimension + number_of_actions,
                          hidden_size=deter_dimension, batch_first=True)
        if use_discrete:
            self.discrete_latent_num = discrete_latent_num
            self.discrete_latent_size = discrete_latent_size
            # Output logits for discrete latent variables.
            self.logits_layer = nn.Linear(deter_dimension, discrete_latent_num * discrete_latent_size)
        else:
            self.mean_layer = nn.Linear(deter_dimension, stoch_dimension)
            self.std_layer = nn.Linear(deter_dimension, stoch_dimension)
            self.minimum_std = min_std
        if use_orthogonal:
            self.gru.apply(orthogonal_initialize)
            if self.use_discrete:
                orthogonal_initialize(self.logits_layer)
            else:
                orthogonal_initialize(self.mean_layer)
                orthogonal_initialize(self.std_layer)

    def observe(self,
                embeddings: torch.Tensor,
                actions: torch.Tensor,
                is_first: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        combined = torch.cat([embeddings, actions], dim=-1)
        outputs, _ = self.gru(combined)
        deter = outputs  # [T, B, deter_dimension]
        if self.use_discrete:
            logits = self.logits_layer(outputs)
            T, B, _ = logits.shape
            logits = logits.view(T, B, self.discrete_latent_num, self.discrete_latent_size)
            posterior = {"deter": deter, "logits": logits}
            prior = posterior  # In a full implementation, prior is computed separately.
        else:
            mean = self.mean_layer(outputs)
            std = F.softplus(self.std_layer(outputs)) + self.minimum_std
            posterior = {"deter": deter, "mean": mean, "std": std}
            prior = posterior
        return posterior, prior

    def observe_step(self,
                     previous_state: Optional[Dict[str, torch.Tensor]],
                     previous_action: Optional[torch.Tensor],
                     embedding: torch.Tensor,
                     is_first: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], None]:
        # Ensure embedding is 2D (B, D)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        # If embedding has an extra time dimension of size 1, squeeze it.
        if embedding.dim() == 3 and embedding.size(1) == 1:
            embedding = embedding.squeeze(1)
            print("[DEBUG observe_step] Squeezed embedding to shape:", embedding.shape, flush=True)
        
        # If previous_action is provided as a 3D tensor (B, T, A), take only the last time step.
        if previous_action is not None and previous_action.dim() == 3:
            print(f"[DEBUG observe_step] previous_action before squeezing: {previous_action.shape}", flush=True)
            previous_action = previous_action[:, -1, :]
            print(f"[DEBUG observe_step] previous_action after squeezing: {previous_action.shape}", flush=True)
        elif previous_action is None:
            previous_action = torch.zeros(embedding.size(0), self.number_of_actions, device=embedding.device)
        elif previous_action.dim() == 1:
            previous_action = torch.nn.functional.one_hot(previous_action.long(), num_classes=self.number_of_actions).float()
        elif previous_action.dim() < 2:
            previous_action = previous_action.unsqueeze(0)
            previous_action = torch.nn.functional.one_hot(previous_action.long(), num_classes=self.number_of_actions).float()

        print(f"[DEBUG observe_step] embedding shape: {embedding.shape}, previous_action shape: {previous_action.shape}", flush=True)
        combined = torch.cat([embedding, previous_action], dim=-1)
        print(f"[DEBUG observe_step] combined shape before unsqueeze: {combined.shape}", flush=True)
        combined = combined.unsqueeze(1)  # shape becomes (B, 1, D+A)
        print(f"[DEBUG observe_step] combined shape after unsqueeze: {combined.shape}", flush=True)
        
        output, _ = self.gru(combined)
        deter = output.squeeze(1)
        
        if self.use_discrete:
            logits = self.logits_layer(deter)
            B = logits.shape[0]
            logits = logits.view(B, self.discrete_latent_num, self.discrete_latent_size)
            new_state = {"deter": deter, "logits": logits}
        else:
            mean = self.mean_layer(deter)
            std = F.softplus(self.std_layer(deter)) + self.minimum_std
            new_state = {"deter": deter, "mean": mean, "std": std}
        return new_state, None

    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_discrete:
            # Compute probabilities from logits.
            probs = torch.softmax(state["logits"], dim=-1)
            print(f"[DEBUG get_features] raw probs shape: {probs.shape}", flush=True)
            if probs.ndim == 4:
                # Expected shape: [B, T, dnum, dsize]
                B, T, dnum, dsize = probs.shape
                try:
                    probs_flat = probs.reshape(B, T, -1)
                    print(f"[DEBUG get_features] probs_flat shape: {probs_flat.shape}", flush=True)
                except Exception as e:
                    print("[ERROR get_features] Cannot reshape probs:", e, flush=True)
                    raise
            elif probs.ndim == 3:
                # If time dimension is missing, add one.
                probs = probs.unsqueeze(1)
                probs_flat = probs.reshape(probs.shape[0], 1, -1)
                print(f"[DEBUG get_features] Added time dimension, probs_flat shape: {probs_flat.shape}", flush=True)
            else:
                probs_flat = probs
            # Get deterministic state.
            deter = state["deter"]
            if deter.ndim == 2:
                # Add time dimension of size 1.
                deter = deter.unsqueeze(1)
                print(f"[DEBUG get_features] Expanded 'deter' to shape: {deter.shape}", flush=True)
            # If time dimensions differ, broadcast the one with time=1.
            if deter.shape[1] != probs_flat.shape[1]:
                if deter.shape[1] == 1:
                    deter = deter.expand(deter.shape[0], probs_flat.shape[1], deter.shape[2])
                    print(f"[DEBUG get_features] Broadcasted 'deter' to shape: {deter.shape}", flush=True)
                elif probs_flat.shape[1] == 1:
                    probs_flat = probs_flat.expand(probs_flat.shape[0], deter.shape[1], probs_flat.shape[2])
                    print(f"[DEBUG get_features] Broadcasted 'probs_flat' to shape: {probs_flat.shape}", flush=True)
            # Final feature concatenation.
            features = torch.cat([deter, probs_flat], dim=-1)
            print(f"[DEBUG get_features] features shape: {features.shape}", flush=True)
            return features
        if "deter" not in state:
            print("Warning: 'deter' not in state. Keys present:", state.keys(), flush=True)
            return state["mean"]
        return torch.cat([state["deter"], state["mean"]], dim=-1)

    def imagine_with_action(self,
                            actions: torch.Tensor,
                            initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        horizon = actions.shape[1]
        repeated_state = {}
        for key in initial_state:
            repeated_state[key] = initial_state[key].unsqueeze(1).repeat(1, horizon, 1)
        return repeated_state

    def compute_kl_loss(self,
                        posterior: Dict[str, torch.Tensor],
                        prior: Dict[str, torch.Tensor],
                        kl_free: float,
                        dynamics_scale: float,
                        representation_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_discrete:
            post_logits = posterior["logits"]
            prior_logits = prior["logits"]
            post = torch.softmax(post_logits, dim=-1)
            prior = torch.softmax(prior_logits, dim=-1)
            # Compute categorical KL divergence.
            kl = post * (torch.log(post + 1e-8) - torch.log(prior + 1e-8))
            kl = kl.sum(dim=-1).mean()
        else:
            mean_diff = posterior["mean"] - prior["mean"]
            var_ratio = (posterior["std"] / prior["std"])**2
            kl = 0.5 * (var_ratio + (mean_diff**2) / (prior["std"]**2) - 1 - torch.log(var_ratio + 1e-8))
            kl = kl.mean()
        kl_loss = torch.clamp(kl, min=kl_free)
        return kl_loss, kl, dynamics_scale, representation_scale

class MultiDecoder(nn.Module):
    def __init__(self,
                 feature_dimension: int,
                 output_shapes: Dict[str, Tuple[int, ...]],
                 dummy_parameter: Any,
                 use_orthogonal: bool = False) -> None:
        super(MultiDecoder, self).__init__()
        raw_output_shape = output_shapes["image"]
        if len(raw_output_shape) == 3 and raw_output_shape[-1] <= 4:
            self.output_shape = (raw_output_shape[-1], raw_output_shape[0], raw_output_shape[1])
        else:
            self.output_shape = raw_output_shape
        if getattr(self, "debug", False):
            print("[DEBUG MultiDecoder] output_shape set to:", self.output_shape, flush=True)
        if len(self.output_shape) == 3:
            self.decoder_type = "conv"
            self.fc = nn.Linear(feature_dimension, 256 * 4 * 4)
            self.deconv_layers = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.output_shape[0], kernel_size=4, stride=2, padding=1)
            )
            if use_orthogonal:
                orthogonal_initialize(self.fc)
                self.deconv_layers.apply(orthogonal_initialize)
            self.debug = False
        else:
            self.decoder_type = "mlp"
            output_dim = 1
            for dim in self.output_shape:
                output_dim *= dim
            self.decoder = nn.Sequential(
                nn.Linear(feature_dimension, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim)
            )
            if use_orthogonal:
                self.decoder.apply(orthogonal_initialize)

    def forward(self, features: torch.Tensor) -> Dict[str, Any]:
        if self.decoder_type == "conv":
            if features.dim() == 3:
                B, T, D = features.shape
                if self.debug:
                    print("[DEBUG MultiDecoder] conv branch input features shape (B, T, D):", features.shape, flush=True)
                features = features.reshape(B * T, D)
                x = self.fc(features)
                if self.debug:
                    print("[DEBUG MultiDecoder] after fc shape:", x.shape, flush=True)
                x = x.reshape(B * T, 256, 4, 4)
                if self.debug:
                    print("[DEBUG MultiDecoder] reshaped to:", x.shape, flush=True)
                reconstruction = self.deconv_layers(x)
                if self.debug:
                    print("[DEBUG MultiDecoder] after deconv shape:", reconstruction.shape, flush=True)
                reconstruction = reconstruction.reshape(B, T, *reconstruction.shape[1:])
                if self.debug:
                    print("[DEBUG MultiDecoder] final reconstruction shape:", reconstruction.shape, flush=True)
            else:
                x = self.fc(features)
                x = x.reshape(features.size(0), 256, 4, 4)
                reconstruction = self.deconv_layers(x)
            return {"image": DistributionWrapper(reconstruction)}
        else:
            if features.dim() == 3:
                B, T, D = features.shape
                features = features.reshape(B * T, D)
                out = self.decoder(features)
                out = out.reshape(B, T, *self.output_shape)
            else:
                out = self.decoder(features)
            return {"image": DistributionWrapper(out)}

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_shape: Tuple[int, ...],
                 layers: int,
                 units: int,
                 activation: str,
                 normalization: str,
                 dist: str = "gaussian",
                 std: Optional[float] = None,
                 min_std: Optional[float] = None,
                 max_std: Optional[float] = None,
                 absmax: float = 1.0,
                 temperature: float = 1.0,
                 unimix_ratio: float = 0.0,
                 outscale: float = 1.0,
                 device: str = "cpu",
                 name: str = "MLP",
                 use_orthogonal: bool = False) -> None:
        super(MLP, self).__init__()
        dims = [input_dim] + [units] * layers + [output_shape[0] if output_shape else 1]
        modules: List[nn.Module] = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i+1])
            if use_orthogonal:
                orthogonal_initialize(linear)
            modules.append(linear)
            if i < len(dims) - 2:
                modules.append(nn.ReLU())
        self.network = nn.Sequential(*modules)
        self.distribution_type = dist
        self.outscale = outscale

    def forward(self, x: torch.Tensor) -> Any:
        output = self.network(x)
        return DistributionWrapper(output, dist_type=self.distribution_type)

class DistributionWrapper:
    def __init__(self, logits: torch.Tensor, dist_type: str = "gaussian") -> None:
        self.logits = logits
        self.dist_type = dist_type
        print(f"[DEBUG DistributionWrapper] initialized with logits shape: {self.logits.shape} and type: {self.dist_type}", flush=True)

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if self.dist_type == "gaussian":
            if target.dim() == self.logits.dim() - 1:
                target = target.unsqueeze(-1)
            if self.logits.shape[0] != target.shape[0]:
                target = target.transpose(0, 1)
            return -((self.logits - target) ** 2).mean()
        elif self.dist_type == "symlog_disc":
            target = target.long().squeeze(-1)
            logits = self.logits.reshape(-1, self.logits.shape[-1])
            return -nn.functional.cross_entropy(logits, target.reshape(-1), reduction='mean')
        elif self.dist_type == "binary":
            return -nn.functional.binary_cross_entropy_with_logits(self.logits, target, reduction='mean')
        else:
            raise ValueError("Unsupported distribution type")

    def mode(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            return self.logits
        elif self.dist_type == "symlog_disc":
            mode = torch.argmax(self.logits, dim=-1)
            return mode.float()
        elif self.dist_type == "binary":
            return (self.logits >= 0).float()
        else:
            raise ValueError("Unsupported distribution type")

    def sample(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            noise = torch.randn_like(self.logits) * 0.1
            return self.logits + noise
        elif self.dist_type == "symlog_disc":
            probs = torch.softmax(self.logits, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs)
            return distribution.sample().float()
        elif self.dist_type == "binary":
            probs = torch.sigmoid(self.logits)
            return torch.bernoulli(probs)
        else:
            raise ValueError("Unsupported distribution type")

    def entropy(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            sigma = 0.1
            entropy_value = 0.5 * math.log(2 * math.pi * math.e * (sigma ** 2))
            return torch.full_like(self.logits, entropy_value)
        elif self.dist_type == "symlog_disc":
            probs = torch.softmax(self.logits, dim=-1)
            return torch.distributions.Categorical(probs=probs).entropy()
        elif self.dist_type == "binary":
            p = torch.sigmoid(self.logits)
            entropy = - p * torch.log(p + 1e-8) - (1 - p) * torch.log(1 - p + 1e-8)
            return entropy
        else:
            raise ValueError("Unsupported distribution type")

class RewardEMA:
    def __init__(self, device: Any, alpha: float = 1e-2) -> None:
        self.device = device
        self.alpha = alpha
        self.quantile_range = torch.tensor([0.05, 0.95], device=device)
    def __call__(self, input_tensor: torch.Tensor, ema_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        flattened = input_tensor.detach().flatten()
        quantiles = torch.quantile(flattened, self.quantile_range)
        ema_values[:] = self.alpha * quantiles + (1 - self.alpha) * ema_values
        scale = torch.clamp(ema_values[1] - ema_values[0], min=1.0)
        offset = ema_values[0]
        return offset.detach(), scale.detach()
