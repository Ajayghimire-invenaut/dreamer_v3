"""
Neural network modules for Dreamer-V3.
Includes MultiEncoder, RSSM, MultiDecoder, MLP, and DistributionWrapper.
Also includes symlog transformation utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from typing import Any, Dict, List, Optional, Tuple
import math
from utils.helper_functions import OneHotDistribution
# ---------- Symlog Transformation Utilities ----------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))

def inv_symlog(y: torch.Tensor) -> torch.Tensor:
    return torch.sign(y) * (torch.expm1(torch.abs(y)))

def discretize_symlog(x: torch.Tensor, num_bins: int = 255, low: float = -10.0, high: float = 10.0) -> torch.Tensor:
    """
    Applies symlog to x, then discretizes to integer bins.
    Assumes x is in a scale where most values lie between low and high.
    """
    y = symlog(x)
    y_clamped = torch.clamp(y, low, high)
    bins = ((y_clamped - low) / (high - low) * (num_bins - 1)).long()
    return bins

def undisc_symlog(bins: torch.Tensor, num_bins: int = 255, low: float = -10.0, high: float = 10.0) -> torch.Tensor:
    """
    Converts discretized symlog bin indices back to a real value.
    """
    y = bins.float() / (num_bins - 1) * (high - low) + low
    return inv_symlog(y)

# ---------- Orthogonal Initialization ----------

def orthogonal_initialize(module: nn.Module, gain: Optional[float] = None) -> None:
    if gain is None:
        gain = nn.init.calculate_gain('relu')
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

# ---------- MultiEncoder ----------

class MultiEncoder(nn.Module):
    def __init__(self,
                 input_shapes: Dict[str, Tuple[int, ...]],
                 output_dimension: int,
                 use_orthogonal: bool = False) -> None:
        super(MultiEncoder, self).__init__()
        self.debug = False  # Set via configuration or externally
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
            # Output: (256, 2, 2) → flattened = 256*2*2 = 1024
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
        debug = getattr(self, "debug", False) or getattr(observations, "debug", False)
        if self.is_image:
            image = observations["image"]
            if debug:
                print("[DEBUG MultiEncoder] received image shape:", image.shape, flush=True)
            if image.dim() == 5:
                if image.shape[-1] in [1, 3, 4]:  # [B, T, H, W, C]
                    image = image.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
                B, T, C, H, W = image.shape
                image = image.reshape(B * T, C, H, W)
                features = self.convolutional_layers(image)
                flattened = features.reshape(B * T, -1)
                out = self.fc(flattened)
                out = out.reshape(B, T, -1)  # [B, T, embed_dim]
                if debug:
                    print("[DEBUG MultiEncoder] output shape (sequence):", out.shape, flush=True)
                return out
            elif image.dim() == 4:
                if image.shape[-1] in [1, 3, 4]:  # [B, H, W, C]
                    image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
                features = self.convolutional_layers(image)
                flattened = features.reshape(features.size(0), -1)
                out = self.fc(flattened)  # [B, embed_dim]
                if debug:
                    print("[DEBUG MultiEncoder] output shape (batch):", out.shape, flush=True)
                return out
            elif image.dim() == 3:
                if image.shape[-1] in [1, 3, 4]:  # [H, W, C]
                    image = image.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                features = self.convolutional_layers(image)
                flattened = features.reshape(features.size(0), -1)
                out = self.fc(flattened)  # [1, embed_dim]
                if debug:
                    print("[DEBUG MultiEncoder] output shape (single):", out.shape, flush=True)
                return out
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            out = self.fc(observations["image"])  # [B, embed_dim] or [B, T, embed_dim]
            if debug:
                print("[DEBUG MultiEncoder] non-image output shape:", out.shape, flush=True)
            return out

# ---------- RSSM (Recurrent State-Space Model) ----------

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
        self.debug = False  # Set externally if needed
        self.gru = nn.GRU(input_size=embedding_dimension + number_of_actions,
                         hidden_size=deter_dimension, batch_first=False)
        if use_discrete:
            self.discrete_latent_num = discrete_latent_num
            self.discrete_latent_size = discrete_latent_size
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
        # Input: [T, B, embed_dim], [T, B, action_dim], [T, B]
        if self.debug:
            print(f"[DEBUG RSSM.observe] embeddings shape: {embeddings.shape}", flush=True)
            print(f"[DEBUG RSSM.observe] actions shape: {actions.shape}", flush=True)
        combined = torch.cat([embeddings, actions], dim=-1)  # [T, B, embed_dim+action_dim]
        outputs, _ = self.gru(combined)  # [T, B, deter_dimension]
        deter = outputs.transpose(0, 1)  # [B, T, deter_dimension]
        if self.use_discrete:
            logits = self.logits_layer(outputs)  # [T, B, discrete_latent_num*discrete_latent_size]
            T, B, _ = logits.shape
            logits = logits.reshape(T, B, self.discrete_latent_num, self.discrete_latent_size)
            logits = logits.transpose(0, 1)  # [B, T, dnum, dsize]
            posterior = {"deter": deter, "logits": logits}
            prior = posterior  # Placeholder; real prior would differ
        else:
            mean = self.mean_layer(outputs)
            std = F.softplus(self.std_layer(outputs)) + self.minimum_std
            mean = mean.transpose(0, 1)  # [B, T, stoch_dimension]
            std = std.transpose(0, 1)    # [B, T, stoch_dimension]
            posterior = {"deter": deter, "mean": mean, "std": std}
            prior = posterior
        if self.debug:
            print(f"[DEBUG RSSM.observe] posterior['deter'] shape: {posterior['deter'].shape}", flush=True)
            if "logits" in posterior:
                print(f"[DEBUG RSSM.observe] posterior['logits'] shape: {posterior['logits'].shape}", flush=True)
            else:
                print(f"[DEBUG RSSM.observe] posterior['mean'] shape: {posterior['mean'].shape}", flush=True)
        return posterior, prior

    def observe_step(self,
                     previous_state: Optional[Dict[str, torch.Tensor]],
                     previous_action: Optional[torch.Tensor],
                     embedding: torch.Tensor,
                     is_first: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], None]:
        # Input: [B, embed_dim] or [B, 1, embed_dim]
        if embedding.dim() == 3:
            embedding = embedding.squeeze(1) if embedding.size(1) == 1 else embedding[:, -1, :]
            if self.debug:
                print("[DEBUG RSSM.observe_step] Processed embedding shape:", embedding.shape, flush=True)
        B = embedding.size(0)
        if previous_action is None:
            previous_action = torch.zeros(B, self.number_of_actions, device=embedding.device)
        elif previous_action.dim() == 3:
            previous_action = previous_action.squeeze(1) if previous_action.size(1) == 1 else previous_action[:, -1, :]
        elif previous_action.dim() == 1:
            previous_action = F.one_hot(previous_action.long(), num_classes=self.number_of_actions).float()
        combined = torch.cat([embedding, previous_action], dim=-1)  # [B, embed_dim+action_dim]
        combined = combined.unsqueeze(0)  # [1, B, embed_dim+action_dim]
        output, _ = self.gru(combined)  # [1, B, deter_dimension]
        deter = output.squeeze(0)       # [B, deter_dimension]
        if self.use_discrete:
            logits = self.logits_layer(deter)
            logits = logits.reshape(B, self.discrete_latent_num, self.discrete_latent_size)  # [B, dnum, dsize]
            new_state = {"deter": deter, "logits": logits}
        else:
            mean = self.mean_layer(deter)
            std = F.softplus(self.std_layer(deter)) + self.minimum_std
            new_state = {"deter": deter, "mean": mean, "std": std}
        if self.debug:
            print(f"[DEBUG RSSM.observe_step] new_state['deter'] shape: {new_state['deter'].shape}", flush=True)
            if "logits" in new_state:
                print(f"[DEBUG RSSM.observe_step] new_state['logits'] shape: {new_state['logits'].shape}", flush=True)
        return new_state, None

    def get_features(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Input: Dict with [B, T, ...] or [B, ...]
        if self.use_discrete:
            probs = torch.softmax(state["logits"], dim=-1)  # [B, T, dnum, dsize] or [B, dnum, dsize]
            if probs.ndim == 4:  # [B, T, dnum, dsize]
                probs_flat = probs.reshape(probs.shape[0], probs.shape[1], -1)  # [B, T, dnum*dsize]
            elif probs.ndim == 3:  # [B, dnum, dsize]
                probs_flat = probs.reshape(probs.shape[0], 1, -1)  # [B, 1, dnum*dsize]
            else:
                raise ValueError(f"Unexpected probs shape: {probs.shape}")
            deter = state["deter"]  # [B, T, deter_dim] or [B, deter_dim]
            if deter.dim() == 2:
                deter = deter.unsqueeze(1)  # [B, 1, deter_dim]
            features = torch.cat([deter, probs_flat], dim=-1)  # [B, T, feature_dim]
        else:
            deter = state["deter"]  # [B, T, deter_dim] or [B, deter_dim]
            if deter.dim() == 2:
                deter = deter.unsqueeze(1)  # [B, 1, deter_dim]
            if "mean" in state:
                mean = state["mean"]  # [B, T, stoch_dim] or [B, stoch_dim]
                if mean.dim() == 2:
                    mean = mean.unsqueeze(1)  # [B, 1, stoch_dim]
                features = torch.cat([deter, mean], dim=-1)
            else:
                features = deter
        if self.debug:
            print(f"[DEBUG RSSM.get_features] features shape: {features.shape}", flush=True)
        return features

    def imagine_with_action(self,
                            actions: torch.Tensor,
                            initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Input: actions [B, H, action_dim]
        horizon = actions.shape[1]
        state = {}
        current_state = {k: v.clone() for k, v in initial_state.items()}
        for key, value in current_state.items():
            if value.dim() == 1:  # [B]
                state[key] = value.unsqueeze(1).expand(-1, horizon, -1)  # [B, H, dim]
            elif value.dim() == 2:  # [B, dim]
                state[key] = value.unsqueeze(1).expand(-1, horizon, -1)  # [B, H, dim]
            else:
                state[key] = value
        
        # Placeholder rollout (simplified)
        for t in range(horizon):
            combined = torch.cat([state["deter"][:, t, :], actions[:, t, :]], dim=-1)  # [B, deter_dim+action_dim]
            combined = combined.unsqueeze(0)  # [1, B, deter_dim+action_dim]
            output, _ = self.gru(combined)  # [1, B, deter_dimension]
            deter = output.squeeze(0)  # [B, deter_dimension]
            if self.use_discrete:
                logits = self.logits_layer(deter)
                logits = logits.reshape(deter.shape[0], self.discrete_latent_num, self.discrete_latent_size)
                state["deter"][:, t] = deter
                state["logits"][:, t] = logits
            else:
                mean = self.mean_layer(deter)
                std = F.softplus(self.std_layer(deter)) + self.minimum_std
                state["deter"][:, t] = deter
                state["mean"][:, t] = mean
                state["std"][:, t] = std
        
        if self.debug:
            print(f"[DEBUG RSSM.imagine_with_action] state['deter'] shape: {state['deter'].shape}", flush=True)
        return state

    def compute_kl_loss(self,
                        posterior: Dict[str, torch.Tensor],
                        prior: Dict[str, torch.Tensor],
                        kl_free: float,
                        dynamics_scale: float,
                        representation_scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_discrete:
            post_logits = posterior["logits"]  # [B, T, dnum, dsize]
            prior_logits = prior["logits"]
            post = torch.softmax(post_logits, dim=-1)
            prior = torch.softmax(prior_logits, dim=-1)
            kl = post * (torch.log(post + 1e-8) - torch.log(prior + 1e-8))
            kl = kl.sum(dim=-1).mean()
        else:
            mean_diff = posterior["mean"] - prior["mean"]  # [B, T, stoch_dim]
            var_ratio = (posterior["std"] / prior["std"])**2
            kl = 0.5 * (var_ratio + (mean_diff**2) / (prior["std"]**2) - 1 - torch.log(var_ratio + 1e-8))
            kl = kl.mean()
        kl_loss = torch.clamp(kl, min=kl_free)
        return kl_loss, kl, dynamics_scale, representation_scale

# ---------- MultiDecoder ----------

class MultiDecoder(nn.Module):
    def __init__(self,
                 feature_dimension: int,
                 output_shapes: Dict[str, Tuple[int, ...]],
                 dummy_parameter: Any,
                 use_orthogonal: bool = False) -> None:
        super(MultiDecoder, self).__init__()
        self.debug = False
        raw_output_shape = output_shapes["image"]
        if len(raw_output_shape) == 3 and raw_output_shape[-1] <= 4:
            self.output_shape = (raw_output_shape[-1], raw_output_shape[0], raw_output_shape[1])  # [C, H, W]
        else:
            self.output_shape = raw_output_shape
        if self.debug:
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
            if features.dim() == 3:  # [B, T, feature_dim]
                B, T, D = features.shape
                x = self.fc(features.reshape(B * T, D))
                x = x.reshape(B * T, 256, 4, 4)
                reconstruction = self.deconv_layers(x)  # [B*T, C, H, W]
                reconstruction = reconstruction.reshape(B, T, *self.output_shape)  # [B, T, C, H, W]
            else:  # [B, feature_dim]
                x = self.fc(features)
                x = x.reshape(features.size(0), 256, 4, 4)
                reconstruction = self.deconv_layers(x)  # [B, C, H, W]
            if self.debug:
                print("[DEBUG MultiDecoder] reconstruction shape:", reconstruction.shape, flush=True)
            return {"image": DistributionWrapper(reconstruction, dist_type="gaussian")}
        else:
            if features.dim() == 3:  # [B, T, feature_dim]
                B, T, D = features.shape
                out = self.decoder(features.reshape(B * T, D))
                out = out.reshape(B, T, *self.output_shape)
            else:  # [B, feature_dim]
                out = self.decoder(features)
            if self.debug:
                print("[DEBUG MultiDecoder] reconstruction shape:", out.shape, flush=True)
            return {"image": DistributionWrapper(out, dist_type="gaussian")}

# ---------- MLP ----------

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
        self.debug = False
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
        orig_shape = x.shape
        if self.debug:
            print(f"[DEBUG MLP] input shape: {x.shape}", flush=True)
        if x.dim() > 2:  # [B, T, feature_dim]
            x = x.reshape(-1, x.shape[-1])
            output = self.network(x)
            new_shape = list(orig_shape[:-1]) + [output.shape[-1]]
            output = output.reshape(new_shape)  # [B, T, out_dim]
        else:  # [B, feature_dim]
            output = self.network(x)  # [B, out_dim]
        if self.debug:
            print(f"[DEBUG MLP] output shape: {output.shape}", flush=True)
        return DistributionWrapper(output, dist_type=self.distribution_type)

# ---------- DistributionWrapper ----------

class DistributionWrapper:
    def __init__(self, logits: torch.Tensor, dist_type: str = "gaussian") -> None:
        self.logits = logits
        self.dist_type = dist_type
        self.debug = False
        self.distribution = self._create_distribution()
        if self.debug:
            print(f"[DEBUG DistributionWrapper] initialized with logits shape: {self.logits.shape} and type: {self.dist_type}", flush=True)

    def _create_distribution(self) -> Any:
        if self.dist_type == "gaussian":
            return td.Normal(self.logits, torch.ones_like(self.logits) * 0.1)
        elif self.dist_type == "symlog_disc":
            return td.Categorical(logits=self.logits)
        elif self.dist_type == "binary":
            return td.Bernoulli(logits=self.logits)
        elif self.dist_type == "onehot":
            return OneHotDistribution(self.logits)
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist_type}")

    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        if self.dist_type == "binary":
            if target.dim() == 2:  # [B, T]
                target = target.unsqueeze(-1)  # [B, T, 1]
            elif target.dim() == 3 and target.shape[-1] != 1:
                raise ValueError("Target for binary distribution must have shape [B, T] or [B, T, 1]")
            log_prob = self.distribution.log_prob(target)
        elif self.dist_type == "symlog_disc":
            if target.dim() == self.logits.dim() - 1:
                target = target.long()
            else:
                target = target.long().squeeze(-1)
            log_prob = self.distribution.log_prob(target)
        else:  # gaussian
            if target.dim() == self.logits.dim() - 1:
                target = target.unsqueeze(-1)
            log_prob = self.distribution.log_prob(target)
        if self.debug:
            print(f"[DEBUG DistributionWrapper.log_prob] target shape: {target.shape}, log_prob shape: {log_prob.shape}", flush=True)
        return log_prob

    def mode(self) -> torch.Tensor:
        if self.dist_type == "gaussian":
            return self.logits
        elif self.dist_type == "symlog_disc":
            mode = torch.argmax(self.logits, dim=-1)
            return undisc_symlog(mode, num_bins=self.logits.shape[-1])
        elif self.dist_type == "binary":
            return (self.logits >= 0).float()
        else:
            raise ValueError(f"Unsupported distribution type: {self.dist_type}")

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy()

# ---------- RewardEMA ----------

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