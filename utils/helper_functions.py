import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import logging

# Setup logger for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------- Setup Functions ----------

def set_random_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value to use
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    logger.info(f"Random seed set to {seed}")

def enable_deterministic_mode() -> None:
    """
    Enable deterministic operations in PyTorch for full reproducibility.
    Note: This may impact performance due to disabling certain optimizations.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info("Deterministic mode enabled")

# ---------- Data Augmentation ----------

def augment_image(image: torch.Tensor, crop_size: int = 64) -> torch.Tensor:
    """
    Apply data augmentation to images following DreamerV3's approach.
    Includes random crop and Gaussian noise (simplified from original for efficiency).
    
    Args:
        image: Image tensor in [batch, channels, height, width] or [batch, time, channels, height, width] format
        crop_size: Target size for random crop
        
    Returns:
        Augmented image tensor with the same batch dimensions
    """
    if image.dim() not in [4, 5]:
        logger.warning(f"Invalid image shape {image.shape}, expected [B, C, H, W] or [B, T, C, H, W]")
        return image
    
    logger.debug(f"Input image: shape={image.shape}, dtype={image.dtype}, "
                 f"min={torch.min(image) if image.numel() > 0 else 'N/A'}, "
                 f"max={torch.max(image) if image.numel() > 0 else 'N/A'}")
    
    original_shape = image.shape
    device = image.device
    
    if image.dim() == 5:  # [batch, time, channels, height, width]
        batch_size, time_steps, channels, height, width = image.shape
        image = image.reshape(batch_size * time_steps, channels, height, width)
        is_sequence = True
        logger.debug(f"Reshaped sequence image to: {image.shape}")
    else:  # [batch, channels, height, width]
        batch_size, channels, height, width = image.shape
        is_sequence = False
        
    if height < crop_size or width < crop_size:
        logger.debug("Image too small for augmentation; returning original.")
        return image.reshape(original_shape) if is_sequence else image
    
    # Random crop with padding if needed
    pad_size = 8
    if height < crop_size + pad_size or width < crop_size + pad_size:
        pad_height = max(0, crop_size + pad_size - height)
        pad_width = max(0, crop_size + pad_size - width)
        image = F.pad(image, [pad_width//2, pad_width - pad_width//2, pad_height//2, pad_height - pad_height//2], mode='reflect')
        height, width = image.shape[-2:]
        logger.debug(f"Applied padding; new image size: height={height}, width={width}")
    
    top_positions = torch.randint(0, height - crop_size + 1, (image.shape[0],), device=device)
    left_positions = torch.randint(0, width - crop_size + 1, (image.shape[0],), device=device)
    cropped_image = torch.zeros(image.shape[0], channels, crop_size, crop_size, device=device)
    for i in range(image.shape[0]):
        cropped_image[i] = image[i, :, top_positions[i]:top_positions[i]+crop_size, left_positions[i]:left_positions[i]+crop_size]
    logger.debug(f"Applied random crop; cropped image shape: {cropped_image.shape}")
    
    # Add Gaussian noise
    noise_scale = 0.01
    noise = torch.randn_like(cropped_image) * noise_scale
    cropped_image = cropped_image + noise
    cropped_image = torch.clamp(cropped_image, 0.0, 1.0)
    
    if is_sequence:
        cropped_image = cropped_image.reshape(batch_size, time_steps, channels, crop_size, crop_size)
        logger.debug(f"Reshaped back to sequence: {cropped_image.shape}")
    
    logger.debug(f"Augmentation complete; output image shape: {cropped_image.shape}")
    return cropped_image

# ---------- Utility Functions ----------

def compute_tensor_statistics(tensor: torch.Tensor, name: str) -> Dict[str, float]:
    """
    Calculate mean and standard deviation of a tensor for monitoring.
    
    Args:
        tensor: Input tensor to analyze
        name: Prefix for the statistic names in the output dictionary
        
    Returns:
        Dictionary containing mean and standard deviation
    """
    cleaned_tensor = torch.nan_to_num(tensor, nan=0.0)
    return {f"{name}_mean": cleaned_tensor.mean().item(), f"{name}_standard_deviation": cleaned_tensor.std().item()}

def count_episode_steps(directory: str) -> int:
    """
    Count total steps across all episodes in a directory.
    
    Args:
        directory: Path to the directory containing episode files
        
    Returns:
        Total number of steps across all episodes
    """
    directory_path = Path(directory)
    total_steps = 0
    
    for file in directory_path.glob("*.npz"):
        try:
            data = np.load(file)
            total_steps += len(data["reward"]) - 1
        except Exception as error:
            logger.error(f"Error counting steps in {file}: {error}")
            continue
    return total_steps

def convert_tensor_to_numpy(tensor: Any) -> np.ndarray:
    """
    Convert a tensor or array-like object to a NumPy array safely.
    
    Args:
        tensor: PyTorch tensor or other array-like object
        
    Returns:
        NumPy array representation of the input
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

# ---------- Model Utilities ----------


def imagine_trajectory(
    world_model,
    batch_size: int,
    horizon: int,
    number_of_possible_actions: int,
    initial_state: Dict[str, torch.Tensor],
    actor_network: Optional[nn.Module] = None,
    use_gradients: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    device = next(world_model.parameters()).device
    has_discrete_actions = hasattr(world_model.action_space, 'n')
    
    logger.debug(f"imagine_trajectory: batch_size={batch_size}, horizon={horizon}, num_actions={number_of_possible_actions}")
    logger.debug(f"imagine_trajectory: Initial state hidden_state shape: {initial_state['hidden_state'].shape}")
    logger.debug(f"imagine_trajectory: Initial state stoch shape: {initial_state['stoch'].shape}")
    logger.debug(f"imagine_trajectory: Initial state deter shape: {initial_state['deter'].shape}")
    
    current_state = initial_state
    
    imagined_features_list = []
    imagined_deter_list = []
    imagined_stoch_list = []
    imagined_actions_list = []
    imagined_action_indices_list = []
    
    for t in range(horizon):
        deter = current_state["hidden_state"][-1]  # [B, hidden_units]
        features = torch.cat([deter, current_state["stoch"]], dim=-1)  # [B, feature_dim]
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        imagined_features_list.append(features)
        logger.debug(f"imagine_trajectory: Step {t}, features shape: {features.shape}")

        if actor_network is not None:
            action_distribution = actor_network(features)
            logger.debug(f"imagine_trajectory: Step {t}, action_distribution logits shape: {action_distribution.logits.shape if hasattr(action_distribution, 'logits') else 'N/A'}")
            if use_gradients and not has_discrete_actions:
                action = action_distribution.reparameterized_sample()  # [B, action_dim]
                action_indices = action
            else:
                with torch.no_grad():
                    action_indices = action_distribution.sample()  # Should be [B]
                    logger.debug(f"imagine_trajectory: Step {t}, raw action_indices shape: {action_indices.shape}")
                    # Ensure action_indices is [B]
                    if action_indices.dim() > 1:
                        action_indices = action_indices.argmax(dim=-1)  # [B]
                    logger.debug(f"imagine_trajectory: Step {t}, action_indices shape after argmax: {action_indices.shape}")
                    if has_discrete_actions:
                        action = F.one_hot(action_indices.long(), num_classes=number_of_possible_actions).float()  # [B, A]
                        logger.debug(f"imagine_trajectory: Step {t}, action shape after one_hot: {action.shape}")
                        # Ensure action is strictly 2D [B, A]
                        if action.dim() > 2:
                            action = action.squeeze()
                            logger.debug(f"imagine_trajectory: Step {t}, action shape after squeeze: {action.shape}")
                        if action.dim() != 2 or action.shape != (batch_size, number_of_possible_actions):
                            action = action.view(batch_size, number_of_possible_actions)
                            logger.debug(f"imagine_trajectory: Step {t}, action shape after view: {action.shape}")
                    else:
                        action = action_indices  # [B, action_dim]
            logger.debug(f"imagine_trajectory: Step {t}, action shape before imagine_step: {action.shape}")
            action_indices = action_indices.view(batch_size)  # Force [B]
            logger.debug(f"imagine_trajectory: Step {t}, action_indices shape final: {action_indices.shape}")
        else:
            with torch.no_grad():
                if has_discrete_actions:
                    action_indices = torch.randint(0, number_of_possible_actions, (batch_size,), device=device)
                    action = F.one_hot(action_indices, num_classes=number_of_possible_actions).float()
                else:
                    action = torch.randn((batch_size, number_of_possible_actions), device=device)
                    action_indices = action

        # Final shape check
        if action.dim() != 2:
            logger.error(f"imagine_trajectory: Step {t}, action has unexpected dimensions: {action.shape}")
            action = action.view(batch_size, number_of_possible_actions)
        logger.debug(f"imagine_trajectory: Step {t}, final action shape before imagine_step: {action.shape}")
        
        imagined_actions_list.append(action)
        imagined_action_indices_list.append(action_indices)

        next_state, _ = world_model.dynamics.imagine_step(action, current_state)
        imagined_deter_list.append(next_state["deter"])
        imagined_stoch_list.append(next_state["stoch"])
        current_state = {
            "hidden_state": next_state["hidden_state"],
            "stoch": next_state["stoch"],
            "deter": next_state["deter"]
        }
        
        logger.debug(f"imagine_trajectory: Step {t}, next_deter shape: {next_state['deter'].shape}")
        logger.debug(f"imagine_trajectory: Step {t}, next_stoch shape: {next_state['stoch'].shape}")

    imagined_features = torch.stack(imagined_features_list, dim=1)
    imagined_state = {
        "deter": torch.stack(imagined_deter_list, dim=1),
        "stoch": torch.stack(imagined_stoch_list, dim=1)
    }
    imagined_actions = torch.stack(imagined_actions_list, dim=1)
    imagined_action_indices = torch.stack(imagined_action_indices_list, dim=1)
    
    logger.debug(f"imagine_trajectory: Final features shape: {imagined_features.shape}")
    logger.debug(f"imagine_trajectory: Final actions shape: {imagined_actions.shape}")
    logger.debug(f"imagine_trajectory: Final action indices shape: {imagined_action_indices.shape}")
    logger.debug(f"imagine_trajectory: Final state deter shape: {imagined_state['deter'].shape}")
    logger.debug(f"imagine_trajectory: Final state stoch shape: {imagined_state['stoch'].shape}")
    
    return imagined_features, imagined_state, imagined_actions, imagined_action_indices


def _create_default_state(key: str, batch_size: int, world_model: Any, device: str) -> torch.Tensor:
    """Create a default state tensor for unexpected input formats."""
    if key in ["deterministic", "deter"]:
        return torch.zeros((batch_size, world_model.dynamics.deterministic_dimension), device=device)
    elif key == "logits":
        if world_model.dynamics.use_discrete:
            return torch.zeros((batch_size, world_model.dynamics.discrete_latent_num, 
                                world_model.dynamics.discrete_latent_size), device=device)
        return torch.zeros((batch_size, world_model.dynamics.stoch_dim), device=device)
    elif key == "stoch":
        return torch.zeros((batch_size, world_model.dynamics.stoch_dim), device=device)
    return torch.zeros((batch_size, 1), device=device)

def compute_lambda_return_target(
    reward: torch.Tensor,
    value: torch.Tensor,
    discount: torch.Tensor,
    lambda_discount_factor: float
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Compute lambda return targets for actor-critic learning with numerical stability.
    Matches DreamerV3's implementation for robustness.
    
    Args:
        reward: Tensor of rewards in [time, batch, 1] format
        value: Tensor of value estimates in [time, batch, 1] format
        discount: Tensor of discount factors in [time, batch, 1] format
        lambda_discount_factor: Mixing parameter between TD(0) and TD(∞)
        
    Returns:
        targets: List of [batch, 1] tensors for each timestep
        weights: List of [batch, 1] tensors of weights for each timestep
        value_predictions: [batch, time, 1] tensor of value predictions
    """
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
    
    targets = []
    weights = []
    last_return = value[-1]
    
    for t in reversed(range(time_steps)):
        current_return = reward[t] + discount[t] * (
            (1 - lambda_discount_factor) * value[t] + lambda_discount_factor * last_return
        )
        current_return = torch.clamp(current_return, min=-20.0, max=20.0)
        targets.insert(0, current_return)
        weights.insert(0, torch.ones_like(reward[t]))
        last_return = current_return

    return targets, weights, value.transpose(0, 1)  # [batch, time, 1]