import time
import numpy as np
import pathlib
import torch
import itertools
from typing import Any, Dict, Generator, Tuple, List, Optional
from torch.utils.data import IterableDataset, DataLoader
import logging

# Setup logger for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

def robust_collate_function(batch: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Robustly collate a list of sample dictionaries into a batch with consistent dimensions.
    Handles missing keys and arrays of different shapes by padding or truncating.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary with consistent dimensions
    """
    if not batch:
        return {}
        
    all_keys = set().union(*[sample.keys() for sample in batch])
    collated_batch = {}
    
    for key in all_keys:
        values = [sample.get(key, None) for sample in batch]
        reference_sample = next((v for v in values if v is not None), None)
        if reference_sample is None:
            continue
            
        reference_shape = reference_sample.shape[1:] if isinstance(reference_sample, np.ndarray) and reference_sample.ndim > 1 else ()
        reference_dtype = reference_sample.dtype if isinstance(reference_sample, np.ndarray) else np.float32
        
        shapes = [v.shape if isinstance(v, np.ndarray) else () for v in values if v is not None]
        max_seq_length = max(s[0] for s in shapes if s) if shapes else 0
        
        padded_values = []
        for value in values:
            if value is None:
                padded_values.append(np.zeros((max_seq_length,) + reference_shape, dtype=reference_dtype))
            else:
                value = np.array(value) if not isinstance(value, np.ndarray) else value
                if value.shape[0] < max_seq_length:
                    pad_shape = [(0, max_seq_length - value.shape[0])] + [(0, 0)] * (value.ndim - 1)
                    padded_values.append(np.pad(value, pad_shape, mode='constant'))
                elif value.shape[0] > max_seq_length:
                    padded_values.append(value[:max_seq_length])
                else:
                    padded_values.append(value)
        
        try:
            collated_batch[key] = np.stack(padded_values, axis=0)  # [B, T, ...]
        except ValueError as error:
            logger.warning(f"Error collating key '{key}': {error}")
            collated_batch[key] = np.array(padded_values)  # Fallback to array of objects if stacking fails
    
    return collated_batch

class EpisodeDataset(IterableDataset):
    """
    Iterable dataset that samples fixed-length segments from episodes for DreamerV3 training.
    Provides efficient sampling with configurable augmentation, boundary emphasis, and optional prioritization.
    """
    def __init__(
        self,
        episodes: Dict[str, Any],
        sequence_length: int,
        seed: int = 0,
        maximum_samples: int = 10000,
        enable_augmentation: bool = True,
        use_priority: bool = False
    ):
        super().__init__()
        self.episodes = list(episodes.values()) if episodes else []
        self.sequence_length = sequence_length
        self.random_generator = np.random.RandomState(seed)
        self.maximum_samples = maximum_samples
        self.required_keys = ["image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]
        self.enable_augmentation = enable_augmentation
        self.use_priority = use_priority
        
        self.total_steps = sum(len(ep["reward"]) for ep in self.episodes) if self.episodes else 0
        
        if not self.episodes:
            logger.warning("No episodes provided; creating dummy episode")
            self._create_dummy_episode()
        else:
            logger.info(f"Loaded {len(self.episodes)} episodes with {self.total_steps} total steps")
        
        self.random_generator.shuffle(self.episodes)

    def _create_dummy_episode(self) -> None:
        """Create a dummy episode for fallback when no real data is available."""
        dummy_episode = {
            "image": np.zeros((self.sequence_length, 64, 64, 3), dtype=np.uint8),
            "action": np.zeros((self.sequence_length, 2), dtype=np.float32),
            "reward": np.zeros((self.sequence_length,), dtype=np.float32),
            "is_first": np.zeros(self.sequence_length, dtype=bool),
            "is_terminal": np.zeros(self.sequence_length, dtype=bool),
            "is_last": np.zeros(self.sequence_length, dtype=bool),
            "discount": np.ones(self.sequence_length, dtype=np.float32),
            "continuation": np.ones(self.sequence_length, dtype=np.float32)
        }
        dummy_episode["is_first"][0] = True
        dummy_episode["is_last"][-1] = True
        dummy_episode["is_terminal"][-1] = True
        dummy_episode["discount"][-1] = 0.0
        dummy_episode["continuation"][-1] = 0.0
        dummy_episode["action"][:, 0] = 1.0
        
        self.episodes.append(dummy_episode)
        self.total_steps = self.sequence_length

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        """
        Iterate through episodes, yielding fixed-length segments with boundary emphasis and optional prioritization.
        Implements DreamerV3's sampling strategy for better learning of transitions.
        """
        sample_count = 0
        episode_list = self.episodes.copy()
        
        while sample_count < self.maximum_samples:
            if self.use_priority:
                weights = np.ones(len(episode_list)) / len(episode_list)  # Uniform weights for now
                episode_idx = self.random_generator.choice(len(episode_list), p=weights)
                episode = episode_list[episode_idx]
            else:
                self.random_generator.shuffle(episode_list)
                episode = episode_list[sample_count % len(episode_list)]
            
            total_length = len(episode["reward"])
            
            if total_length < self.sequence_length:
                segment = self._pad_short_episode(episode)
            else:
                if self.random_generator.random() < 0.1:
                    start_index = 0
                elif self.random_generator.random() < 0.1 and total_length >= self.sequence_length:
                    start_index = total_length - self.sequence_length
                else:
                    start_index = self.random_generator.randint(0, total_length - self.sequence_length + 1)
                
                segment = {key: value[start_index:start_index + self.sequence_length] for key, value in episode.items()}
            
            segment = self._normalize_action_shapes(segment)
            self._ensure_required_keys(segment)
            
            if self.enable_augmentation:
                segment = self._augment_segment(segment)
            
            yield segment
            sample_count += 1

    def _pad_short_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pad a short episode to reach the required sequence length with appropriate values.
        """
        segment = {}
        for key, value in episode.items():
            array = np.array(value)
            pad_length = self.sequence_length - array.shape[0]
            
            if pad_length <= 0:
                segment[key] = array[:self.sequence_length]
                continue
                
            if key == "image":
                pad_array = np.tile(array[-1:], (pad_length, 1, 1, 1))
            elif key == "action":
                pad_array = np.tile(array[-1:], (pad_length, 1)) if array.ndim > 1 else np.full((pad_length,), array[-1], dtype=array.dtype)
            elif key == "reward":
                pad_array = np.zeros(pad_length, dtype=array.dtype)
            elif key in ["is_terminal", "is_last"]:
                pad_array = np.zeros(pad_length, dtype=bool)
                pad_array[-1] = True
            elif key == "is_first":
                pad_array = np.zeros(pad_length, dtype=bool)
            elif key in ["discount", "continuation"]:
                pad_array = np.ones(pad_length, dtype=np.float32)
                pad_array[-1] = 0.0
            else:
                pad_array = np.zeros((pad_length,) + array.shape[1:], dtype=array.dtype)
                
            segment[key] = np.concatenate([array, pad_array], axis=0)
            
        return segment

    def _ensure_required_keys(self, segment: Dict[str, np.ndarray]) -> None:
        """Ensure all required keys exist in the segment with correct format."""
        for key in self.required_keys:
            if key not in segment or segment[key] is None:
                if key == "image":
                    segment[key] = np.zeros((self.sequence_length, 64, 64, 3), dtype=np.uint8)
                elif key == "action":
                    segment[key] = np.zeros((self.sequence_length, 2), dtype=np.float32)
                    segment[key][:, 0] = 1.0
                elif key == "reward":
                    segment[key] = np.zeros(self.sequence_length, dtype=np.float32)
                elif key == "is_first":
                    segment[key] = np.zeros(self.sequence_length, dtype=bool)
                    segment[key][0] = True
                elif key in ["is_terminal", "is_last"]:
                    segment[key] = np.zeros(self.sequence_length, dtype=bool)
                    segment[key][-1] = True
                elif key in ["discount", "continuation"]:
                    segment[key] = np.ones(self.sequence_length, dtype=np.float32)
                    segment[key][-1] = 0.0

    def _augment_segment(self, segment: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply data augmentation to a segment, including reward scaling."""
        augmented_segment = segment.copy()
        
        if "reward" in augmented_segment and np.abs(augmented_segment["reward"]).max() > 1000:
            from agent.networks import symmetric_logarithm_transformation
            augmented_segment["reward"] = symmetric_logarithm_transformation(
                torch.tensor(augmented_segment["reward"])).numpy()
            
        return augmented_segment

    def _normalize_action_shapes(self, segment: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize action shapes, keeping indices for discrete actions and vectors for continuous actions."""
        if "action" in segment and segment["action"] is not None:
            action = segment["action"]
            if action.ndim == 1:
                # Discrete actions: keep as indices
                segment["action"] = action.astype(np.int32)
            elif action.ndim == 2:
                # Continuous actions: keep as vectors
                segment["action"] = action.astype(np.float32)
            else:
                logger.warning(f"Unexpected action shape {action.shape}, treating as discrete indices")
                segment["action"] = action.flatten().astype(np.int32)
        return segment

def create_dataset(episodes: Dict[str, Any], configuration: Any) -> Generator[Dict[str, Any], None, None]:
    """
    Create an infinite dataset generator for training with shuffling, augmentation, and optional prioritization.
    
    Args:
        episodes: Dictionary of episodes
        configuration: Configuration object with dataset parameters
        
    Returns:
        Generator yielding batches of data
    """
    sequence_length = getattr(configuration, "sequence_length", 50)
    batch_size = getattr(configuration, "batch_size", 16)
    seed = getattr(configuration, "random_seed", 42)
    maximum_samples_per_epoch = getattr(configuration, "max_samples_per_epoch", 10000)
    enable_augmentation = getattr(configuration, "dataset_augmentation", True)
    use_priority = getattr(configuration, "use_priority_replay", False)
    enable_debugging = getattr(configuration, "debug", False)
    
    if not episodes:
        logger.warning("No episodes provided to create_dataset, creating dummy data")
        dummy_episodes = {"dummy_episode": _create_dummy_episode(sequence_length, batch_size)}
        episodes = dummy_episodes
    
    total_steps = sum(len(ep.get("reward", [])) for ep in episodes.values())
    number_of_episodes = len(episodes)
    logger.info(f"Loaded {number_of_episodes} episodes with {total_steps} total steps")
    
    required_keys = ["image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]
    valid_episodes = {}
    
    for key, episode in episodes.items():
        missing_keys = [req_key for req_key in required_keys if req_key not in episode or episode[req_key] is None]
        if missing_keys:
            logger.warning(f"Episode {key} missing required keys: {missing_keys}, fixing...")
            valid_episodes[key] = _fix_episode_data(episode, sequence_length)
        else:
            valid_episodes[key] = _ensure_episode_format(episode)
    
    dataset = EpisodeDataset(
        valid_episodes,
        sequence_length=sequence_length,
        seed=seed,
        maximum_samples=maximum_samples_per_epoch,
        enable_augmentation=enable_augmentation,
        use_priority=use_priority
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=getattr(configuration, "num_workers", 0),
        collate_fn=robust_collate_function,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if getattr(configuration, "num_workers", 0) > 0 else None,
        persistent_workers=getattr(configuration, "num_workers", 0) > 0,
        shuffle=False  # Shuffling handled in EpisodeDataset
    )
    
    return itertools.cycle(dataloader)

def _create_dummy_episode(sequence_length: int, batch_size: int = 1) -> Dict[str, np.ndarray]:
    """Create dummy episode data for fallback when no real data is available."""
    episode_length = max(sequence_length * 2, 100)
    
    action = np.zeros((episode_length, 2), dtype=np.float32)
    action[:, 0] = 1.0
    
    dummy_episode = {
        "image": np.zeros((episode_length, 64, 64, 3), dtype=np.uint8),
        "action": action,
        "reward": np.zeros((episode_length,), dtype=np.float32),
        "is_first": np.zeros(episode_length, dtype=bool),
        "is_terminal": np.zeros(episode_length, dtype=bool),
        "is_last": np.zeros(episode_length, dtype=bool),
        "discount": np.ones(episode_length, dtype=np.float32),
        "continuation": np.ones(episode_length, dtype=np.float32)
    }
    
    dummy_episode["is_first"][0] = True
    dummy_episode["is_last"][-1] = True
    dummy_episode["is_terminal"][-1] = True
    dummy_episode["discount"][-1] = 0.0
    dummy_episode["continuation"][-1] = 0.0
    
    for i in range(episode_length):
        value = int(255 * (i / episode_length))
        dummy_episode["image"][i, 20:40, 20:40, 0] = value
        dummy_episode["image"][i, 25:35, 25:35, 1] = 255 - value
        if i % 10 == 0:
            dummy_episode["reward"][i] = 1.0
    
    return dummy_episode

def _fix_episode_data(episode: Dict[str, np.ndarray], sequence_length: int) -> Dict[str, np.ndarray]:
    """
    Fix episode data by adding missing required keys or correcting formats.
    
    Args:
        episode: Episode data dictionary
        sequence_length: Target sequence length for consistency
        
    Returns:
        Fixed episode dictionary with all required keys
    """
    episode_length = max([len(episode[key]) for key in episode if episode[key] is not None and len(episode[key]) > 0], default=max(sequence_length * 2, 100))
    
    fixed_episode = {key: value for key, value in episode.items() if value is not None and len(value) > 0}
    
    if "image" not in fixed_episode or fixed_episode["image"] is None:
        fixed_episode["image"] = np.zeros((episode_length, 64, 64, 3), dtype=np.uint8)
    
    if "action" not in fixed_episode or fixed_episode["action"] is None:
        fixed_episode["action"] = np.zeros((episode_length, 2), dtype=np.float32)
        fixed_episode["action"][:, 0] = 1.0
    
    if "reward" not in fixed_episode or fixed_episode["reward"] is None:
        fixed_episode["reward"] = np.zeros(episode_length, dtype=np.float32)
    
    if "is_first" not in fixed_episode or fixed_episode["is_first"] is None:
        fixed_episode["is_first"] = np.zeros(episode_length, dtype=bool)
        fixed_episode["is_first"][0] = True
    
    if "is_terminal" not in fixed_episode or fixed_episode["is_terminal"] is None:
        fixed_episode["is_terminal"] = np.zeros(episode_length, dtype=bool)
        fixed_episode["is_terminal"][-1] = True
    
    if "is_last" not in fixed_episode or fixed_episode["is_last"] is None:
        fixed_episode["is_last"] = np.zeros(episode_length, dtype=bool)
        fixed_episode["is_last"][-1] = True
    
    if "discount" not in fixed_episode or fixed_episode["discount"] is None:
        fixed_episode["discount"] = np.ones(episode_length, dtype=np.float32)
        if "is_terminal" in fixed_episode:
            fixed_episode["discount"][fixed_episode["is_terminal"]] = 0.0
    
    if "continuation" not in fixed_episode or fixed_episode["continuation"] is None:
        fixed_episode["continuation"] = np.ones(episode_length, dtype=np.float32)
        if "is_terminal" in fixed_episode:
            fixed_episode["continuation"][fixed_episode["is_terminal"]] = 0.0
    
    return fixed_episode

def _ensure_episode_format(episode: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Ensure episode data has the correct format, adding continuation and is_last if missing.
    
    Args:
        episode: Episode data dictionary
        
    Returns:
        Episode with correct formats and continuation/is_last fields
    """
    fixed_episode = episode.copy()
    episode_length = max([len(v) for v in episode.values() if v is not None and len(v) > 0], default=1)
    
    if "continuation" not in fixed_episode or fixed_episode["continuation"] is None:
        fixed_episode["continuation"] = np.ones(episode_length, dtype=np.float32)
        if "is_terminal" in fixed_episode and fixed_episode["is_terminal"] is not None:
            fixed_episode["continuation"][fixed_episode["is_terminal"]] = 0.0
    
    if "is_last" not in fixed_episode or fixed_episode["is_last"] is None:
        fixed_episode["is_last"] = np.zeros(episode_length, dtype=bool)
        if "is_terminal" in fixed_episode and fixed_episode["is_terminal"] is not None:
            terminal_indices = np.where(fixed_episode["is_terminal"])[0]
            fixed_episode["is_last"][terminal_indices[-1] if len(terminal_indices) > 0 else -1] = True
        else:
            fixed_episode["is_last"][-1] = True
    
    if "action" in fixed_episode and fixed_episode["action"] is not None:
        action = fixed_episode["action"]
        if action.shape[0] != episode_length:
            fixed_episode["action"] = np.zeros((episode_length, 2), dtype=np.float32)
            fixed_episode["action"][:, 0] = 1.0
        elif action.ndim == 1:
            one_hot_action = np.zeros((episode_length, 2), dtype=np.float32)
            for i, action_index in enumerate(action):
                one_hot_action[i, min(max(int(action_index), 0), 1)] = 1.0
            fixed_episode["action"] = one_hot_action
        elif action.ndim == 2 and action.shape[1] != 2:
            one_hot_action = np.zeros((episode_length, 2), dtype=np.float32)
            for i, action_index in enumerate(action.flatten()):
                one_hot_action[i, min(max(int(action_index), 0), 1)] = 1.0
            fixed_episode["action"] = one_hot_action
    
    return fixed_episode

def load_episode_data(directory: str, limit: Optional[int] = None, reverse: bool = True) -> Dict[str, Any]:
    """
    Load episode data from .npz files in a directory with validation, including is_last.
    
    Args:
        directory: Path to directory containing episode files
        limit: Maximum number of steps to load
        reverse: Whether to load newest episodes first
        
    Returns:
        Dictionary of validated episodes
    """
    directory_path = pathlib.Path(directory).expanduser()
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        return {}
        
    episodes = {}
    total_steps = 0
    files = sorted(directory_path.glob("*.npz"), reverse=reverse)
    required_keys = {"action", "reward", "is_first", "is_terminal", "image", "discount", "is_last"}
    
    for file_path in files:
        try:
            with file_path.open("rb") as file:
                episode = np.load(file)
                episode = {key: episode[key] for key in episode.keys()}
                
            if not required_keys.issubset(episode.keys()) or any(v.size == 0 for v in episode.values()):
                logger.warning(f"Skipping invalid episode file: {file_path}")
                continue
                
            episodes[str(file_path.stem)] = episode
            episode_steps = len(episode["reward"])
            total_steps += episode_steps
            
            if limit and total_steps >= limit:
                logger.info(f"Reached step limit ({limit}), loaded {len(episodes)} episodes")
                break
                
        except Exception as error:
            logger.error(f"Failed to load {file_path}: {error}")
            
    logger.info(f"Loaded {len(episodes)} episodes with total_steps={total_steps}")
    return episodes

def save_episode(directory: str, environment_identifier: str, episode: Dict[str, Any]) -> None:
    directory_path = pathlib.Path(directory).expanduser()
    directory_path.mkdir(parents=True, exist_ok=True)
    
    episode_length = len(episode.get("reward", []))
    if episode_length == 0:
        logger.warning("Attempted to save empty episode, skipping")
        return
    
    # Ensure all arrays match reward length
    for key in episode.keys():
        if len(episode[key]) != episode_length:
            if len(episode[key]) > episode_length:
                episode[key] = episode[key][:episode_length]
            else:
                pad_size = episode_length - len(episode[key])
                if key in ["is_first", "is_terminal", "is_last"]:
                    padding = np.zeros(pad_size, dtype=bool)
                else:
                    padding = np.zeros((pad_size,) + episode[key].shape[1:], dtype=episode[key].dtype)
                episode[key] = np.concatenate([episode[key], padding])
    
    # Ensure all required keys are present
    required_keys = ["image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]
    for key in required_keys:
        if key not in episode:
            if key == "continuation":
                episode[key] = np.ones(episode_length, dtype=np.float32)
                if "is_terminal" in episode:
                    episode[key][episode["is_terminal"]] = 0.0
            elif key in ["is_first", "is_terminal", "is_last"]:
                episode[key] = np.zeros(episode_length, dtype=bool)
                if key == "is_last":
                    episode[key][-1] = True
            else:
                episode[key] = np.zeros(episode_length, dtype=np.float32)
    
    if "action" in episode:
        actions = np.array(episode["action"])
        if actions.ndim > 1 and actions.shape[1] > 1:
            actions = np.argmax(actions, axis=1)
        episode["action"] = actions.astype(np.int32)
    
    timestamp = int(time.time())
    filename = directory_path / f"{environment_identifier}-{episode_length}-{timestamp}.npz"
    
    episode_arrays = {key: np.array(value) for key, value in episode.items()}
    with open(filename, "wb") as file:
        np.savez_compressed(file, **episode_arrays)
        
    logger.debug(f"Saved episode with {episode_length} steps to {filename}")

def simulate_episode(
    agent_or_policy: Any,
    environment: Any,
    num_episodes: int,
    directory: str = "episodes/train"  # Add directory parameter with default
) -> Tuple[int, int, bool, Dict[str, torch.Tensor]]:
    agent_state = None
    total_step_count = 0
    episode_count = 0
    done = False

    for _ in range(num_episodes):
        try:
            observation = environment.reset()
            if observation is None or "image" not in observation:
                logger.error("Environment reset returned invalid observation.")
                return total_step_count, episode_count, True, agent_state
        except Exception as error:
            logger.error(f"Error during environment reset: {error}")
            return total_step_count, episode_count, True, agent_state

        current_episode_data = {key: [] for key in ["image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"]}
        is_first_step = True

        while True:
            try:
                policy_output, agent_state = agent_or_policy(observation, agent_state, training=True)
                action = policy_output["action"]
                if isinstance(action, torch.Tensor) and action.ndim > 1:
                    action = action[0]
            except Exception as error:
                logger.error(f"Error getting action from agent: {error}")
                break

            processed_action, action_to_save = _process_action(action, environment)
            if processed_action is None or action_to_save is None:
                logger.error("Action processing failed, stopping episode.")
                break

            current_episode_data["image"].append(observation["image"][0] if observation["image"].ndim > 3 else observation["image"])
            current_episode_data["action"].append(action_to_save)
            current_episode_data["is_first"].append(is_first_step)
            is_first_step = False

            try:
                next_observation, reward, done, info = environment.step(processed_action)
                is_terminal = info.get("is_terminal", done)
                is_last = info.get("is_last", done)
            except Exception as error:
                logger.error(f"Error during environment step: {error}")
                done = True
                is_terminal = True
                is_last = True
                next_observation, reward = None, 0.0

            current_episode_data["reward"].append(reward)
            current_episode_data["is_terminal"].append(is_terminal)
            current_episode_data["is_last"].append(is_last)
            current_episode_data["discount"].append(1.0 if not is_terminal else 0.0)
            current_episode_data["continuation"].append(1.0 if not is_terminal else 0.0)

            observation = next_observation
            total_step_count += 1

            if done or is_last:
                episode_count += 1
                logger.info(f"Episode {episode_count} finished after {len(current_episode_data['reward'])} steps.")
                episode_data_np = {k: np.array(v) for k, v in current_episode_data.items()}
                save_episode(directory, environment.identifier, episode_data_np)  # Pass directory
                break

    return total_step_count, episode_count, done, agent_state

def _process_action(action: Any, environment: Any) -> Tuple[Any, Any]:
    """
    Process action for environment step and saving, ensuring validity for Discrete action spaces.
    
    Args:
        action: Raw action from agent (e.g., one-hot tensor or scalar)
        environment: Environment with action_space attribute
        
    Returns:
        Tuple of (processed_action for env.step, action_to_save for episode data)
    """
    try:
        action_np = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.array(action)
        if hasattr(environment.action_space, 'n'):
            num_actions = environment.action_space.n
            if action_np.ndim > 1:
                action_index = np.argmax(action_np, axis=-1)
            else:
                action_index = int(action_np.item() if action_np.size == 1 else action_np[0])
            action_index = np.clip(action_index, 0, num_actions - 1)
            return action_index, action_index
        else:
            expected_shape = environment.action_space.shape
            if action_np.ndim > len(expected_shape):
                action_np = action_np.reshape(expected_shape)
            return action_np, action_np
    except Exception as error:
        logger.error(f"Error processing action: {error}")
        return None, None

def save_episode(directory: str, environment_identifier: str, episode: Dict[str, Any]) -> None:
    directory_path = pathlib.Path(directory).expanduser()
    directory_path.mkdir(parents=True, exist_ok=True)
    
    episode_length = len(episode.get("reward", []))
    if episode_length == 0:
        logger.warning("Attempted to save empty episode, skipping")
        return
    
    # Ensure all arrays match reward length
    for key in episode.keys():
        if len(episode[key]) != episode_length:
            if len(episode[key]) > episode_length:
                episode[key] = episode[key][:episode_length]
            else:
                pad_size = episode_length - len(episode[key])
                if key in ["is_first", "is_terminal", "is_last"]:
                    padding = np.zeros(pad_size, dtype=bool)
                else:
                    padding = np.zeros((pad_size,) + episode[key].shape[1:], dtype=episode[key].dtype)
                episode[key] = np.concatenate([episode[key], padding])
    
    if "is_last" not in episode:
        episode["is_last"] = np.zeros(episode_length, dtype=bool)
        if "is_terminal" in episode:
            terminal_indices = np.where(episode["is_terminal"])[0]
            episode["is_last"][terminal_indices[-1] if len(terminal_indices) > 0 else -1] = True
        else:
            episode["is_last"][-1] = True
    
    if "action" in episode:
        actions = np.array(episode["action"])
        if actions.ndim > 1 and actions.shape[1] > 1:
            actions = np.argmax(actions, axis=1)
        episode["action"] = actions.astype(np.int32)
    
    timestamp = int(time.time())
    filename = directory_path / f"{environment_identifier}-{episode_length}-{timestamp}.npz"
    
    episode_arrays = {key: np.array(value) for key, value in episode.items()}
    
    # Add debug log here
    logger.debug(f"Saving episode to {filename}")
    with open(filename, "wb") as file:
        np.savez_compressed(file, **episode_arrays)
    
    logger.debug(f"Saved episode with {episode_length} steps to {filename}")

def preprocess_batch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    processed_batch = {}
    required_keys = {"image", "action", "reward", "is_first", "is_terminal", "is_last", "discount", "continuation"}
    
    for key, value in batch.items():
        tensor = torch.tensor(value, device=device)
        if key == "image":
            tensor = tensor.float() / 255.0
            if tensor.dim() == 4:
                tensor = tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            elif tensor.dim() == 5:
                tensor = tensor.permute(0, 1, 4, 2, 3)  # [B, T, H, W, C] -> [B, T, C, H, W]
        elif key in ["is_first", "is_terminal", "is_last"]:
            tensor = tensor.bool()
        elif key == "action":
            if tensor.dim() == 2:  # [B, T] discrete indices
                tensor = tensor.long()
            elif tensor.dim() == 3:  # [B, T, D] continuous vectors
                tensor = tensor.float()
            else:
                raise ValueError(f"Unexpected action tensor shape: {tensor.shape}")
        else:
            tensor = tensor.float()
        processed_batch[key] = tensor
    
    # Ensure all required keys
    batch_size = processed_batch["image"].shape[0]
    sequence_length = processed_batch["reward"].shape[1] if processed_batch["reward"].ndim > 1 else 1
    
    for key in required_keys - set(processed_batch.keys()):
        if key == "image":
            processed_batch[key] = torch.zeros((batch_size, sequence_length, 3, 64, 64), device=device, dtype=torch.float32)
        elif key in ["is_first", "is_terminal", "is_last"]:
            tensor = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.bool)
            if key == "is_first":
                tensor[:, 0] = True
            processed_batch[key] = tensor
        elif key in ["discount", "continuation"]:
            processed_batch[key] = torch.ones((batch_size, sequence_length), device=device, dtype=torch.float32)
            if "is_terminal" in processed_batch:
                processed_batch[key][processed_batch["is_terminal"]] = 0.0
        elif key == "action":
            # Default to discrete action indices
            processed_batch[key] = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.long)
        else:
            processed_batch[key] = torch.zeros((batch_size, sequence_length), device=device, dtype=torch.float32)
    
    return processed_batch