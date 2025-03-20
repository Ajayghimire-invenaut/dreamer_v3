import time
import collections
import io
import numpy as np
import pathlib
import torch
from typing import Any, Dict, Generator, Tuple
from torch.utils.data import IterableDataset, DataLoader

class EpisodeDataset(IterableDataset):
    def __init__(self, episodes: Dict[str, Any], sequence_length: int, seed: int = 0, max_samples: int = 10000):
        """
        An iterable dataset that samples fixed-length segments from episodes.
        :param episodes: Dictionary mapping episode IDs to episode data.
        :param sequence_length: The fixed length for each sample.
        :param seed: Random seed for reproducibility.
        :param max_samples: Maximum number of segments to yield in one epoch.
        """
        self.episodes = list(episodes.values())
        self.sequence_length = sequence_length
        self.rng = np.random.default_rng(seed)
        self.max_samples = max_samples
        self.debug = False  # Set externally via configuration
        if not self.episodes:
            raise ValueError("No episodes provided to the dataset.")

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        count = 0
        while count < self.max_samples:
            # Randomly choose an episode
            episode = self.rng.choice(self.episodes)
            total_length = len(next(iter(episode.values())))
            segment = {}
            # Sample or pad a segment to the desired length
            if total_length < self.sequence_length:
                for key, value in episode.items():
                    arr = np.array(value, dtype=np.float32)
                    pad_length = self.sequence_length - arr.shape[0]
                    if key == "image":  # Pad image with last frame
                        pad_value = arr[-1]
                        pad_array = np.tile(pad_value[np.newaxis, ...], (pad_length, 1, 1, 1))
                    elif key in ["action", "reward", "is_first", "is_terminal"]:  # Pad scalars with last value
                        pad_value = arr[-1]
                        pad_array = np.full((pad_length,) + arr.shape[1:], pad_value, dtype=arr.dtype)
                    else:
                        pad_array = np.zeros((pad_length,) + arr.shape[1:], dtype=arr.dtype)
                    segment[key] = np.concatenate([arr, pad_array], axis=0)
            else:
                start_index = self.rng.integers(0, total_length - self.sequence_length + 1)
                for key, value in episode.items():
                    arr = np.array(value, dtype=np.float32)
                    segment[key] = arr[start_index:start_index + self.sequence_length]
            # Ensure the first time step is marked as a reset if not already set
            if "is_first" in segment:
                segment["is_first"][0] = True
            if "is_terminal" not in segment:
                segment["is_terminal"] = np.zeros_like(segment["is_first"], dtype=np.float32)
            if "reward" in segment and segment["reward"].ndim == 1:
                segment["reward"] = segment["reward"][:, np.newaxis]  # [T] → [T, 1]
            if self.debug:
                print(f"[DEBUG EpisodeDataset] Yielding segment {count}: keys={list(segment.keys())}, "
                      f"action shape={segment.get('action', 'N/A').shape}", flush=True)
            yield segment
            count += 1

def create_dataset(episodes: Dict[str, Any], configuration: Any) -> Generator[Dict[str, Any], None, None]:
    """
    Creates a dataset generator for training.
    Wraps EpisodeDataset in a DataLoader and returns an iterator with batch-first data [B, T, ...].
    """
    dataset = EpisodeDataset(
        episodes,
        sequence_length=configuration.sequence_length,
        seed=configuration.random_seed,
        max_samples=getattr(configuration, "max_samples_per_epoch", 10000)
    )
    dataset.debug = getattr(configuration, "debug", False)
    dataloader = DataLoader(
        dataset,
        batch_size=configuration.batch_size,
        num_workers=0,
        collate_fn=lambda x: {k: np.stack([d[k] for d in x], axis=0) for k in x[0].keys()}  # [B, T, ...]
    )
    if dataset.debug:
        print(f"[DEBUG create_dataset] Batch size: {configuration.batch_size}, Sequence length: {configuration.sequence_length}", flush=True)
    return iter(dataloader)

def from_generator(generator: Generator[Dict[str, Any], None, None], batch_size: int) -> Generator[Dict[str, Any], None, None]:
    """
    Continuously collects batch_size samples from the generator and collates them.
    Each key is stacked along axis 0, so the batch dimension is first: [B, T, ...].
    """
    while True:
        batch = [next(generator) for _ in range(batch_size)]
        data = {key: np.stack([episode[key] for episode in batch], axis=0) for key in batch[0].keys()}
        if getattr(generator, "debug", False):
            print(f"[DEBUG from_generator] Yielding batch: keys={list(data.keys())}, "
                  f"action shape={data.get('action', 'N/A').shape}", flush=True)
        yield data

def sample_episodes(episodes: Dict[str, Any], sequence_length: int, seed: int = 0) -> Generator[Dict[str, Any], None, None]:
    """
    Legacy infinite generator yielding [T, ...] segments. Use create_dataset for new code.
    """
    np_random = np.random.RandomState(seed)
    if not episodes:
        raise ValueError("No episodes found. Please prefill the dataset.")
    iteration = 0
    while True:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"[DEBUG sample_episodes] Iteration {iteration}", flush=True)
        episode = np_random.choice(list(episodes.values()))
        total_length = len(next(iter(episode.values())))
        if total_length < sequence_length:
            continue
        start_index = np_random.randint(0, total_length - sequence_length + 1)
        segment = {}
        for key, value in episode.items():
            seg = np.array(value[start_index:start_index + sequence_length], dtype=np.float32)
            if key == "action" and seg.shape[0] == sequence_length - 1:
                pad_value = seg[-1] if seg.size > 0 else 0
                seg = np.concatenate([seg, np.array([pad_value], dtype=seg.dtype)], axis=0)
            segment[key] = seg
        if "is_first" in segment:
            segment["is_first"][0] = True
        if "is_terminal" not in segment:
            segment["is_terminal"] = np.zeros_like(segment["is_first"], dtype=np.float32)
        yield segment

def load_episode_data(directory: str, limit: Any = None, reverse: bool = True) -> Dict[str, Any]:
    """
    Loads episode data from .npz files in the given directory.
    Only episodes containing the required keys are loaded.
    """
    directory_path = pathlib.Path(directory).expanduser()
    episodes: Dict[str, Any] = collections.OrderedDict()
    total_steps = 0
    files = sorted(directory_path.glob("*.npz"), reverse=reverse)
    required_keys = {"action", "reward", "is_first", "is_terminal", "image"}
    for file_path in files:
        try:
            with file_path.open("rb") as file:
                episode = np.load(file)
                episode = {key: episode[key] for key in episode.keys()}
            missing_keys = required_keys - set(episode.keys())
            if missing_keys:
                print(f"[DEBUG load_episode_data] Skipping {file_path} as it lacks: {missing_keys}", flush=True)
                continue
        except Exception as error:
            print(f"[DEBUG load_episode_data] Failed to load {file_path}: {error}", flush=True)
            continue
        episodes[str(file_path.stem)] = episode
        total_steps += len(episode.get("reward", []))
        if limit and total_steps >= limit:
            break
    print(f"[DEBUG load_episode_data] Loaded {len(episodes)} episodes with total_steps={total_steps}", flush=True)
    return episodes

def simulate_episode(agent_or_policy: Any,
                     environment: Any,
                     episodes: Dict[str, Any],
                     save_directory: str,
                     logger: Any,
                     is_evaluation: bool = False,
                     episodes_num: int = 0,
                     steps: int = 0,
                     state: Any = None) -> Tuple[int, int, bool, Any]:
    """
    Simulates an episode using the given agent or policy.
    Returns a tuple: (total_steps_taken, total_episodes_completed, done_flag, agent_state).
    Ensures batch-first data in episodes: [B, T, ...].
    """
    debug = getattr(agent_or_policy, "debug", False)
    if state is None:
        step_count, episode_count, done_flag, agent_state = 0, 0, True, None
    else:
        step_count, episode_count, done_flag, agent_state = state

    while (steps and step_count < steps) or (episodes_num and episode_count < episodes_num):
        iter_start = time.time()
        if done_flag:
            observation = environment.reset()
            if isinstance(observation, dict) and "image" in observation and debug:
                print(f"[DEBUG simulate_episode] Reset observation image shape: {np.array(observation['image']).shape}", flush=True)
            episode_data = {key: [val] for key, val in observation.items()}
            episode_data["reward"] = [0.0]
            episode_data["action"] = []
        
        action, agent_state = agent_or_policy(observation, [done_flag], agent_state)
        act = action["action"]
        if isinstance(act, torch.Tensor):
            act = act.cpu().numpy()
            if act.ndim == 2 and act.shape[1] > 1:  # One-hot [B, num_actions]
                act = int(np.argmax(act))
            else:
                act = int(act.item())
        episode_data["action"].append(act)
        
        obs_start = time.time()
        observation, reward, done_flag, info = environment.step(act)
        obs_elapsed = time.time() - obs_start
        if debug:
            print(f"[DEBUG simulate_episode] Step observation processing took {obs_elapsed:.4f} seconds", flush=True)
        if isinstance(observation, dict) and "image" in observation and debug:
            print(f"[DEBUG simulate_episode] Step observation image shape: {np.array(observation['image']).shape}", flush=True)
        
        for key, val in observation.items():
            episode_data.setdefault(key, []).append(val)
        episode_data["reward"].append(float(reward))
        step_count += 1
        
        if step_count % 10 == 0 and debug:
            print(f"[DEBUG simulate_episode] Step {step_count} Reward: {reward}", flush=True)
        
        if done_flag:
            episode_count += 1
            if debug:
                print(f"[DEBUG simulate_episode] Episode {episode_count} finished at step {step_count}, saving episode.", flush=True)
            save_episode(save_directory, environment.identifier, episode_data)
            episodes[f"{environment.identifier}-{episode_count}"] = {k: np.array(v) for k, v in episode_data.items()}
        
        iter_elapsed = time.time() - iter_start
        if debug:
            print(f"[DEBUG simulate_episode] Iteration took {iter_elapsed:.4f} seconds", flush=True)
    
    return step_count, episode_count, done_flag, agent_state

def save_episode(directory: str, environment_identifier: str, episode: Dict[str, Any]) -> None:
    directory_path = pathlib.Path(directory).expanduser()
    directory_path.mkdir(parents=True, exist_ok=True)
    episode_length = len(episode["reward"])
    filename = directory_path / f"{environment_identifier}-{episode_length}-{int(time.time())}.npz"
    with io.BytesIO() as temporary_file:
        np.savez_compressed(temporary_file, **{k: np.array(v) for k, v in episode.items()})
        temporary_file.seek(0)
        with filename.open("wb") as output_file:
            output_file.write(temporary_file.read())