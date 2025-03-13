import torch
import time
import collections
import io
import numpy as np
import pathlib
from typing import Any, Dict, Generator
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

    def __iter__(self):
      count = 0
      while count < self.max_samples:
          episode = self.rng.choice(self.episodes)
          total_length = len(next(iter(episode.values())))
          # If the episode is too short, pad it.
          segment = {}
          for key, value in episode.items():
              arr = np.array(value)
              if arr.shape[0] < self.sequence_length:
                  pad_length = self.sequence_length - arr.shape[0]
                  # Use the last value as the pad; modify if needed.
                  pad_value = arr[-1]
                  pad_array = np.full((pad_length,) + arr.shape[1:], pad_value)
                  seg = np.concatenate([arr, pad_array], axis=0)
              else:
                  start_index = self.rng.integers(0, total_length - self.sequence_length + 1)
                  seg = arr[start_index:start_index + self.sequence_length]
              segment[key] = seg
          # Mark the first time step as a reset.
          if "is_first" in segment:
              segment["is_first"][0] = True
          print(f"[Dataset Debug] Yielding segment {count}: keys={list(segment.keys())}, action shape={segment.get('action', 'N/A').shape}", flush=True)
          yield segment
          count += 1


def create_dataset(episodes: Dict[str, Any], configuration: Any) -> Generator[Dict[str, Any], None, None]:
    """
    Creates a dataset generator for training.
    Wraps EpisodeDataset in a DataLoader and returns an iterator.
    """
    dataset = EpisodeDataset(
        episodes,
        sequence_length=configuration.sequence_length,
        seed=configuration.random_seed,
        max_samples=10000  # Adjust if needed.
    )
    dataloader = DataLoader(dataset, batch_size=configuration.batch_size)
    return iter(dataloader)

def from_generator(generator: Generator[Dict[str, Any], None, None], batch_size: int) -> Generator[Dict[str, Any], None, None]:
    while True:
        batch = [next(generator) for _ in range(batch_size)]
        data: Dict[str, Any] = {}
        for key in batch[0].keys():
            data[key] = np.stack([episode[key] for episode in batch], axis=0)
        print(f"[Dataset Debug] Yielding batch: keys={list(data.keys())}, action shape={(data['action'].shape if 'action' in data else 'N/A')}", flush=True)
        yield data

def sample_episodes(episodes: Dict[str, Any], sequence_length: int, seed: int = 0) -> Generator[Dict[str, Any], None, None]:
    """
    Legacy infinite generator (use create_dataset for new code).
    """
    np_random = np.random.RandomState(seed)
    if len(episodes) == 0:
        raise ValueError("No episodes found. Please prefill the dataset.")
    iteration = 0
    while True:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"[Dataset Sampling] Iteration {iteration}", flush=True)
        episode = np_random.choice(list(episodes.values()))
        total_length = len(next(iter(episode.values())))
        if total_length < sequence_length:
            continue
        start_index = np_random.randint(0, total_length - sequence_length + 1)
        segment = {}
        for key, value in episode.items():
            seg = np.array(value[start_index:start_index + sequence_length])
            if key == "action" and seg.shape[0] == sequence_length - 1:
                pad_value = seg[-1] if seg.shape[0] > 0 else 0
                seg = np.concatenate([seg, np.array([pad_value])], axis=0)
            segment[key] = seg
        if "is_first" in segment:
            segment["is_first"][0] = True
        yield segment

def load_episode_data(directory: str, limit: Any = None, reverse: bool = True) -> Dict[str, Any]:
    directory_path = pathlib.Path(directory).expanduser()
    episodes: Dict[str, Any] = collections.OrderedDict()
    total_steps = 0
    files = sorted(directory_path.glob("*.npz"), reverse=reverse)
    for file_path in files:
        try:
            with file_path.open("rb") as file:
                episode = np.load(file)
                episode = {key: episode[key] for key in episode.keys()}
            required_keys = ["action", "reward", "is_first", "is_terminal"]
            missing_keys = [k for k in required_keys if k not in episode]
            if missing_keys:
                print(f"[Dataset Debug] Skipping {file_path} as it lacks: {missing_keys}", flush=True)
                continue
        except Exception as error:
            print(f"[Dataset Debug] Failed to load {file_path}: {error}", flush=True)
            continue
        episodes[str(file_path.stem)] = episode
        total_steps += len(episode.get("reward", [])) - 1
        if limit and total_steps >= limit:
            break
    print(f"[Dataset Debug] Loaded {len(episodes)} episodes with total_steps={total_steps}", flush=True)
    return episodes

def simulate_episode(agent_or_policy: Any,
                     environment: Any,
                     episodes: Dict[str, Any],
                     save_directory: str,
                     logger: Any,
                     is_evaluation: bool = False,
                     episodes_num: int = 0,
                     steps: int = 0,
                     state: Any = None) -> tuple:
    """
    Simulates an episode using the given agent or policy.
    Returns a tuple: (total_steps_taken, total_episodes_completed, done_flag, agent_state).
    """
    if state is None:
        step_count, episode_count = 0, 0
        done_flag = True
        agent_state = None
    else:
        step_count, episode_count, done_flag, agent_state = state

    while ((steps and step_count < steps) or (episodes_num and episode_count < episodes_num)):
        iter_start = time.time()
        if done_flag:
            observation = environment.reset()
            if "image" in observation:
                print(f"[Simulate Episode Debug] Reset observation image shape: {np.array(observation['image']).shape}", flush=True)
            episode_data = {key: [observation[key]] for key in observation}
            episode_data["reward"] = [0.0]
            episode_data["action"] = []
        action, agent_state = agent_or_policy(observation, [done_flag], agent_state)
        act = action["action"]
        if isinstance(act, torch.Tensor):
            if act.dim() == 2 and act.size(1) > 1:
                act = int(torch.argmax(act, dim=-1).item())
            else:
                act = int(act.item())
        episode_data["action"].append(act)
        obs_start = time.time()
        observation, reward, done_flag, info = environment.step(act)
        obs_elapsed = time.time() - obs_start
        print(f"[Simulate Episode Debug] Step observation processing took {obs_elapsed:.4f} seconds", flush=True)
        if "image" in observation:
            print(f"[Simulate Episode Debug] Step observation image shape: {np.array(observation['image']).shape}", flush=True)
        for key in observation:
            episode_data.setdefault(key, []).append(observation[key])
        episode_data.setdefault("reward", []).append(reward)
        step_count += 1
        if step_count % 10 == 0:
            print(f"[Simulate Episode Debug] Step {step_count} Reward: {reward}", flush=True)
        if done_flag:
            episode_count += 1
            print(f"[Simulate Episode Debug] Episode finished at step {step_count}, saving episode.", flush=True)
            save_episode(save_directory, environment.identifier, episode_data)
        iter_elapsed = time.time() - iter_start
        print(f"[Simulate Episode Debug] Iteration took {iter_elapsed:.4f} seconds", flush=True)
    return step_count, episode_count, done_flag, agent_state

def save_episode(directory: str, environment_identifier: str, episode: Dict[str, Any]) -> None:
    directory_path = pathlib.Path(directory).expanduser()
    directory_path.mkdir(parents=True, exist_ok=True)
    episode_length = len(episode["reward"])
    filename = directory_path / f"{environment_identifier}-{episode_length}.npz"
    with io.BytesIO() as temporary_file:
        np.savez_compressed(temporary_file, **episode)
        temporary_file.seek(0)
        with filename.open("wb") as output_file:
            output_file.write(temporary_file.read())
