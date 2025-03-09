import torch
import collections
import io
import numpy as np
import pathlib
from typing import Any, Dict, Generator

def create_dataset(episodes: Dict[str, Any], configuration: Any) -> Generator[Dict[str, Any], None, None]:
    generator = sample_episodes(episodes, configuration.sequence_length, seed=configuration.random_seed)
    return from_generator(generator, configuration.batch_size)

def from_generator(generator: Generator[Dict[str, Any], None, None], batch_size: int) -> Generator[Dict[str, Any], None, None]:
    while True:
        batch = [next(generator) for _ in range(batch_size)]
        data: Dict[str, Any] = {}
        for key in batch[0].keys():
            data[key] = np.stack([episode[key] for episode in batch], axis=0)
        return_data = data  # You may add a debug print here if needed.
        # Uncomment the following line to print the shapes for each key:
        # print("Batch shapes:", {k: v.shape for k, v in return_data.items()}, flush=True)
        yield return_data

def sample_episodes(episodes: Dict[str, Any], sequence_length: int, seed: int = 0) -> Generator[Dict[str, Any], None, None]:
    """
    Samples a segment of a fixed length from the episodes.
    Pads the 'action' key if it is one element shorter than sequence_length.
    """
    np_random = np.random.RandomState(seed)
    if len(episodes) == 0:
        raise ValueError("No episodes found in the dataset. Please prefill the dataset.")
    iteration = 0
    while True:
        iteration += 1
        if iteration % 1000 == 0:
            print(f"[Dataset Sampling] Iteration {iteration}", flush=True)
        episode = np_random.choice(list(episodes.values()))
        # Use one key (e.g., 'reward') to determine total length.
        total_length = len(next(iter(episode.values())))
        if total_length < sequence_length:
            continue
        start_index = np_random.randint(0, total_length - sequence_length + 1)
        segment = {}
        for key, value in episode.items():
            seg = np.array(value[start_index: start_index + sequence_length])
            # For 'action', if its length is one less than expected, pad it.
            if key == "action" and seg.shape[0] == sequence_length - 1:
                pad_value = seg[-1] if seg.shape[0] > 0 else 0
                seg = np.concatenate([seg, np.array([pad_value])], axis=0)
            segment[key] = seg
        # Ensure that the first time step is marked as a reset.
        if "is_first" in segment:
            segment["is_first"][0] = True
        yield segment

def load_episode_data(directory: str, limit: Any = None, reverse: bool = True) -> Dict[str, Any]:
    """
    Loads episodes from the given directory.
    Skips any episode that does not include all required keys.
    """
    directory_path = pathlib.Path(directory).expanduser()
    episodes: Dict[str, Any] = collections.OrderedDict()
    total_steps = 0
    files = sorted(directory_path.glob("*.npz"), reverse=reverse)
    for file_path in files:
        try:
            with file_path.open("rb") as file:
                episode = np.load(file)
                episode = {key: episode[key] for key in episode.keys()}
            # Check for required keys.
            required_keys = ["action", "reward", "is_first", "is_terminal"]
            missing_keys = [k for k in required_keys if k not in episode]
            if missing_keys:
                print(f"Skipping {file_path} as it lacks the following keys: {missing_keys}", flush=True)
                continue
        except Exception as error:
            print(f"Failed to load {file_path}: {error}", flush=True)
            continue
        episodes[str(file_path.stem)] = episode
        total_steps += len(episode.get("reward", [])) - 1
        if limit and total_steps >= limit:
            break
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
    Simulates an episode using a given agent or policy, saves the episode, and returns simulation state.
    """
    if state is None:
        step_count, episode_count = 0, 0
        done_flag = True
        agent_state = None
    else:
        step_count, episode_count, done_flag, agent_state = state

    while ((steps and step_count < steps) or (episodes_num and episode_count < episodes_num)):
        if done_flag:
            observation = environment.reset()
            # Debug print: show shape of the image observation.
            if "image" in observation:
                print("Reset observation image shape:", np.array(observation["image"]).shape, flush=True)
            episode_data = {key: [observation[key]] for key in observation}
            episode_data["reward"] = [0.0]
            # Ensure "action" key is always present.
            episode_data["action"] = []
        action, agent_state = agent_or_policy(observation, [done_flag], agent_state)
        act = action["action"]
        if isinstance(act, torch.Tensor):
            if act.dim() == 2 and act.size(1) > 1:
                act = int(torch.argmax(act, dim=-1).item())
            else:
                act = int(act.item())
        # Save the action taken.
        episode_data["action"].append(act)
        observation, reward, done_flag, info = environment.step(act)
        # Debug print: show shape of the new observation image.
        if "image" in observation:
            print("Step observation image shape:", np.array(observation["image"]).shape, flush=True)
        for key in observation:
            episode_data.setdefault(key, []).append(observation[key])
        episode_data.setdefault("reward", []).append(reward)
        step_count += 1
        if step_count % 10 == 0:
            print(f"[Simulate Episode] Step {step_count} Reward: {reward}", flush=True)
        if done_flag:
            episode_count += 1
            save_episode(save_directory, environment.identifier, episode_data)
    return step_count, episode_count, done_flag, agent_state

def save_episode(directory: str, environment_identifier: str, episode: Dict[str, Any]) -> None:
    """
    Saves an episode as a compressed .npz file.
    """
    directory_path = pathlib.Path(directory).expanduser()
    directory_path.mkdir(parents=True, exist_ok=True)
    episode_length = len(episode["reward"])
    filename = directory_path / f"{environment_identifier}-{episode_length}.npz"
    with io.BytesIO() as temporary_file:
        np.savez_compressed(temporary_file, **episode)
        temporary_file.seek(0)
        with filename.open("wb") as output_file:
            output_file.write(temporary_file.read())
