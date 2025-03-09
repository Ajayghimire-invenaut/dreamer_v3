"""
Module creating a single environment instance.
Wraps the Gym CartPole-v1 environment with a consistent interface,
but returns rendered images (e.g. 64×64×3) instead of raw vector states.
"""

import gym
from gym.wrappers import TimeLimit
import numpy as np
import cv2  # For image resizing
from typing import Any, Dict, Tuple

class SingleEnvironment:
    def __init__(self, task_name: str, action_repeat: int = 2, seed: int = 42) -> None:
        # Set render_mode to "rgb_array" so that render() returns an image.
        self.environment = gym.make("CartPole-v1", render_mode="rgb_array")
        self.environment = TimeLimit(self.environment, max_episode_steps=500)
        self.environment.reset(seed=seed)
        self.environment.action_space.seed(seed)
        self.action_repeat = action_repeat
        self.identifier = "environment_0"
        # Override observation_space to be image-based (e.g., 64x64 RGB).
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = self.environment.action_space

    def reset(self) -> Dict[str, Any]:
        reset_result = self.environment.reset(seed=None)
        # If gym.reset() returns a tuple (observation, info), use only observation.
        if isinstance(reset_result, tuple):
            observation = reset_result[0]
        else:
            observation = reset_result
        img = self.environment.render()
        processed_img = self._process_image(img)
        return {"image": processed_img,
                "is_first": np.array(True),
                "is_terminal": np.array(False),
                "discount": np.array(1.0)}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict]:
        cumulative_reward = 0.0
        done = False
        info = {}
        # Loop over action repeats.
        for _ in range(self.action_repeat):
            # Get the result from step.
            result = self.environment.step(action)
            if len(result) == 5:
                observation, reward, terminated, truncated, info = result
                done = terminated or truncated
            elif len(result) == 4:
                observation, reward, done, info = result
            else:
                raise ValueError("Unexpected number of values returned from environment.step")
            cumulative_reward += reward
            if done:
                break
        # Use the rendered image as the observation.
        img = self.environment.render()
        processed_img = self._process_image(img)
        return self._format_observation(processed_img, is_first=False), cumulative_reward, done, info

    def _process_image(self, img: Any) -> np.ndarray:
      try:
          if img is None:
              print("Warning: render() returned None. Creating a dummy image.")
              img = np.zeros((64, 64, 3), dtype=np.uint8)
          else:
              img = np.array(img)
          # Check if the image has at least 2 dimensions and non-zero size in the first two dimensions.
          if img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
              print("Warning: render() returned an invalid image shape. Creating a dummy image.")
              img = np.zeros((64, 64, 3), dtype=np.uint8)
          resized = cv2.resize(img, (64, 64))
          return resized.astype(np.uint8)
      except Exception as e:
          print("Error in _process_image:", e)
          return np.zeros((64, 64, 3), dtype=np.uint8)

    def _format_observation(self, processed_img: np.ndarray, is_first: bool) -> Dict[str, Any]:
        return {
            "image": processed_img,
            "is_first": np.array(is_first, dtype=bool),
            "is_terminal": np.array(False, dtype=bool),
            "discount": np.array(1.0, dtype=np.float32)
        }

    def close(self) -> None:
        self.environment.close()
