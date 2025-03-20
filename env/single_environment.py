import gymnasium as gym
import numpy as np
import cv2
from typing import Any, Dict, Tuple
import time
import logging

# Logger setup: Initialize a logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to DEBUG for more detailed logs
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

class SingleEnvironment:
    def __init__(self, task_name: str, action_repeat: int = 2, seed: int = 42) -> None:
        """
        A wrapper around a Gymnasium environment for CartPole with RGB rendering.
        :param task_name: Identifier for the environment (e.g., "CartPole-v1").
        :param action_repeat: Number of times to repeat each action.
        :param seed: Random seed for reproducibility.
        """
        self.task_name = task_name
        self.action_repeat = action_repeat
        self.seed = seed
        # Create the environment with render_mode="rgb_array" (new step API is default in Gymnasium)
        self.environment = gym.make(task_name, render_mode="rgb_array")
        self.environment.reset(seed=seed)
        self.environment.action_space.seed(seed)
        self.identifier = f"{task_name}-0"
        # Define observation_space as 64x64 RGB images
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = self.environment.action_space

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step through the environment with action repetition.
        :param action: Action to take (0 or 1 for CartPole).
        :return: Tuple of (observation dict, total_reward, done, info).
        """
        step_start = time.time()
        cumulative_reward = 0.0
        done = False
        info = {}

        for _ in range(self.action_repeat):
            result = self.environment.step(action)
            observation, reward, terminated, truncated, info = result  # New step API
            done = terminated or truncated
            cumulative_reward += float(reward)
            if done:
                break

        elapsed = time.time() - step_start
        logger.info("Environment step took %.4f seconds", elapsed)

        # Render and process the RGB frame
        img = self.environment.render()
        processed_img = self._process_image(img)
        return self._format_observation(processed_img, is_first=False), cumulative_reward, done, info

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        :return: Dictionary with processed RGB image and metadata.
        """
        start_time = time.time()
        reset_result = self.environment.reset(seed=self.seed)
        observation, info = reset_result  # New step API returns (obs, info)
        elapsed = time.time() - start_time
        logger.info("Environment reset took %.4f seconds", elapsed)
        
        # Render and process the initial RGB frame
        img = self.environment.render()
        processed_img = self._process_image(img)
        return {
            "image": processed_img,
            "is_first": np.array(True, dtype=bool),
            "is_terminal": np.array(False, dtype=bool),
            "discount": np.array(1.0, dtype=np.float32),
        }

    def _process_image(self, img: Any) -> np.ndarray:
        """
        Process the rendered RGB image to 64x64.
        :param img: Raw RGB image from render().
        :return: Processed 64x64 RGB image as uint8.
        """
        start_time = time.time()
        try:
            if img is None:
                logger.warning("render() returned None. Creating a dummy image.")
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                img = np.array(img)
                # Handle potential extra batch dimension (e.g., [1, H, W, C])
                if img.ndim == 4 and img.shape[0] == 1:
                    img = np.squeeze(img, axis=0)
                    logger.debug("Squeezed image shape: %s", img.shape)
                # Validate image dimensions
                if img.size == 0 or img.ndim != 3 or img.shape[0] == 0 or img.shape[1] == 0:
                    logger.warning("render() returned an invalid image. Creating a dummy image.")
                    img = np.zeros((64, 64, 3), dtype=np.uint8)
                else:
                    # Resize to 64x64
                    resize_start = time.time()
                    resized = cv2.resize(img, (64, 64))
                    logger.debug("cv2.resize took %.4f seconds", time.time() - resize_start)
                    img = resized.astype(np.uint8)
            return img
        except Exception as e:
            logger.error("Error in _process_image: %s", e)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        finally:
            logger.debug("Total _process_image took %.4f seconds", time.time() - start_time)

    def _format_observation(self, processed_img: np.ndarray, is_first: bool) -> Dict[str, Any]:
        """
        Format the observation into a dictionary.
        :param processed_img: Processed 64x64 RGB image.
        :param is_first: Whether this is the first step after reset.
        :return: Dictionary with observation data.
        """
        return {
            "image": processed_img,
            "is_first": np.array(is_first, dtype=bool),
            "is_terminal": np.array(False, dtype=bool),
            "discount": np.array(1.0, dtype=np.float32),
        }

    def close(self) -> None:
        """
        Close the environment.
        """
        self.environment.close()