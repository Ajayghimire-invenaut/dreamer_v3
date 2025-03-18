import gym
from gym.wrappers import TimeLimit
import numpy as np
import cv2
from typing import Any, Dict
import time
import logging

# Logger setup: Initialize a logger for the module.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Change to DEBUG for more detailed logs.
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)

class SingleEnvironment:
    def __init__(self, task_name: str, action_repeat: int = 2, seed: int = 42) -> None:
        # Create the environment with the new_step_api enabled.
        self.environment = gym.make("CartPole-v1", render_mode="rgb_array", new_step_api=True)
        # Wrap with TimeLimit if needed.
        self.environment = TimeLimit(self.environment, max_episode_steps=500)
        # Reset the environment to initialize it.
        self.environment.reset(seed=seed)
        self.environment.action_space.seed(seed)
        self.action_repeat = action_repeat
        self.identifier = "environment_0"
        # Override observation_space to be image-based (e.g. 64x64 RGB).
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self.action_space = self.environment.action_space

    def reset(self) -> Dict[str, Any]:
        start_time = time.time()
        reset_result = self.environment.reset(seed=None)
        elapsed = time.time() - start_time
        logger.info("Environment reset took %.4f seconds", elapsed)
        
        # The new_step_api returns a tuple (observation, info) sometimes.
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        img = self.environment.render()
        processed_img = self._process_image(img)
        return {
            "image": processed_img,
            "is_first": np.array(True),
            "is_terminal": np.array(False),
            "discount": np.array(1.0, dtype=np.float32),
        }

    def step(self, action: Any) -> (Dict[str, Any], float, bool, Dict[str, Any]):
        step_start = time.time()
        cumulative_reward = 0.0
        done = False
        info = {}

        for _ in range(self.action_repeat):
            result = self.environment.step(action)
            # Depending on the API, result can have 4 or 5 elements.
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

        logger.info("Environment step took %.4f seconds", time.time() - step_start)

        img = self.environment.render()
        processed_img = self._process_image(img)
        return self._format_observation(processed_img, is_first=False), cumulative_reward, done, info

    def _process_image(self, img: Any) -> np.ndarray:
        start_time = time.time()
        try:
            if img is None:
                logger.warning("render() returned None. Creating a dummy image.")
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                img = np.array(img)
            # If image has an extra batch dimension (e.g. [1, H, W, C]), squeeze it.
            if img.ndim == 4 and img.shape[0] == 1:
                img = np.squeeze(img, axis=0)
                logger.debug("Squeezed image shape: %s", img.shape)
            logger.debug("Received image of type %s with shape %s and size %s", type(img), img.shape, img.size)
            # Validate image size.
            if img.size == 0 or img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
                logger.warning("render() returned an invalid image. Creating a dummy image.")
                img = np.zeros((64, 64, 3), dtype=np.uint8)
            # Resize image to 64x64.
            resize_start = time.time()
            resized = cv2.resize(img, (64, 64))
            logger.debug("cv2.resize took %.4f seconds", time.time() - resize_start)
            return resized.astype(np.uint8)
        except Exception as e:
            logger.error("Error in _process_image: %s", e)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        finally:
            logger.debug("Total _process_image took %.4f seconds", time.time() - start_time)

    def _format_observation(self, processed_img: np.ndarray, is_first: bool) -> Dict[str, Any]:
        return {
            "image": processed_img,
            "is_first": np.array(is_first, dtype=bool),
            "is_terminal": np.array(False, dtype=bool),
            "discount": np.array(1.0, dtype=np.float32),
        }

    def close(self) -> None:
        self.environment.close()
