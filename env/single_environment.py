import gymnasium as gym
import numpy as np
import cv2
from typing import Any, Dict, Tuple, Optional, Union
import time
import logging
import traceback
import torch

# Setup logger for debugging and information
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

class SingleEnvironment:
    """
    Wrapper for a Gymnasium environment tailored for DreamerV3.
    Provides image-based observations, action repetition, and episode tracking with is_last support.
    """
    def __init__(
        self,
        task_name: str,
        action_repeat: int = 1,
        seed: int = 42,
        image_size: int = 64,
        enable_debugging: bool = False
    ) -> None:
        """
        Initialize the environment wrapper with specified settings.
        
        Args:
            task_name: Name of the environment (e.g., "CartPole-v1")
            action_repeat: Number of times to repeat each action
            seed: Random seed for reproducibility
            image_size: Size of the square observation images
            enable_debugging: Whether to enable detailed debug logging
        """
        self.task_name = task_name
        self.action_repeat = max(1, action_repeat)
        self.seed = seed
        self.image_size = image_size
        self.enable_debugging = enable_debugging
        
        if self.enable_debugging:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled for SingleEnvironment")
            
        self.episode_steps = 0
        self.total_episodes = 0
        self.total_reward = 0.0
        self.identifier = f"{task_name}-0"
        
        try:
            self.environment = gym.make(task_name, render_mode="rgb_array")
            self.environment.reset(seed=seed)
            self.environment.action_space.seed(seed)
            
            self.original_observation_space = self.environment.observation_space
            self.action_space = self.environment.action_space
            
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8
            )
            
            self.max_episode_steps = getattr(self.environment.spec, "max_episode_steps", 1000) if hasattr(self.environment, "spec") else 1000
            
            logger.info(f"Initialized {task_name} environment with action_repeat={action_repeat}, max_episode_steps={self.max_episode_steps}")
            if self.enable_debugging:
                logger.debug(f"Action space: {self.action_space}")
                logger.debug(f"Original observation space: {self.original_observation_space}")
                logger.debug(f"Wrapped observation space: {self.observation_space}")
        except Exception as error:
            logger.error(f"Error initializing environment: {error}")
            self._create_fallback_environment()

    def _create_fallback_environment(self) -> None:
        """Create a fallback dummy environment if initialization fails."""
        logger.warning("Creating fallback environment")
        try:
            self.environment = gym.make("CartPole-v1", render_mode="rgb_array")
            self.environment.reset(seed=self.seed)
            self.action_space = self.environment.action_space
            self.original_observation_space = self.environment.observation_space
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8
            )
            self.max_episode_steps = 500
        except Exception as error:
            logger.error(f"Failed to create fallback environment: {error}")
            self.environment = None
            self.action_space = gym.spaces.Discrete(2)
            self.original_observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4,),
                dtype=np.float32
            )
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size, 3),
                dtype=np.uint8
            )
            self.max_episode_steps = 1000

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute a step in the environment with action repetition.
        
        Args:
            action: Action to execute in the environment
            
        Returns:
            Tuple of (observation_dict, cumulative_reward, done, info)
        """
        step_start_time = time.time()
        cumulative_reward = 0.0
        done = False
        info = {}
        
        processed_action = self._process_action(action)
        
        if self.enable_debugging:
            logger.debug(f"Original action: {action}, type: {type(action)}")
            logger.debug(f"Processed action: {processed_action}, type: {type(processed_action)}")
        
        for _ in range(self.action_repeat):
            if self.environment is None:
                logger.warning("Environment is None during step, using dummy values")
                observation = np.zeros((4,), dtype=np.float32)
                reward = 0.0
                terminated = True
                truncated = False
                done = True
                info = {"error": "No environment available"}
            else:
                try:
                    observation, reward, terminated, truncated, info = self.environment.step(processed_action)
                    done = terminated or truncated
                except Exception as error:
                    logger.error(f"Error in environment step: {error}")
                    observation = np.zeros((4,), dtype=np.float32)
                    reward = 0.0
                    terminated = True
                    truncated = False
                    done = True
                    info = {"error": str(error)}
            
            cumulative_reward += float(reward)
            self.total_reward += float(reward)
            self.episode_steps += 1
            
            is_last = truncated or (self.episode_steps >= self.max_episode_steps)
            is_terminal = terminated and not truncated
            
            if done or is_last:
                if self.enable_debugging:
                    logger.debug(f"Episode {'terminated' if terminated else 'truncated'} after {self.episode_steps} steps with total reward {self.total_reward}")
                if terminated or is_last:
                    self.total_episodes += 1
                break
        
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8) if done else self._get_frame()
        processed_image = self._process_image(image)
        observation_dict = self._format_observation(processed_image, is_first=False, is_terminal=is_terminal, is_last=is_last, reward=cumulative_reward)
        
        # Add action to observation_dict with consistent shape
        if hasattr(self.action_space, 'n'):
            action_array = np.zeros(self.action_space.n, dtype=np.float32)
            action_array[processed_action] = 1.0 if isinstance(processed_action, (int, np.integer)) else 0.0
            observation_dict["action"] = action_array[np.newaxis, :]  # Shape: [1, action_dimension]
        else:
            action_array = np.array(processed_action, dtype=np.float32)
            if action_array.ndim == 0:
                action_array = action_array[np.newaxis]
            observation_dict["action"] = action_array[np.newaxis, :]  # Shape: [1, action_dimension]
        
        if self.enable_debugging:
            elapsed_time = time.time() - step_start_time
            logger.debug(f"Environment step took {elapsed_time:.4f} seconds")
        
        return observation_dict, cumulative_reward, done, info

    def _process_action(self, action: Any) -> Any:
        """
        Process the action to match the environment's action space, ensuring validity.
        
        Args:
            action: Input action to process
            
        Returns:
            Processed action suitable for the environment
        """
        try:
            if isinstance(self.action_space, gym.spaces.Discrete):
                num_actions = self.action_space.n
                if isinstance(action, np.ndarray):
                    action_index = int(np.argmax(action)) if action.size > 1 else int(action.item())
                elif isinstance(action, (list, tuple)):
                    action_index = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                elif torch.is_tensor(action):
                    action_index = int(torch.argmax(action).item()) if action.numel() > 1 else int(action.item())
                else:
                    action_index = int(action)
                
                action_index = np.clip(action_index, 0, num_actions - 1)
                if not self.action_space.contains(action_index):
                    logger.warning(f"Action {action_index} not in action space {self.action_space}, clamped")
                return action_index
            
            elif isinstance(self.action_space, gym.spaces.Box):
                action_array = action.detach().cpu().numpy() if torch.is_tensor(action) else np.array(action, dtype=np.float32)
                if action_array.ndim > 1:
                    action_array = action_array.flatten()
                if action_array.size != self.action_space.shape[0]:
                    if action_array.size > self.action_space.shape[0]:
                        action_array = action_array[:self.action_space.shape[0]]
                    else:
                        padded = np.zeros(self.action_space.shape[0], dtype=np.float32)
                        padded[:action_array.size] = action_array
                        action_array = padded
                return np.clip(action_array, self.action_space.low, self.action_space.high)
            
        except Exception as error:
            logger.error(f"Error processing action {action}: {error}")
            return 0 if isinstance(self.action_space, gym.spaces.Discrete) else np.zeros(self.action_space.shape, dtype=np.float32)
        
        return action

    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment and return the initial observation.
        
        Returns:
            Initial observation dictionary
        """
        start_time = time.time()
        self.episode_steps = 0
        self.total_reward = 0.0
        
        if self.environment is None:
            logger.warning("Environment is None during reset, using dummy observation")
            observation = np.zeros((4,), dtype=np.float32)
            info = {}
        else:
            try:
                reset_result = self.environment.reset(seed=self.seed)
                observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                info = reset_result[1] if isinstance(reset_result, tuple) and len(reset_result) > 1 else {}
            except Exception as error:
                logger.error(f"Error resetting environment: {error}")
                observation = np.zeros((4,), dtype=np.float32)
                info = {"error": str(error)}
        
        image = self._get_frame()
        processed_image = self._process_image(image)
        observation_dict = self._format_observation(processed_image, is_first=True, is_terminal=False, is_last=False, reward=0.0)
        
        # Add initial action with consistent shape
        if hasattr(self.action_space, 'n'):
            action_array = np.zeros(self.action_space.n, dtype=np.float32)
            action_array[0] = 1.0
            observation_dict["action"] = action_array[np.newaxis, :]  # Shape: [1, action_dimension]
        else:
            observation_dict["action"] = np.zeros(self.action_space.shape, dtype=np.float32)[np.newaxis, :]  # Shape: [1, action_dimension]
        
        if self.enable_debugging:
            elapsed_time = time.time() - start_time
            logger.debug(f"Environment reset took {elapsed_time:.4f} seconds")
        
        return observation_dict

    def _get_frame(self) -> np.ndarray:
        """Retrieve the current frame from the environment."""
        if self.environment is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        try:
            image = self.environment.render()
            return image if image is not None else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        except Exception as error:
            logger.error(f"Error in render: {error}")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the rendered RGB image to the desired size and format.
        
        Args:
            image: Raw image from the environment
            
        Returns:
            Processed RGB image
        """
        if self.enable_debugging:
            start_time = time.time()
        
        if image is None or image.size == 0:
            logger.warning("Image is None or empty, returning dummy image")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image_array = np.array(image)
        if image_array.ndim == 4 and image_array.shape[0] == 1:
            image_array = image_array.squeeze(0)
        
        if image_array.ndim != 3 or image_array.shape[0] == 0 or image_array.shape[1] == 0:
            logger.warning(f"Invalid image dimensions {image_array.shape}, returning dummy image")
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # Ensure channels-last format (H, W, C)
        if image_array.shape[2] not in [1, 3, 4]:
            if image_array.shape[0] in [1, 3, 4]:
                image_array = np.transpose(image_array, (1, 2, 0))
        
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)
        elif image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        interpolation = cv2.INTER_AREA if image_array.shape[0] > self.image_size or image_array.shape[1] > self.image_size else cv2.INTER_LINEAR
        resized_image = cv2.resize(image_array, (self.image_size, self.image_size), interpolation=interpolation)
        
        if resized_image.dtype != np.uint8:
            resized_image = (resized_image * 255).astype(np.uint8) if resized_image.max() <= 1.0 else resized_image.astype(np.uint8)
        
        if self.enable_debugging:
            logger.debug(f"Image processing took {time.time() - start_time:.4f} seconds")
        
        return resized_image

    def _format_observation(self, processed_image: np.ndarray, is_first: bool, is_terminal: bool, is_last: bool, reward: float) -> Dict[str, Any]:
        """
        Format the observation into a dictionary for DreamerV3 with is_last support.
        
        Args:
            processed_image: Processed RGB image
            is_first: Whether this is the first step of an episode
            is_terminal: Whether this is a true termination (not truncation)
            is_last: Whether this is the last step (including truncation)
            reward: Reward for the step
            
        Returns:
            Observation dictionary with required fields
        """
        if processed_image is None or processed_image.size == 0:
            logger.warning("Invalid image in _format_observation, using dummy image")
            processed_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        if processed_image.ndim != 3 or processed_image.shape[2] != 3:
            logger.warning(f"Image has unexpected shape {processed_image.shape}, converting to RGB")
            if processed_image.ndim == 2:
                processed_image = np.repeat(processed_image[:, :, np.newaxis], 3, axis=2)
            else:
                processed_image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        if processed_image.shape[0] != self.image_size or processed_image.shape[1] != self.image_size:
            processed_image = cv2.resize(processed_image, (self.image_size, self.image_size))
        
        # Add batch dimension for consistency
        processed_image = processed_image[np.newaxis, :]  # Shape: [1, H, W, C]
        
        return {
            "image": processed_image,
            "is_first": np.array([is_first], dtype=bool),
            "is_terminal": np.array([is_terminal], dtype=bool),
            "is_last": np.array([is_last], dtype=bool),
            "discount": np.array([0.0 if is_terminal else 1.0], dtype=np.float32),
            "reward": np.array([reward], dtype=np.float32),
            "step_count": np.array([self.episode_steps], dtype=np.int32),
            "continuation": np.array([0.0 if is_terminal else 1.0], dtype=np.float32)
        }

    def render(self) -> np.ndarray:
        """
        Render the environment and return the RGB image.
        
        Returns:
            Current environment frame as an RGB image
        """
        frame = self._get_frame()
        return frame[np.newaxis, :]  # Shape: [1, H, W, C]

    def close(self) -> None:
        """Close the environment and release resources."""
        if self.environment is not None:
            try:
                self.environment.close()
                logger.info("Environment closed")
            except Exception as error:
                logger.error(f"Error closing environment: {error}")

    @property
    def unwrapped(self) -> gym.Env:
        """
        Access the underlying unwrapped environment.
        
        Returns:
            Unwrapped Gymnasium environment
        """
        if self.environment is None:
            logger.warning("Attempted to access unwrapped environment when environment is None")
            return None
        return self.environment.unwrapped
        
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retrieve statistics about the environment's performance.
        
        Returns:
            Dictionary containing episode and step statistics
        """
        return {
            "total_episodes": self.total_episodes,
            "current_episode_steps": self.episode_steps,
            "current_episode_reward": self.total_reward
        }
        
    def __del__(self) -> None:
        """Clean up resources when the instance is deleted."""
        self.close()