import numpy as np
from dm_control import suite
import torch

class EnvironmentManager:
    def __init__(self, task_name="walker_walk", domain_name="walker", image_size=64, action_repeat=2):
        """
        Initialize the environment manager for DMC tasks.
        
        Args:
            task_name (str): Task name (e.g., "walk" for walker_walk)
            domain_name (str): Domain name (e.g., "walker")
            image_size (int): Size of rendered image (e.g., 64 for 64x64)
            action_repeat (int): Number of times to repeat actions
        """
        self.task_name = task_name
        self.domain_name = domain_name
        self.image_size = image_size
        self.action_repeat = action_repeat
        
        # Load DMC environment
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.action_space = self.env.action_spec()
        self.action_size = self.action_space.shape[0]
        self.wrapped_env = None

    def _wrap_environment(self):
        """Wrap the DMC environment for image observations and action repeat."""
        class DMCWrapper:
            def __init__(self, env, image_size, action_repeat):
                self.env = env
                self.image_size = image_size
                self.action_repeat = action_repeat
                self.action_space = {
                    'minimum': env.action_spec().minimum,
                    'maximum': env.action_spec().maximum,
                    'shape': env.action_spec().shape
                }
                self.observation_space = {
                    'shape': (3, image_size, image_size),
                    'dtype': np.uint8,
                    'low': 0,
                    'high': 255
                }

            def reset(self):
                time_step = self.env.reset()
                return self._get_observation(time_step)

            def step(self, action):
                reward_sum = 0.0
                for _ in range(self.action_repeat):
                    time_step = self.env.step(action)
                    reward_sum += time_step.reward or 0.0
                    if time_step.last():
                        break
                obs = self._get_observation(time_step)
                done = time_step.last()
                info = {}
                return obs, reward_sum, done, info

            def _get_observation(self, time_step):
                pixels = self.env.physics.render(
                    height=self.image_size,
                    width=self.image_size,
                    camera_id=0
                )
                # Ensure contiguous array with positive strides
                return np.transpose(pixels, (2, 0, 1)).copy()

        return DMCWrapper(self.env, self.image_size, self.action_repeat)

    def reset(self):
        if self.wrapped_env is None:
            self.wrapped_env = self._wrap_environment()
        obs = self.wrapped_env.reset()
        # Double-check contiguity
        return torch.tensor(obs.copy(), dtype=torch.float32)

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action.cpu().numpy())
        return (torch.tensor(obs.copy(), dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
                done,
                info)

    def get_action_space_size(self):
        return self.action_size

    def close(self):
        if self.wrapped_env is not None:
            self.wrapped_env.env.close()