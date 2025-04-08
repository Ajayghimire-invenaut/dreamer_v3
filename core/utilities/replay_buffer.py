import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_size, sequence_length, batch_size, device):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        
        # Use lists instead of pre-allocated arrays
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

    def add(self, obs, action, reward, next_obs, done):
        if len(self.observations) >= self.capacity:
            self.observations.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_observations.pop(0)
            self.dones.pop(0)
        
        self.observations.append(obs.cpu().numpy())
        self.actions.append(action.cpu().numpy())
        self.rewards.append(reward.item() if torch.is_tensor(reward) else reward)
        self.next_observations.append(next_obs.cpu().numpy())
        self.dones.append(done)

    def sample(self):
        if len(self.observations) < self.sequence_length + 1:
            return None
        
        valid_indices = len(self.observations) - self.sequence_length
        if valid_indices <= 0:
            return None
        start_indices = np.random.randint(0, valid_indices, self.batch_size)
        
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []
        
        for start in start_indices:
            indices = range(start, start + self.sequence_length)
            obs_batch.append(np.stack([self.observations[i] for i in indices]))
            action_batch.append(np.stack([self.actions[i] for i in indices]))
            reward_batch.append(np.stack([self.rewards[i] for i in indices]))
            next_obs_batch.append(np.stack([self.next_observations[i] for i in indices]))
            done_batch.append(np.stack([self.dones[i] for i in indices]))
        
        obs = torch.tensor(np.stack(obs_batch), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(action_batch), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.stack(reward_batch), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_obs = torch.tensor(np.stack(next_obs_batch), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.stack(done_batch), dtype=torch.bool, device=self.device).unsqueeze(-1)
        
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.observations)