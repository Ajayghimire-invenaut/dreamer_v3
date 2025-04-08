# Handles logging and metrics for DreamerV3 training
import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir="./logs", experiment_name=None):
        """
        Initialize the logger with TensorBoard support.
        
        Args:
            log_dir (str): Directory for logs
            experiment_name (str, optional): Custom name for the experiment
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_path)
        self.step = 0

    def log_scalar(self, name, value, step=None):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            name (str): Name of the metric
            value (float or torch.Tensor): Value to log
            step (int, optional): Step number (defaults to internal step)
        """
        if step is None:
            step = self.step
        if torch.is_tensor(value):
            value = value.item()
        self.writer.add_scalar(name, value, step)

    def log_metrics(self, metrics_dict, step=None):
        """
        Log multiple metrics at once.
        
        Args:
            metrics_dict (dict): Dictionary of name-value pairs
            step (int, optional): Step number
        """
        if step is None:
            step = self.step
        for name, value in metrics_dict.items():
            self.log_scalar(name, value, step)

    def increment_step(self):
        """Increment the internal step counter."""
        self.step += 1

    def log_episode(self, episode_reward, episode_length):
        """Log episode-level metrics."""
        self.log_scalar("episode/reward", episode_reward)
        self.log_scalar("episode/length", episode_length)
        self.increment_step()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()