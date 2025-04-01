import json
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional, Union, List
import logging

class Logger:
    """
    Logger for DreamerV3 that records training metrics, images, and videos.
    Outputs to TensorBoard and a JSONL file with enhanced visualization features.
    """
    def __init__(self, log_directory: Union[str, pathlib.Path], global_step: int = 0) -> None:
        """
        Initialize the Logger with a directory and starting global step.
        
        Args:
            log_directory: Path object or string for the log directory
            global_step: Initial global step for logging (reflects environment steps)
        """
        self.log_directory = pathlib.Path(log_directory).expanduser()
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer for visualization
        self.tensorboard_writer = SummaryWriter(
            log_dir=str(self.log_directory),
            max_queue=1000,
            flush_secs=10
        )
        
        # Initialize logging state
        self.last_step: Optional[int] = None
        self.last_time: Optional[float] = None
        self.scalar_metrics: Dict[str, float] = {}
        self.image_metrics: Dict[str, np.ndarray] = {}
        self.video_metrics: Dict[str, np.ndarray] = {}
        self.global_step = global_step
        self.episode_returns: List[float] = []
        
        # Ensure metrics file exists
        self.metrics_file = self.log_directory / "metrics.jsonl"
        if not self.metrics_file.exists():
            with self.metrics_file.open("w"):
                pass
        
        # Initialize training log file
        self.log_file = self.log_directory / "training.log"
        with self.log_file.open("a") as file:
            file.write(f"=== New training session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            file.write(f"Initial global step: {global_step}\n")
        
        # Configure Python logging
        self._setup_python_logging()
        
        print(f"[Logger] Initialized at {self.log_directory} with global_step={global_step}", flush=True)

    def _setup_python_logging(self) -> None:
        """Configure Python logging to both file and console outputs."""
        self.python_logger = logging.getLogger("dreamer")
        self.python_logger.setLevel(logging.INFO)
        
        # Clear existing handlers to prevent duplicates
        if self.python_logger.handlers:
            for handler in self.python_logger.handlers[:]:
                self.python_logger.removeHandler(handler)
                
        # File handler for persistent logs
        file_handler = logging.FileHandler(str(self.log_directory / "training.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.python_logger.addHandler(file_handler)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        self.python_logger.addHandler(console_handler)

    def log_scalar(self, name: str, value: Any) -> None:
        """
        Log a scalar metric to be written later.
        
        Args:
            name: Name of the metric
            value: Value of the metric
        """
        try:
            self.scalar_metrics[name] = float(value)
        except (TypeError, ValueError):
            self.warning(f"Failed to convert scalar '{name}' value {value} to float")

    def log_image(self, name: str, value: np.ndarray) -> None:
        """
        Log an image metric for visualization.
        
        Args:
            name: Name of the image metric
            value: Image array to log
        """
        try:
            image_array = np.array(value, dtype=np.float32)
            if image_array.ndim == 3 and image_array.shape[-1] in [1, 3, 4]:
                image_array = image_array.transpose(2, 0, 1)  # Convert to CHW format
            elif image_array.ndim not in [2, 3]:
                self.warning(f"Invalid image dimensions for {name}: {image_array.shape}")
                return
            self.image_metrics[name] = np.clip(image_array, 0, 1)
        except Exception as error:
            self.warning(f"Failed to process image '{name}': {error}")

    def log_video(self, name: str, value: np.ndarray) -> None:
        """
        Log a video metric for visualization.
        
        Args:
            name: Name of the video metric
            value: Video array to log
        """
        try:
            video_array = np.array(value, dtype=np.float32)
            if video_array.ndim == 5 and video_array.shape[2] in [1, 3, 4]:
                pass  # Already in correct format [B, T, C, H, W]
            elif video_array.ndim == 4 and video_array.shape[-1] in [1, 3, 4]:
                video_array = video_array.transpose(0, 3, 1, 2)[np.newaxis, ...]  # Convert to [1, T, C, H, W]
            else:
                self.warning(f"Invalid video dimensions for {name}: {video_array.shape}")
                return
            self.video_metrics[name] = np.clip(video_array, 0, 1)
        except Exception as error:
            self.warning(f"Failed to process video '{name}': {error}")

    def log_histogram(self, name: str, values: np.ndarray, bins: int = 30) -> None:
        """
        Log a histogram metric directly to TensorBoard.
        
        Args:
            name: Name of the histogram metric
            values: Values to create the histogram from
            bins: Number of bins for the histogram
        """
        try:
            self.tensorboard_writer.add_histogram(name, values, self.global_step, bins=bins)
        except Exception as error:
            self.warning(f"Failed to log histogram {name}: {error}")

    def log_episode_return(self, episode_return: float) -> None:
        """
        Log the total return of a completed episode.
        
        Args:
            episode_return: Total return achieved in the episode
        """
        self.episode_returns.append(episode_return)
        self.log_scalar("environment/episode_return", episode_return)
        
        # Log running statistics of returns
        if len(self.episode_returns) > 0:
            recent_returns = self.episode_returns[-100:]
            self.log_scalar("environment/mean_episode_return", np.mean(recent_returns))
            self.log_scalar("environment/median_episode_return", np.median(recent_returns))
            
            if len(self.episode_returns) % 10 == 0:
                self.log_histogram("environment/return_histogram", np.array(recent_returns))
                self._plot_episode_returns()

    def _plot_episode_returns(self) -> None:
        """Generate and save plots of episode returns and their moving average."""
        try:
            # Plot raw episode returns
            plt.figure(figsize=(10, 6))
            plt.plot(self.episode_returns)
            plt.title("Episode Returns")
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.grid(True)
            plt.savefig(str(self.log_directory / "episode_returns.png"))
            plt.close()
            
            # Plot moving average of returns
            if len(self.episode_returns) > 10:
                plt.figure(figsize=(10, 6))
                window_size = min(100, len(self.episode_returns))
                rolling_mean = [np.mean(self.episode_returns[max(0, i - window_size):i]) 
                                for i in range(1, len(self.episode_returns) + 1)]
                plt.plot(rolling_mean)
                plt.title(f"Moving Average Episode Return (Window={window_size})")
                plt.xlabel("Episode")
                plt.ylabel("Average Return")
                plt.grid(True)
                plt.savefig(str(self.log_directory / "average_returns.png"))
                plt.close()
        except Exception as error:
            self.warning(f"Failed to plot returns: {error}")

    def log_metrics(self, metrics: Dict[str, Any], compute_fps: bool = False, debug: bool = False) -> Dict[str, float]:
        """
        Log a dictionary of metrics, writing them to TensorBoard and JSONL file.
        
        Args:
            metrics: Dictionary of metrics to log
            compute_fps: Whether to calculate and log frames per second
            debug: Whether to enable debug-level logging
            
        Returns:
            Dictionary of scalar metrics written
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.log_scalar(name, value)
            elif isinstance(value, np.ndarray) and value.ndim in [2, 3]:
                self.log_image(name, value)
            elif isinstance(value, np.ndarray) and value.ndim in [4, 5]:
                self.log_video(name, value)
            else:
                self.warning(f"Unsupported metric type for {name}: {type(value)}")
        
        return self.write(compute_frames_per_second=compute_fps, enable_debugging=debug)

    def write(self, compute_frames_per_second: bool = False, enable_debugging: bool = False) -> Dict[str, float]:
        """
        Write all logged metrics to TensorBoard and JSONL file, then clear buffers.
        
        Args:
            compute_frames_per_second: Whether to calculate and log FPS
            enable_debugging: Whether to print detailed debug information
            
        Returns:
            Dictionary of scalar metrics written
        """
        current_step = self.global_step
        metrics_to_return = dict(self.scalar_metrics)
        metrics_list = list(self.scalar_metrics.items())
        
        # Compute FPS if requested
        if compute_frames_per_second:
            fps_value = self._compute_frames_per_second(current_step)
            metrics_list.append(("frames_per_second", fps_value))
            metrics_to_return["frames_per_second"] = fps_value
        
        # Log metrics based on debug mode
        if enable_debugging:
            metrics_string = " / ".join(f"{key}: {value:.4f}" for key, value in metrics_list)
            print(f"[DEBUG Logger] Step {current_step} | Metrics: {metrics_string}", flush=True)
            if self.image_metrics:
                print(f"[DEBUG Logger] Image metrics: {list(self.image_metrics.keys())}", flush=True)
            if self.video_metrics:
                print(f"[DEBUG Logger] Video metrics: {list(self.video_metrics.keys())}", flush=True)
        else:
            important_metrics = ["actor_loss", "value_loss", "kl_loss", "world_model_loss", "frames_per_second"]
            filtered_metrics = [(key, value) for key, value in metrics_list 
                                if any(imp in key for imp in important_metrics)]
            if filtered_metrics:
                metrics_string = " / ".join(f"{key.split('/')[-1]}: {value:.2f}" for key, value in filtered_metrics)
                print(f"[Step {current_step}] {metrics_string}", flush=True)

        # Write to JSONL file
        try:
            with self.metrics_file.open("a") as file:
                file.write(json.dumps({"step": current_step, **dict(metrics_list)}) + "\n")
        except Exception as error:
            self.warning(f"Failed to write to metrics.jsonl: {error}")

        # Log to TensorBoard
        for name, value in metrics_list:
            self.tensorboard_writer.add_scalar(f"scalars/{name}", value, current_step)
            
        for name, value in self.image_metrics.items():
            try:
                self.tensorboard_writer.add_image(name, value, current_step, dataformats="CHW")
            except Exception as error:
                self.warning(f"Failed to log image {name}: {error}")
                
        for name, value in self.video_metrics.items():
            try:
                self.tensorboard_writer.add_video(name, value, current_step, fps=16)
            except Exception as error:
                self.warning(f"Failed to log video {name}: {error}")
        
        self.tensorboard_writer.flush()
        self.global_step += 1
        
        # Clear metric buffers
        self.scalar_metrics.clear()
        self.image_metrics.clear()
        self.video_metrics.clear()
        
        return metrics_to_return

    def save_state_dictionary(self, state_dictionary: Dict[str, Any], name: str = "state_dict") -> None:
        """
        Save a state dictionary to a JSON file.
        
        Args:
            state_dictionary: Dictionary containing state data
            name: Name for the saved file
        """
        try:
            serializable_dictionary = {}
            for key, value in state_dictionary.items():
                if isinstance(value, (int, float, str, bool, list, dict, tuple)):
                    serializable_dictionary[key] = value
                else:
                    serializable_dictionary[key] = str(value)
                    
            with (self.log_directory / f"{name}.json").open("w") as file:
                json.dump(serializable_dictionary, file, indent=2)
        except Exception as error:
            self.warning(f"Failed to save state dictionary '{name}': {error}")

    def _compute_frames_per_second(self, current_step: int) -> float:
        """
        Compute frames per second based on step difference and elapsed time.
        
        Args:
            current_step: Current environment step
            
        Returns:
            Frames per second value
        """
        if self.last_step is None:
            self.last_time = time.time()
            self.last_step = current_step
            return 0.0
            
        step_difference = current_step - self.last_step
        time_elapsed = time.time() - self.last_time
        self.last_time = time.time()
        self.last_step = current_step
        
        return step_difference / time_elapsed if time_elapsed > 0 else 0.0

    def get_current_metrics(self) -> Dict[str, float]:
        """
        Retrieve the current scalar metrics without writing them.
        
        Returns:
            Dictionary of current scalar metrics
        """
        return dict(self.scalar_metrics)

    def log_hyperparameters(self, parameters: Dict[str, Any]) -> None:
        """
        Log hyperparameters to TensorBoard and a JSON file.
        
        Args:
            parameters: Dictionary of hyperparameters to log
        """
        try:
            with (self.log_directory / "hyperparameters.json").open("w") as file:
                json.dump(parameters, file, indent=2)
                
            tensorboard_parameters = {}
            for key, value in parameters.items():
                if isinstance(value, (int, float, str, bool)):
                    tensorboard_parameters[key] = value
                elif isinstance(value, (list, tuple)) and all(isinstance(x, (int, float, str, bool)) for x in value):
                    tensorboard_parameters[key] = str(value)
                    
            self.tensorboard_writer.add_hparams(tensorboard_parameters, {})
            self.info(f"Logged {len(tensorboard_parameters)} hyperparameters")
        except Exception as error:
            self.warning(f"Failed to log hyperparameters: {error}")

    def debug(self, message: str) -> None:
        """Log a debug-level message."""
        self._log("DEBUG", message)
        self.python_logger.debug(message)

    def info(self, message: str) -> None:
        """Log an info-level message."""
        self._log("INFO", message)
        self.python_logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning-level message."""
        self._log("WARNING", message)
        self.python_logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error-level message."""
        self._log("ERROR", message)
        self.python_logger.error(message)

    def _log(self, level: str, message: str) -> None:
        """
        Write a log message to the training log file.
        
        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Message to log
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level} - {message}"
        try:
            with self.log_file.open("a") as file:
                file.write(log_message + "\n")
        except Exception as error:
            print(f"[Logger Error] Failed to write to log file: {error}", flush=True)

    def log_model_parameters(self, model: Any) -> None:
        """
        Log statistics about model parameters to TensorBoard and a JSON file.
        
        Args:
            model: PyTorch model to analyze
        """
        try:
            total_parameters = 0
            layer_statistics = {}
            
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter_count = parameter.numel()
                    total_parameters += parameter_count
                    
                    module_name = name.split('.')[0]
                    if module_name not in layer_statistics:
                        layer_statistics[module_name] = 0
                    layer_statistics[module_name] += parameter_count
            
            self.log_scalar("model/total_parameters", total_parameters)
            self.info(f"Model has {total_parameters:,} trainable parameters")
            
            for module, count in layer_statistics.items():
                self.log_scalar(f"model/parameters/{module}", count)
            
            with (self.log_directory / "model_parameters.json").open("w") as file:
                json.dump({"total": total_parameters, "by_layer": layer_statistics}, file, indent=2)
        except Exception as error:
            self.warning(f"Failed to log model parameters: {error}")

    def close(self) -> None:
        """Close the TensorBoard writer and finalize logging operations."""
        try:
            self.tensorboard_writer.close()
            with self.log_file.open("a") as file:
                file.write(f"=== Session ended at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            print(f"[Logger] Closed logger at {self.log_directory}", flush=True)
        except Exception as error:
            print(f"[Logger Warning] Error closing logger: {error}", flush=True)