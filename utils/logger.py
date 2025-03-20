"""
Logger module.
Logs training metrics, images, and videos to TensorBoard and a JSONL file.
"""

import json
import pathlib
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, Optional

class Logger:
    def __init__(self, log_directory: pathlib.Path, global_step: int = 0) -> None:
        """
        Initialize the Logger with a directory and starting global step.
        
        :param log_directory: Path object for the log directory.
        :param global_step: Initial global step for logging.
        """
        self.log_directory = pathlib.Path(log_directory).expanduser()
        self.log_directory.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_directory), max_queue=1000)
        self.last_step: Optional[int] = None
        self.last_time: Optional[float] = None
        self.scalar_metrics: Dict[str, float] = {}
        self.image_metrics: Dict[str, np.ndarray] = {}
        self.video_metrics: Dict[str, np.ndarray] = {}
        self.global_step = global_step

    def scalar(self, name: str, value: Any) -> None:
        """
        Log a scalar metric.
        
        :param name: Name of the scalar metric.
        :param value: Value to log (converted to float).
        """
        try:
            self.scalar_metrics[name] = float(value)
        except (TypeError, ValueError) as e:
            print(f"[Logger Warning] Failed to convert scalar '{name}' value {value} to float: {e}", flush=True)

    def image(self, name: str, value: np.ndarray) -> None:
        """
        Log an image metric.
        
        :param name: Name of the image metric.
        :param value: Image array [C, H, W] or [H, W, C].
        """
        value = np.array(value, dtype=np.float32)
        if value.ndim == 3 and value.shape[-1] in [1, 3, 4]:  # [H, W, C]
            value = value.transpose(2, 0, 1)  # [C, H, W] for TensorBoard
        elif value.ndim not in [2, 3]:
            print(f"[Logger Warning] Skipping image '{name}': Expected 2D or 3D array, got shape {value.shape}", flush=True)
            return
        self.image_metrics[name] = np.clip(value, 0, 1)  # Ensure values are in [0, 1]

    def video(self, name: str, value: np.ndarray) -> None:
        """
        Log a video metric.
        
        :param name: Name of the video metric.
        :param value: Video array [B, T, C, H, W] or [T, H, W, C].
        """
        value = np.array(value, dtype=np.float32)
        if value.ndim == 5 and value.shape[2] in [1, 3, 4]:  # [B, T, C, H, W]
            pass
        elif value.ndim == 4 and value.shape[-1] in [1, 3, 4]:  # [T, H, W, C]
            value = value.transpose(0, 3, 1, 2)  # [T, C, H, W]
            value = value[np.newaxis, ...]  # [1, T, C, H, W] for batch dimension
        else:
            print(f"[Logger Warning] Skipping video '{name}': Expected [B, T, C, H, W] or [T, H, W, C], got shape {value.shape}", flush=True)
            return
        self.video_metrics[name] = np.clip(value, 0, 1)  # Ensure values are in [0, 1]

    def write(self, fps: bool = False) -> Dict[str, float]:
        """
        Write all logged metrics to TensorBoard and JSONL file, then clear buffers.
        
        :param fps: If True, compute and log frames per second.
        :return: Dictionary of scalar metrics logged in this step.
        """
        current_step = self.global_step
        metrics_to_return = dict(self.scalar_metrics)
        metrics_list = list(self.scalar_metrics.items())
        
        if fps:
            fps_value = self._compute_fps(current_step)
            metrics_list.append(("frames_per_second", fps_value))
            metrics_to_return["frames_per_second"] = fps_value
        
        # Log to console
        metrics_str = " / ".join(f"{key}: {value:.2f}" for key, value in metrics_list)
        print(f"[Step {current_step}] {metrics_str}", flush=True)

        # Write to JSONL file
        metrics_file = self.log_directory / "metrics.jsonl"
        with metrics_file.open("a") as file:
            file.write(json.dumps({"step": current_step, **dict(metrics_list)}) + "\n")

        # Write to TensorBoard
        for name, value in metrics_list:
            self.tensorboard_writer.add_scalar(f"scalars/{name}", value, current_step)
        for name, value in self.image_metrics.items():
            try:
                self.tensorboard_writer.add_image(name, value, current_step, dataformats="CHW")
            except Exception as e:
                print(f"[Logger Warning] Failed to log image '{name}' to TensorBoard: {e}", flush=True)
        for name, value in self.video_metrics.items():
            try:
                self.tensorboard_writer.add_video(name, value, current_step, fps=16)
            except Exception as e:
                print(f"[Logger Warning] Failed to log video '{name}' to TensorBoard: {e}", flush=True)
        
        self.tensorboard_writer.flush()
        self.global_step += 1  # Increment step after writing

        # Clear buffers
        self.scalar_metrics.clear()
        self.image_metrics.clear()
        self.video_metrics.clear()
        return metrics_to_return

    def _compute_fps(self, current_step: int) -> float:
        """
        Compute frames per second based on step difference and elapsed time.
        
        :param current_step: Current global step.
        :return: Frames per second as a float.
        """
        if self.last_step is None:
            self.last_time = time.time()
            self.last_step = current_step
            return 0.0
        step_diff = current_step - self.last_step
        time_elapsed = time.time() - self.last_time
        self.last_time = time.time()
        self.last_step = current_step
        return step_diff / time_elapsed if time_elapsed > 0 else 0.0

    def close(self) -> None:
        """
        Close the TensorBoard writer.
        """
        self.tensorboard_writer.close()