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
    def __init__(self, log_directory: pathlib.Path, global_step: int) -> None:
        self.log_directory = log_directory
        self.tensorboard_writer = SummaryWriter(log_dir=str(log_directory), max_queue=1000)
        self.last_step: Optional[int] = None
        self.last_time: Optional[float] = None
        self.scalar_metrics: Dict[str, float] = {}
        self.image_metrics: Dict[str, np.ndarray] = {}
        self.video_metrics: Dict[str, np.ndarray] = {}
        self.global_step = global_step

    def scalar(self, name: str, value: float) -> None:
        self.scalar_metrics[name] = float(value)

    def image(self, name: str, value: np.ndarray) -> None:
        self.image_metrics[name] = np.array(value)

    def video(self, name: str, value: np.ndarray) -> None:
        self.video_metrics[name] = np.array(value)

    def write(self, fps: bool = False) -> Dict[str, float]:
        current_step = self.global_step
        # Save a copy of scalar metrics to return later.
        metrics_to_return = dict(self.scalar_metrics)
        metrics_list = list(self.scalar_metrics.items())
        if fps:
            metrics_list.append(("frames_per_second", self._compute_fps(current_step)))
        metrics_str = " / ".join(f"{key}: {value:.2f}" for key, value in metrics_list)
        print(f"[Step {current_step}] {metrics_str}")
        metrics_file = self.log_directory / "metrics.jsonl"
        with metrics_file.open("a") as file:
            file.write(json.dumps({"step": current_step, **dict(metrics_list)}) + "\n")
        for name, value in metrics_list:
            self.tensorboard_writer.add_scalar(f"scalars/{name}", value, current_step)
        for name, value in self.image_metrics.items():
            self.tensorboard_writer.add_image(name, value, current_step)
        for name, value in self.video_metrics.items():
            self.tensorboard_writer.add_video(name, value, current_step, fps=16)
        self.tensorboard_writer.flush()
        # Clear the metrics after writing.
        self.scalar_metrics.clear()
        self.image_metrics.clear()
        self.video_metrics.clear()
        return metrics_to_return

    def _compute_fps(self, current_step: int) -> float:
        if self.last_step is None:
            self.last_time = time.time()
            self.last_step = current_step
            return 0.0
        step_diff = current_step - self.last_step
        time_elapsed = time.time() - self.last_time
        self.last_time = time.time()
        self.last_step = current_step
        return step_diff / time_elapsed if time_elapsed > 0 else 0.0
