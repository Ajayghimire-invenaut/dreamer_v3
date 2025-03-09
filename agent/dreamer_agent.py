"""
Agent module for Dreamer-V3.
Integrates the world model, actor-critic behavior, and exploration strategies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, Generator, Optional, Tuple

from agent.behavior import ImaginedBehavior
from agent.random_explorer import RandomExplorer
from agent.world_model import WorldModel

class DreamerAgent(nn.Module):
    def __init__(self,
                 observation_space: Any,
                 action_space: Any,
                 configuration: Any,
                 logger_obj: Any,
                 dataset: Generator[Dict[str, Any], None, None]) -> None:
        super(DreamerAgent, self).__init__()
        self.configuration = configuration
        self.logger = logger_obj
        self.dataset = dataset

        self.log_schedule = configuration.logging_interval
        self.training_schedule = int((configuration.batch_size * configuration.sequence_length) / configuration.training_update_ratio)
        self.pretrain_once = True
        self.exploration_schedule = configuration.exploration_termination_step // configuration.action_repeat

        self.metrics: Dict[str, list] = {}
        self.current_step = logger_obj.global_step // configuration.action_repeat
        self.update_count = 0

        self.world_model = WorldModel(observation_space, action_space, self.current_step, configuration)
        self.task_behavior = ImaginedBehavior(configuration, self.world_model)

        if configuration.compile_models and torch.cuda.is_available() and (configuration.os_name != "nt"):
            self.world_model = torch.compile(self.world_model)
            self.task_behavior = torch.compile(self.task_behavior)

        device = configuration.computation_device if torch.cuda.is_available() else "cpu"
        self.explorer = RandomExplorer(configuration, action_space)
        if configuration.actor.get("exploration_behavior", "greedy") != "greedy":
            behavior_options = {
                "greedy": lambda: self.task_behavior,
                "random": lambda: RandomExplorer(configuration, action_space),
                "plan2explore": lambda: self.task_behavior
            }
            self.explorer = behavior_options[configuration.actor.get("exploration_behavior", "greedy")]()
        self.explorer = self.explorer.to(device)

    def forward(self,
                observation: Dict[str, Any],
                reset_flags: list,
                state: Optional[Any] = None,
                training: bool = True) -> Tuple[Dict[str, Any], Any]:
        if training:
            num_updates = self.configuration.number_of_pretraining_updates if self.pretrain_once else self.training_schedule
            for _ in range(num_updates):
                self._train(next(self.dataset))
                self.update_count += 1
                self.metrics.setdefault("update_count", []).append(self.update_count)
            if self.current_step % self.log_schedule == 0:
                for metric_name, metric_values in self.metrics.items():
                    self.logger.scalar(metric_name, float(np.mean(metric_values)))
                    self.metrics[metric_name] = []
                if self.configuration.log_video_predictions:
                    video_prediction = self.world_model.generate_video(next(self.dataset))
                    self.logger.video("training_video", video_prediction.detach().cpu().numpy())
                self.logger.write(fps=True)
        policy_output, updated_state = self._compute_policy(observation, state, training)
        if training:
            self.current_step += len(reset_flags)
            self.logger.global_step = self.configuration.action_repeat * self.current_step
        return policy_output, updated_state

    def _compute_policy(self,
                        observation: Dict[str, Any],
                        state: Optional[Any],
                        training: bool) -> Tuple[Dict[str, Any], Any]:
        if state is None:
            latent_state, previous_action = None, None
        else:
            latent_state, previous_action = state

        processed_obs = self.world_model.preprocess(observation)
        embedding = self.world_model.encoder(processed_obs)
        latent_state, _ = self.world_model.dynamics.observe_step(latent_state,
                                                                 previous_action,
                                                                 embedding,
                                                                 processed_obs["is_first"])
        if self.configuration.use_state_mean_for_evaluation:
            latent_state["stochastic"] = latent_state["mean"]
        features = self.world_model.dynamics.get_features(latent_state)

        if not training:
            actor_dist = self.task_behavior.actor(features)
            action_output = actor_dist.mode()
        else:
            if self.current_step < self.exploration_schedule:
                actor_dist = self.explorer.actor(features)
            else:
                actor_dist = self.task_behavior.actor(features)
            action_output = actor_dist.sample()

        log_probability = actor_dist.log_prob(action_output)
        latent_state = {k: v.detach() for k, v in latent_state.items()}
        action_output = action_output.detach()
        if self.configuration.actor.get("distribution_type", "gaussian") in ["onehot", "onehot_gumble"]:
            action_output = torch.nn.functional.one_hot(torch.argmax(action_output, dim=-1),
                                                          num_classes=self.configuration.number_of_possible_actions)
        policy_output = {"action": action_output, "log_probability": log_probability}
        new_state = (latent_state, action_output)
        return policy_output, new_state

    def _train(self, batch_data: Dict[str, Any]) -> None:
        metrics: Dict[str, float] = {}
        posterior, context, world_model_metrics = self.world_model.train_step(batch_data)
        metrics.update(world_model_metrics)
        reward_function = lambda features, state, action: self.world_model.heads["reward"](
            self.world_model.dynamics.get_features(state)
        ).mode()
        behavior_metrics = self.task_behavior.train_step(posterior, reward_function)[-1]
        metrics.update(behavior_metrics)
        if self.configuration.actor.get("exploration_behavior", "greedy") != "greedy":
            exploration_metrics = self.explorer.train(posterior, context, batch_data)[-1]
            metrics.update({f"exploration_{name}": value for name, value in exploration_metrics.items()})
        for name, value in metrics.items():
            self.metrics.setdefault(name, []).append(value)

    def collect_optimizer_states(self) -> Dict[str, Any]:
        return {
            "world_model": self.world_model.get_optimizer_state(),
            "task_behavior": self.task_behavior.get_optimizer_state()
        }

    def reset_pretraining_flag(self) -> None:
        self.pretrain_once = False
