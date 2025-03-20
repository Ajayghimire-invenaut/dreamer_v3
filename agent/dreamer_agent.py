import torch
import torch.nn as nn
import numpy as np
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
        self.debug = configuration.debug  # Propagate debug flag

        self.log_schedule = configuration.logging_interval
        self.training_updates = configuration.training_updates_per_forward
        self.pretrain_once = True
        self.exploration_schedule = configuration.exploration_termination_step // configuration.action_repeat

        self.metrics: Dict[str, list] = {}
        self.current_step = logger_obj.global_step // configuration.action_repeat
        self.update_count = 0

        # Instantiate world model and imagined behavior modules
        self.world_model = WorldModel(observation_space, action_space, self.current_step, configuration)
        self.task_behavior = ImaginedBehavior(configuration, self.world_model)

        # Optionally compile models for speed (disabled on CPU)
        if configuration.compile_models and torch.cuda.is_available() and (configuration.os_name != "nt"):
            self.world_model = torch.compile(self.world_model)
            self.task_behavior = torch.compile(self.task_behavior)
        else:
            if self.debug:
                print("[DEBUG DreamerAgent] Model compilation skipped: CPU or unsupported OS", flush=True)

        device = configuration.computation_device if torch.cuda.is_available() else "cpu"
        self.to(device)

        # Set up a random explorer for early-stage exploration
        self.explorer = RandomExplorer(configuration, action_space)
        if configuration.actor.get("exploration_behavior", "greedy") != "greedy":
            behavior_options = {
                "greedy": lambda: self.task_behavior,
                "random": lambda: RandomExplorer(configuration, action_space),
                "plan2explore": lambda: self.task_behavior
            }
            self.explorer = behavior_options[configuration.actor.get("exploration_behavior", "greedy")]()
        self.explorer = self.explorer.to(device)

        if self.debug:
            print(f"[DEBUG DreamerAgent] Initialized on {device}, "
                  f"training_updates: {self.training_updates}, "
                  f"exploration_schedule: {self.exploration_schedule}", flush=True)

    def forward(self,
                observation: Dict[str, Any],
                reset_flags: list,
                state: Optional[Any] = None,
                training: bool = True) -> Tuple[Dict[str, Any], Any]:
        if training:
            # Perform multiple training updates per forward call
            for _ in range(self.training_updates):
                self._train(next(self.dataset))
                self.update_count += 1
                self.metrics.setdefault("update_count", []).append(self.update_count)
            # Log metrics periodically
            if self.current_step % self.log_schedule == 0:
                for metric_name, metric_values in self.metrics.items():
                    values_cpu = [v.cpu().item() if hasattr(v, "cpu") else v for v in metric_values]
                    self.logger.scalar(metric_name, float(np.mean(values_cpu)))
                    self.metrics[metric_name] = []
                if self.configuration.log_video_predictions:
                    video_prediction = self.world_model.generate_video(next(self.dataset))
                    self.logger.video("training_video", video_prediction.detach().cpu().numpy())
                self.logger.write(fps=True)
        policy_output, updated_state = self._compute_policy(observation, state, training)
        self.current_step += 1  # Increment step after each forward pass
        return policy_output, updated_state

    def _compute_policy(self,
                        observation: Dict[str, Any],
                        state: Optional[Any],
                        training: bool) -> Tuple[Dict[str, Any], Any]:
        # Unpack previous state and action (if any)
        if state is None:
            latent_state, previous_action = None, None
        else:
            latent_state, previous_action = state

        # Preprocess observation and compute embedding
        processed_obs = self.world_model.preprocess(observation)  # [B, C, H, W] or [B, T, C, H, W]
        embedding = self.world_model.encoder(processed_obs)  # [B, embed_dim] or [B, T, embed_dim]
        if embedding.dim() == 3:  # [B, T, embed_dim]
            embedding = embedding[:, -1, :]  # Take last timestep: [B, embed_dim]
        
        # Update state using single-step RSSM transition
        is_first = processed_obs["is_first"]
        if is_first.dim() == 2:  # [B, 1]
            is_first = is_first.squeeze(1)  # [B]
        latent_state, _ = self.world_model.dynamics.observe_step(
            latent_state,
            previous_action,
            embedding,
            is_first
        )
        # Use mean state for evaluation if configured
        if self.configuration.use_state_mean_for_evaluation and "mean" in latent_state and not training:
            latent_state["stochastic"] = latent_state["mean"]

        # Extract features from the updated state
        features = self.world_model.dynamics.get_features(latent_state)  # [B, feature_dim] or [B, T, feature_dim]
        if features.dim() == 3:  # [B, T, feature_dim]
            features = features[:, -1, :]  # Take last timestep: [B, feature_dim]
        if self.debug:
            print("[DEBUG _compute_policy] Features shape:", features.shape, flush=True)

        # Select policy and compute action
        if not training:
            actor_dist = self.task_behavior.actor(features)  # [B, num_actions]
            action_output = actor_dist.mode()  # [B, num_actions]
            action_indices = None  # Not needed for evaluation
        else:
            if self.current_step < self.exploration_schedule:
                actor_dist = self.explorer.actor(features)  # [B, num_actions]
                if self.debug:
                    print("[DEBUG _compute_policy] Using explorer policy", flush=True)
            else:
                actor_dist = self.task_behavior.actor(features)  # [B, num_actions]
                if self.debug:
                    print("[DEBUG _compute_policy] Using task behavior policy", flush=True)
            action_indices = actor_dist.sample()  # [B] or [B, action_dim]
            if action_indices.dim() == 1:  # Discrete action indices
                action_output = torch.nn.functional.one_hot(
                    action_indices.long(),
                    num_classes=self.configuration.number_of_possible_actions
                ).float()  # [B, num_actions]
            else:  # Continuous actions
                action_output = action_indices

        # Compute log probability of the chosen action
        if action_indices is not None and action_indices.dim() == 1:  # Discrete case
            log_probability = actor_dist.log_prob(action_indices)  # [B]
        else:  # Continuous or evaluation case
            log_probability = actor_dist.log_prob(action_output)  # [B] or [B, num_actions]

        # Detach state and action to prevent actor gradients from flowing into the world model
        latent_state = {k: v.detach() for k, v in latent_state.items()}
        action_output = action_output.detach()

        policy_output = {"action": action_output, "log_probability": log_probability}
        new_state = (latent_state, action_output)
        if self.debug:
            print("[DEBUG _compute_policy] Action output shape:", action_output.shape, flush=True)
            print("[DEBUG _compute_policy] Log probability shape:", log_probability.shape, flush=True)
        return policy_output, new_state

    def _train(self, batch_data: Dict[str, Any]) -> None:
        metrics: Dict[str, float] = {}
        # Train the world model
        posterior, context, world_model_metrics = self.world_model.train_step(batch_data)
        metrics.update(world_model_metrics)
        # Train the behavior (actor & critic) using the posterior state
        behavior_metrics = self.task_behavior.train_step(posterior, None)[-1]
        metrics.update(behavior_metrics)
        # Optionally train the explorer if using a non-greedy exploration strategy
        if self.configuration.actor.get("exploration_behavior", "greedy") != "greedy" and self.current_step < self.exploration_schedule:
            exploration_metrics = self.explorer.train(posterior, context, batch_data)[-1]
            metrics.update({f"exploration_{name}": value for name, value in exploration_metrics.items()})
        for name, value in metrics.items():
            self.metrics.setdefault(name, []).append(value)
        if self.debug:
            print(f"[DEBUG _train] Metrics: {metrics}", flush=True)

    def train_step(self, batch_data: Dict[str, Any]) -> None:
        """Explicit training step for external calls"""
        self._train(batch_data)

    def collect_optimizer_states(self) -> Dict[str, Any]:
        return {
            "world_model": self.world_model.get_optimizer_state(),
            "task_behavior": self.task_behavior.collect_optimizer_states()
        }

    def reset_pretraining_flag(self) -> None:
        self.pretrain_once = False