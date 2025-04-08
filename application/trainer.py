import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os  # Added for checkpoint path handling
from core.world_domain.dynamics import WorldDynamics
from core.world_domain.encoder import ObservationEncoder
from core.world_domain.decoder import ObservationDecoder
from core.world_domain.reward_head import RewardPredictor
from core.behavior_domain.actor import PolicyActor
from core.behavior_domain.critic import ValueCritic
from core.utilities.data_tools import DataTools
from core.utilities.replay_buffer import ReplayBuffer
from infrastructure.environment_manager import EnvironmentManager
from infrastructure.logger import TrainingLogger

class DreamerTrainer:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.tools = DataTools()
        self.step = 0  # Track training step

        self.env = EnvironmentManager(
            task_name=config["task_name"],
            domain_name=config["domain_name"],
            image_size=config["image_size"],
            action_repeat=config["action_repeat"]
        )
        
        self.encoder = ObservationEncoder(input_channels=3).to(device)
        self.decoder = ObservationDecoder(latent_size=config["stoch_size"]).to(device)
        self.dynamics = WorldDynamics(
            latent_size=config["latent_size"],
            action_size=self.env.get_action_space_size(),
            stoch_size=config["stoch_size"],
            deter_size=config["deter_size"]
        ).to(device)
        self.reward_predictor = RewardPredictor(latent_size=config["stoch_size"]).to(device)
        
        self.actor = PolicyActor(
            latent_size=config["stoch_size"],
            action_size=self.env.get_action_space_size()
        ).to(device)
        self.critic = ValueCritic(latent_size=config["stoch_size"]).to(device)
        
        self.world_optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) +
            list(self.dynamics.parameters()) + list(self.reward_predictor.parameters()),
            lr=config["world_lr"]
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["actor_lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["critic_lr"])
        
        self.logger = TrainingLogger(log_dir=config["log_dir"], experiment_name=config["experiment_name"])
        
        self.buffer = ReplayBuffer(
            capacity=config["buffer_capacity"],
            obs_shape=(3, config["image_size"], config["image_size"]),
            action_size=self.env.get_action_space_size(),
            sequence_length=config["batch_length"],
            batch_size=config["batch_size"],
            device=self.device
        )
        self.checkpoint_path = os.path.join(config["log_dir"], "checkpoint.pt")  # Default checkpoint path

    def save_checkpoint(self, step):
        """Save model state and optimizer state."""
        checkpoint = {
            "step": step,
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "dynamics_state": self.dynamics.state_dict(),
            "reward_predictor_state": self.reward_predictor.state_dict(),
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "world_optimizer_state": self.world_optimizer.state_dict(),
            "actor_optimizer_state": self.actor_optimizer.state_dict(),
            "critic_optimizer_state": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load model state and optimizer state."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint["encoder_state"])
            self.decoder.load_state_dict(checkpoint["decoder_state"])
            self.dynamics.load_state_dict(checkpoint["dynamics_state"])
            self.reward_predictor.load_state_dict(checkpoint["reward_predictor_state"])
            self.actor.load_state_dict(checkpoint["actor_state"])
            self.critic.load_state_dict(checkpoint["critic_state"])
            self.world_optimizer.load_state_dict(checkpoint["world_optimizer_state"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
            print(f"Loaded checkpoint from step {checkpoint['step']}")
            return checkpoint["step"]
        return 0

    def collect_data(self, num_steps=100):
        obs = self.env.reset().to(self.device)
        total_reward = 0.0
        episode_length = 0
        max_episode_length = 1000
        
        for _ in range(num_steps):
            with torch.no_grad():
                stoch, deter = self.dynamics.initial_state(1, self.device)
                enc_obs = self.encoder(obs.unsqueeze(0).unsqueeze(0))[:, 0]
                stoch, deter, _, _ = self.dynamics(stoch, deter, torch.zeros(1, self.env.get_action_space_size(), device=self.device), enc_obs)
                action = self.actor.act(stoch, deterministic=False)
            
            next_obs, reward, done, _ = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, done)
            total_reward += reward.item()
            episode_length += 1
            
            print(f"Step {episode_length}: Reward={reward.item()}, Done={done}")
            
            obs = next_obs
            if done or episode_length % 100 == 0 or episode_length >= max_episode_length:
                print(f"Logging episode: Total Reward={total_reward}, Length={episode_length}")
                self.logger.log_episode(total_reward, episode_length, step=self.step)
                if done or episode_length >= max_episode_length:
                    print("Episode completed!")
                    obs = self.env.reset().to(self.device)
                    total_reward = 0.0
                    episode_length = 0

    def train_step(self):
        batch = self.buffer.sample()
        if batch is None:
            return
        
        obs, actions, rewards, next_obs, dones = batch
        
        obs = self.tools.preprocess_observation(obs, self.device)
        rewards = self.tools.symlog(rewards)

        # World model training
        self.world_optimizer.zero_grad()
        stoch, deter = self.dynamics.initial_state(self.config["batch_size"], self.device)
        enc_obs = self.encoder(obs)
        kl_losses, recon_losses, reward_losses = [], [], []
        
        for t in range(self.config["batch_length"]):
            action_t = actions[:, t]
            stoch, deter, post_dist, prior_dist = self.dynamics(stoch, deter, action_t, enc_obs[:, t])
            recon_obs = self.decoder(stoch.unsqueeze(1)).squeeze(1)
            pred_reward = self.reward_predictor(stoch.unsqueeze(1)).squeeze(1)
            
            kl_loss = self.tools.compute_kl_loss(post_dist, prior_dist, self.config["kl_free"])
            recon_loss = F.mse_loss(recon_obs, obs[:, t])
            reward_loss = F.mse_loss(pred_reward, rewards[:, t])
            
            kl_losses.append(kl_loss.mean())
            recon_losses.append(recon_loss)
            reward_losses.append(reward_loss)
        
        world_loss = (self.config["kl_scale"] * sum(kl_losses) / len(kl_losses) +
                      sum(recon_losses) / len(recon_losses) +
                      sum(reward_losses) / len(reward_losses))
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_optimizer.param_groups[0]["params"], 1.0)
        self.world_optimizer.step()

        # Imagination-based behavior training (fully detached)
        with torch.no_grad():
            stoch, deter = self.dynamics.initial_state(self.config["batch_size"], self.device)
            for t in range(self.config["batch_length"]):
                action_t = actions[:, t]
                stoch, deter, _, _ = self.dynamics(stoch, deter, action_t)
            initial_stoch = stoch.detach()
            initial_deter = deter.detach()

        # Actor training
        self.actor_optimizer.zero_grad()
        stoch, deter = initial_stoch, initial_deter
        imagined_states, imagined_rewards = [], []
        for h in range(self.config["horizon"]):
            action_dist, action = self.actor(stoch.unsqueeze(1))
            action = action.squeeze(1)
            stoch, deter, _, _ = self.dynamics(stoch, deter, action)
            reward = self.reward_predictor(stoch.unsqueeze(1)).squeeze(1)
            imagined_states.append(stoch)
            imagined_rewards.append(reward)
        
        imagined_rewards = torch.stack(imagined_rewards, dim=1)
        with torch.no_grad():
            imagined_values = torch.stack([self.critic(s.unsqueeze(1)).squeeze(1) for s in imagined_states], dim=1)
            returns = self.tools.compute_lambda_returns(imagined_rewards, imagined_values,
                                                        discount=self.config["discount"],
                                                        lambda_=self.config["lambda_"])
        
        actor_loss = -torch.mean(returns) + self.config["actor_ent"] * action_dist.entropy().mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_optimizer.param_groups[0]["params"], 1.0)
        self.actor_optimizer.step()

        # Critic training (new graph)
        self.critic_optimizer.zero_grad()
        stoch, deter = initial_stoch, initial_deter
        imagined_states = []
        for h in range(self.config["horizon"]):
            with torch.no_grad():
                action_dist, action = self.actor(stoch.unsqueeze(1))
                action = action.squeeze(1)
            stoch, deter, _, _ = self.dynamics(stoch, deter, action)
            imagined_states.append(stoch)
        
        imagined_values = torch.stack([self.critic(s.unsqueeze(1)).squeeze(1) for s in imagined_states], dim=1)
        with torch.no_grad():
            imagined_rewards = torch.stack([self.reward_predictor(s.unsqueeze(1)).squeeze(1) for s in imagined_states], dim=1)
            returns = self.tools.compute_lambda_returns(imagined_rewards, imagined_values,
                                                        discount=self.config["discount"],
                                                        lambda_=self.config["lambda_"])
        
        critic_loss = F.mse_loss(imagined_values, returns.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_optimizer.param_groups[0]["params"], 1.0)
        self.critic_optimizer.step()

        # Logging
        metrics = {
            "world/kl_loss": sum(kl_losses) / len(kl_losses),
            "world/recon_loss": sum(recon_losses) / len(recon_losses),
            "world/reward_loss": sum(reward_losses) / len(reward_losses),
            "actor/loss": actor_loss.item(),
            "critic/loss": critic_loss.item()
        }
        self.logger.log_metrics(metrics, step=self.step)

    def train(self):
        start_step = self.load_checkpoint()  # Load checkpoint if exists
        self.step = start_step
        for step in range(start_step, self.config["max_steps"]):
            self.collect_data(num_steps=self.config["collect_steps"])
            self.train_step()
            self.step += 1
            if self.step % 100 == 0:
                print(f"Step {self.step}: Training in progress...")
            if self.step % 1000 == 0 and self.step > 0:  # Save checkpoint every 1,000 steps
                self.save_checkpoint(self.step)
        
        self.env.close()
        self.logger.close()