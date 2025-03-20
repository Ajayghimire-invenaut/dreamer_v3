"""
Main entry point for Dreamer-V3 training.
Loads configuration, sets up the environment, dataset, logger, and agent,
runs a data collection (prefill) phase if needed, prints intermediate debug metrics,
and finally plots the loss history.

Revisions:
  - Limits training updates per forward call.
  - Uses a proper imagined rollout via DreamerAgent.
  - Global step is updated based on actual environment steps with action repeat.
  - Extra debug prints around environment resets and simulation.
  - Consistent tensor dimension ordering (batch-first [B, T, ...]).
  - Uses symlog transformation utilities in DistributionWrapper.
"""

import os
import argparse
import pathlib
import torch
import time
from typing import Any
import matplotlib.pyplot as plt
import ruamel.yaml
from ruamel.yaml import YAML
import numpy as np

# Suppress deprecation warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set XDG_RUNTIME_DIR to avoid warnings
uid = os.getuid() if hasattr(os, "getuid") else "win"
os.environ["XDG_RUNTIME_DIR"] = f"/tmp/runtime-{uid}"

from agent.dreamer_agent import DreamerAgent
from env.single_environment import SingleEnvironment
from utils.dataset import create_dataset, load_episode_data, simulate_episode
from utils.logger import Logger
from utils.helper_functions import set_random_seed, enable_deterministic_mode, count_episode_steps
from parallel.parallel import Parallel

# Global lists for loss histories
actor_loss_history = []
kl_loss_history = []
reconstruction_loss_history = []
exploration_loss_history = []

def prefill_dataset(env: SingleEnvironment, save_dir: str, logger_obj: Logger, num_episodes: int, args: Any) -> None:
    """
    Prefill dataset using a random policy.
    """
    from agent.random_explorer import RandomExplorer
    random_policy = RandomExplorer(args, env.action_space)
    random_policy.debug = args.debug  # Enable debug if set in args
    steps, episodes, _, _ = simulate_episode(random_policy, env, {}, save_dir, logger_obj, episodes_num=num_episodes)
    if args.debug:
        print(f"[DEBUG prefill_dataset] Prefilled dataset with {episodes} episodes, {steps} steps", flush=True)

def evaluate_agent(agent: DreamerAgent, env: SingleEnvironment, num_episodes: int = 5, render: bool = False) -> float:
    """
    Evaluate agent using a deterministic policy.
    Returns average reward over episodes.
    """
    total_reward = 0.0
    agent.eval()  # Switch to evaluation mode
    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset()
            if agent.debug:
                print(f"[DEBUG evaluate_agent] Episode {ep+1}/{num_episodes}: Reset observation shape: "
                      f"{np.array(obs['image']).shape}", flush=True)
                print(f"[DEBUG evaluate_agent] Episode {ep+1}/{num_episodes}: Observation mean = "
                      f"{np.mean(obs['image']):.2f}, std = {np.std(obs['image']):.2f}", flush=True)
            done = False
            ep_reward = 0.0
            state = None
            while not done:
                policy_output, state = agent.forward(obs, [done], state, training=False)
                action = policy_output["action"]
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                    if action.ndim == 2 and action.shape[1] > 1:  # One-hot [B, num_actions]
                        action = np.argmax(action)
                    else:
                        action = action.item()
                obs, reward, done, _ = env.step(action)
                if agent.debug:
                    print(f"[DEBUG evaluate_agent] Episode {ep+1}/{num_episodes}: Step reward = {reward}, done = {done}", flush=True)
                ep_reward += reward
                if render:
                    env.render()
            if agent.debug:
                print(f"[DEBUG evaluate_agent] Episode {ep+1}/{num_episodes}: Cumulative Reward = {ep_reward:.2f}", flush=True)
            total_reward += ep_reward
    agent.train()  # Switch back to training mode
    avg_reward = total_reward / num_episodes
    print(f"[INFO evaluate_agent] Average Reward over {num_episodes} episodes: {avg_reward:.2f}", flush=True)
    return avg_reward

def main(args: Any) -> None:
    if args.debug:
        print("[DEBUG main] Loaded Arguments:", vars(args), flush=True)
    
    set_random_seed(args.random_seed)
    if args.use_deterministic_mode:
        enable_deterministic_mode()
    if not torch.cuda.is_available():
        args.computation_device = "cpu"
        print("[DEBUG main] CUDA unavailable, using CPU", flush=True)

    # Create directories
    log_dir = pathlib.Path(args.log_dir).expanduser()
    train_ep_dir = pathlib.Path(args.training_episode_dir) if args.training_episode_dir else log_dir / "train_episodes"
    eval_ep_dir = pathlib.Path(args.evaluation_episode_dir) if args.evaluation_episode_dir else log_dir / "eval_episodes"
    for directory in (log_dir, train_ep_dir, eval_ep_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Initialize logger with initial step count
    initial_steps = count_episode_steps(train_ep_dir)
    logger_obj = Logger(log_directory=log_dir, global_step=args.action_repeat * initial_steps)
    if args.debug:
        print("[DEBUG main] Logging to:", log_dir, f"with initial steps: {initial_steps}", flush=True)

    # Create environment with RGB rendering
    env_instance = SingleEnvironment(task_name=args.task_name, action_repeat=args.action_repeat, seed=args.random_seed)
    if args.debug:
        print("[DEBUG main] Created environment:", env_instance.identifier, flush=True)
    if args.use_parallel_environments:
        env_instance = Parallel(env_instance, strategy="process")
        if args.debug:
            print("[DEBUG main] Using parallel environments", flush=True)

    # Load or prefill training episodes
    training_episodes = load_episode_data(str(train_ep_dir), limit=args.maximum_dataset_size)
    if len(training_episodes) < args.minimum_episodes_to_start_training:
        if args.debug:
            print(f"[DEBUG main] Found {len(training_episodes)} episodes, less than minimum "
                  f"{args.minimum_episodes_to_start_training}. Prefilling dataset...", flush=True)
        prefill_dataset(env_instance, str(train_ep_dir), logger_obj, num_episodes=args.prefill_episodes, args=args)
        training_episodes = load_episode_data(str(train_ep_dir), limit=args.maximum_dataset_size)

    # Load evaluation episodes (optional)
    evaluation_episodes = load_episode_data(str(eval_ep_dir), limit=args.evaluation_number_of_episodes)
    training_dataset = create_dataset(training_episodes, args)
    evaluation_dataset = create_dataset(evaluation_episodes, args) if evaluation_episodes else None

    # Instantiate agent
    agent = DreamerAgent(
        env_instance.observation_space,
        env_instance.action_space,
        args,
        logger_obj,
        training_dataset
    )
    agent.to(args.computation_device)
    if args.debug:
        print("[DEBUG main] Agent instantiated on device:", args.computation_device, flush=True)

    # Load checkpoint if available
    checkpoint_file = log_dir / "checkpoint_latest.pt"
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=args.computation_device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.reset_pretraining_flag()
        if args.debug:
            print("[DEBUG main] Loaded checkpoint from", checkpoint_file, flush=True)

    simulation_state = None
    iteration = 0
    start_time = time.time()

    # Main training loop
    while agent.current_step < args.total_training_steps:
        metrics = logger_obj.write(fps=True)
        if args.debug:
            print(f"\n[DEBUG main] Iteration {iteration} | Current Step: {agent.current_step}/{args.total_training_steps}", flush=True)
            print(f"[DEBUG main] Metrics: {metrics}", flush=True)

        # Record losses
        actor_loss = metrics.get("actor_loss", 0.0)
        kl_loss = metrics.get("world_model_kl_loss", 0.0)
        reconstruction_loss = metrics.get("world_model_decoder_loss", 0.0)
        exploration_loss = metrics.get("exploration_exploration_loss", 0.0)
        actor_loss_history.append(actor_loss)
        kl_loss_history.append(kl_loss)
        reconstruction_loss_history.append(reconstruction_loss)
        exploration_loss_history.append(exploration_loss)

        for metric_name, metric_value in metrics.items():
            try:
                value = float(np.mean(metric_value)) if isinstance(metric_value, (np.ndarray, torch.Tensor)) else float(metric_value)
                logger_obj.scalar(metric_name, value)
            except (TypeError, ValueError) as e:
                if args.debug:
                    print(f"[DEBUG main] Error logging {metric_name}: {e}", flush=True)

        if args.debug:
            print(f"[DEBUG main] Actor Loss: {actor_loss:.4f}, KL Loss: {kl_loss:.4f}, "
                  f"Reconstruction Loss: {reconstruction_loss:.4f}, Exploration Loss: {exploration_loss:.4f}", flush=True)

        # Periodic evaluation
        if agent.current_step > 0 and agent.current_step % args.evaluation_interval_steps == 0:
            print("\n[INFO main] ===== Evaluating agent =====", flush=True)
            avg_eval_reward = evaluate_agent(agent, env_instance,
                                             num_episodes=args.evaluation_number_of_episodes,
                                             render=args.render_during_evaluation)
            logger_obj.scalar("evaluation/average_reward", avg_eval_reward)
            if args.log_video_predictions:
                batch = next(evaluation_dataset) if evaluation_dataset else next(training_dataset)
                video = agent.world_model.generate_video(batch)
                logger_obj.video("predictions/video", video)
            logger_obj.write(fps=True)
            elapsed = time.time() - start_time
            print(f"[INFO main] [Evaluation] Global step: {agent.current_step} | Elapsed: {elapsed/60:.2f} min\n", flush=True)

        # Collect data and train
        if args.debug:
            print("[DEBUG main] [Training] Simulating episode(s) to collect data...", flush=True)
        steps_taken, episodes_completed, done, simulation_state = simulate_episode(
            agent, env_instance, training_episodes,
            str(train_ep_dir), logger_obj,
            steps=args.steps_per_iteration,
            state=simulation_state
        )
        if args.debug:
            print(f"[DEBUG main] simulate_episode returned: steps={steps_taken}, episodes={episodes_completed}, done={done}", flush=True)
        agent.current_step += steps_taken * args.action_repeat  # Account for action repeat
        logger_obj.global_step = agent.current_step

        # Train the agent
        for _ in range(args.training_updates_per_forward):
            agent.train_step(next(training_dataset))

        # Save checkpoint
        checkpoint_data = {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state": agent.collect_optimizer_states(),
            "current_step": agent.current_step
        }
        torch.save(checkpoint_data, checkpoint_file)
        if args.debug:
            print(f"[DEBUG main] Checkpoint saved at step {agent.current_step}", flush=True)
        iteration += 1

    total_time = time.time() - start_time
    print(f"\n[INFO main] Training completed in {total_time/60:.2f} minutes over {iteration} iterations.", flush=True)

    # Plot training loss history
    plt.figure(figsize=(12, 6))
    plt.plot(actor_loss_history, label="Actor Loss")
    plt.plot(kl_loss_history, label="KL Loss")
    plt.plot(reconstruction_loss_history, label="Reconstruction Loss")
    plt.plot(exploration_loss_history, label="Exploration Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True)
    plt.savefig(log_dir / "training_loss_history.png")
    plt.show()

    try:
        env_instance.close()
    except Exception as e:
        if args.debug:
            print(f"[DEBUG main] Error closing environment: {e}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreamer-V3 Training for Single Environment")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--training_episode_dir", type=str, default="", help="Directory for training episodes.")
    parser.add_argument("--evaluation_episode_dir", type=str, default="", help="Directory for evaluation episodes.")
    parser.add_argument("--task_name", type=str, default="CartPole-v1", help="Task identifier for the environment.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_deterministic_mode", action="store_true", help="Enable deterministic execution.")
    parser.add_argument("--action_repeat", type=int, default=2, help="Number of times each action is repeated.")
    parser.add_argument("--number_of_environments", type=int, default=1, help="Number of environments.")
    parser.add_argument("--maximum_dataset_size", type=int, default=10000, help="Maximum dataset size.")
    parser.add_argument("--total_training_steps", type=int, default=1000000, help="Total training steps.")
    parser.add_argument("--evaluation_interval_steps", type=int, default=10000, help="Evaluation interval (steps).")
    parser.add_argument("--evaluation_number_of_episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument("--use_parallel_environments", action="store_true", help="Enable parallel execution.")
    parser.add_argument("--computation_device", type=str, default="cuda", help="Device (cuda or cpu).")
    parser.add_argument("--log_video_predictions", action="store_true", help="Log video predictions to TensorBoard.")
    parser.add_argument("--sequence_length", type=int, default=50, help="Length of sequences used in training.")
    parser.add_argument("--training_updates_per_forward", type=int, default=1, help="Number of training updates per forward call.")
    parser.add_argument("--steps_per_iteration", type=int, default=1000, help="Steps to collect per iteration.")
    parser.add_argument("--prefill_episodes", type=int, default=10, help="Number of episodes to prefill.")
    parser.add_argument("--minimum_episodes_to_start_training", type=int, default=2, help="Minimum episodes to start training.")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints.")
    parser.add_argument("--render_during_evaluation", action="store_true", help="Render environment during evaluation.")

    args = parser.parse_args()

    # Load YAML configuration
    yaml_loader = YAML(typ='safe', pure=True)
    config_file_path = pathlib.Path(__file__).parent / "config" / "configuration.yaml"
    with config_file_path.open("r") as config_file:
        config_data = yaml_loader.load(config_file)
        if args.debug:
            print("[DEBUG main] Loaded configuration from", config_file_path, flush=True)
    for key, value in config_data.get("default", {}).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    main(args)