"""
Main entry point for Dreamer-V3 training.
Loads configuration, sets up the environment, dataset, logger, and agent,
runs a data collection (prefill) phase if needed, prints intermediate debug metrics,
and at the end plots the loss history.
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

# Use os.getuid() if available; otherwise, use a fallback value.
uid = os.getuid() if hasattr(os, "getuid") else "win"
os.environ["XDG_RUNTIME_DIR"] = f"/tmp/runtime-{uid}"

from agent.dreamer_agent import DreamerAgent
from env.single_environment import SingleEnvironment
from utils.dataset import create_dataset, load_episode_data, simulate_episode
from utils.logger import Logger
from utils.helper_functions import set_random_seed, enable_deterministic_mode, count_episode_steps
from parallel.parallel import Parallel

# Global lists to store loss metrics over iterations.
actor_loss_history = []
kl_loss_history = []
reconstruction_loss_history = []
exploration_loss_history = []

def prefill_dataset(environment: SingleEnvironment, save_directory: str, logger_obj: Logger, num_episodes: int, args: Any) -> None:
    """
    Run a data collection phase to prefill the training dataset using a random policy.
    """
    from agent.random_explorer import RandomExplorer
    random_policy = RandomExplorer(args, environment.action_space)
    for ep in range(num_episodes):
        print(f"Collecting prefill episode {ep+1}/{num_episodes}", flush=True)
        simulate_episode(random_policy, environment, {}, save_directory, logger_obj, episodes_num=1)

def evaluate_agent(agent: DreamerAgent, env: SingleEnvironment, num_episodes: int = 5, render: bool = False) -> float:
    """
    Run evaluation episodes using a deterministic policy (using the actor's mode).
    Logs cumulative reward for each episode and returns the average reward.
    Also logs simple image statistics (mean and std) to verify diversity.
    """
    total_reward = 0.0
    for ep in range(num_episodes):
        obs = env.reset()
        # Log observation statistics to ensure images are diverse.
        img = obs["image"]
        print(f"DEBUG: Evaluation observation mean: {np.mean(img):.2f}, std: {np.std(img):.2f}", flush=True)
        done = False
        ep_reward = 0.0
        while not done:
            # Use a deterministic policy (mode of the actor distribution).
            policy_output, _ = agent.forward(obs, reset_flags=[True], training=False)
            action = policy_output["action"]
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        print(f"[Evaluation] Episode {ep+1}: Cumulative Reward = {ep_reward}", flush=True)
        total_reward += ep_reward
    avg_reward = total_reward / num_episodes
    print(f"[Evaluation] Average Reward over {num_episodes} episodes: {avg_reward}", flush=True)
    return avg_reward

def main(args: Any) -> None:
    set_random_seed(args.random_seed)
    if args.use_deterministic_mode:
        enable_deterministic_mode()

    if not torch.cuda.is_available():
        args.computation_device = "cpu"

    log_directory = pathlib.Path(args.log_dir).expanduser()
    training_episode_directory = pathlib.Path(args.training_episode_dir) if args.training_episode_dir else log_directory / "train_episodes"
    evaluation_episode_directory = pathlib.Path(args.evaluation_episode_dir) if args.evaluation_episode_dir else log_directory / "eval_episodes"
    for directory in (log_directory, training_episode_directory, evaluation_episode_directory):
        directory.mkdir(parents=True, exist_ok=True)

    initial_steps = count_episode_steps(training_episode_directory)
    logger_obj = Logger(log_directory=log_directory, global_step=args.action_repeat * initial_steps)
    print("Logging to:", log_directory, flush=True)

    # Create the environment.
    environment_instance = SingleEnvironment(task_name=args.task_name, action_repeat=args.action_repeat, seed=args.random_seed)
    if args.use_parallel_environments:
        environment_instance = Parallel(environment_instance, strategy="process")

    training_episodes = load_episode_data(str(training_episode_directory), limit=args.maximum_dataset_size)
    if len(training_episodes) == 0:
        print("No training episodes found. Prefilling dataset with random policy...", flush=True)
        prefill_dataset(environment_instance, str(training_episode_directory), logger_obj, num_episodes=10, args=args)
        training_episodes = load_episode_data(str(training_episode_directory), limit=args.maximum_dataset_size)

    evaluation_episodes = load_episode_data(str(evaluation_episode_directory), limit=1)
    training_dataset = create_dataset(training_episodes, args)
    evaluation_dataset = create_dataset(evaluation_episodes, args)

    # Instantiate the agent.
    agent = DreamerAgent(environment_instance.observation_space,
                           environment_instance.action_space,
                           args, logger_obj, training_dataset)
    agent.to(args.computation_device)

    checkpoint_file = log_directory / "checkpoint_latest.pt"
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        agent.reset_pretraining_flag()
        print("Loaded checkpoint from", checkpoint_file, flush=True)

    simulation_state = None
    iteration = 0
    start_time = time.time()

    # Main training loop.
    while agent.current_step < args.total_training_steps + args.evaluation_interval_steps:
        metrics = logger_obj.write(fps=True)
        print(f"[Iteration {iteration}] Metrics: {metrics}", flush=True)
        
        # Record loss metrics.
        actor_loss = metrics.get("actor_loss", 0.0)
        kl_loss = metrics.get("kl_loss", 0.0)
        reconstruction_loss = metrics.get("reconstruction_loss", 0.0)
        exploration_loss = metrics.get("exploration_exploration_loss", 0.0)
        actor_loss_history.append(actor_loss)
        kl_loss_history.append(kl_loss)
        reconstruction_loss_history.append(reconstruction_loss)
        exploration_loss_history.append(exploration_loss)
        
        # Convert GPU tensors to CPU before taking the mean and logging.
        for metric_name, metric_values in metrics.items():
            try:
                if isinstance(metric_values, torch.Tensor):
                    value = float(np.mean(metric_values.cpu().numpy()))
                else:
                    value = float(np.mean(metric_values))
                logger_obj.scalar(metric_name, value)
            except Exception as e:
                print(f"Error logging {metric_name}: {e}")
        
        print(f"[Iteration {iteration}] Actor Loss: {actor_loss:.4f}, KL Loss: {kl_loss:.4f}, "
              f"Reconstruction Loss: {reconstruction_loss:.4f}, Exploration Loss: {exploration_loss:.4f}", flush=True)
        
        # Periodically evaluate the agent.
        if args.evaluation_number_of_episodes > 0 and agent.current_step % args.evaluation_interval_steps == 0:
            print("Evaluating agent...", flush=True)
            avg_eval_reward = evaluate_agent(agent, environment_instance, num_episodes=args.evaluation_number_of_episodes, render=False)
            logger_obj.scalar("evaluation/average_reward", avg_eval_reward)
            logger_obj.write(fps=True)
            elapsed = time.time() - start_time
            print(f"Global step: {agent.current_step} | Elapsed time: {elapsed/60:.2f} minutes", flush=True)
        
        print("Starting training phase...", flush=True)
        simulation_state = simulate_episode(agent, environment_instance, training_episodes,
                                            str(training_episode_directory), logger_obj,
                                            steps=args.evaluation_interval_steps,
                                            state=simulation_state)
        checkpoint_data = {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state": agent.collect_optimizer_states()
        }
        torch.save(checkpoint_data, checkpoint_file)
        iteration += 1

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds over {iteration} iterations.", flush=True)

    # Plot loss histories at the end.
    plt.figure(figsize=(10, 5))
    plt.plot(actor_loss_history, label="Actor Loss")
    plt.plot(kl_loss_history, label="KL Loss")
    plt.plot(reconstruction_loss_history, label="Reconstruction Loss")
    plt.plot(exploration_loss_history, label="Exploration Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.legend()
    plt.savefig("training_loss_history.png")
    plt.show()

    try:
        environment_instance.close()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dreamer-V3 Training for Single Environment")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--training_episode_dir", type=str, default="", help="Directory for training episodes.")
    parser.add_argument("--evaluation_episode_dir", type=str, default="", help="Directory for evaluation episodes.")
    parser.add_argument("--task_name", type=str, default="cartpole", help="Task identifier for the environment.")
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

    args = parser.parse_args()

    # Load configuration from YAML.
    yaml_loader = YAML(typ='safe', pure=True)
    config_file_path = pathlib.Path(__file__).parent / "config" / "configuration.yaml"
    with config_file_path.open("r") as config_file:
        config_data = yaml_loader.load(config_file)
    for key, value in config_data.get("default", {}).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    main(args)
