import torch
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, List
import os
import logging
import argparse
import pathlib
import sys
import importlib
import inspect
import traceback

from agent.dreamer_agent import DreamerAgent
from utils.dataset import create_dataset, load_episode_data, simulate_episode, save_episode
from env.single_environment import SingleEnvironment
from utils.logger import Logger
from utils.optimizer import Optimizer
from utils.helper_functions import set_random_seed, enable_deterministic_mode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_imports() -> None:
    """
    Check for import issues in the codebase to ensure all modules are available.
    """
    print("Checking imports...")
    
    modules_to_check = [
        'agent.dreamer_agent',
        'agent.world_model',
        'agent.behavior',
        'agent.random_explorer',
        'agent.networks',
        'env.single_environment',
        'utils.dataset',
        'utils.optimizer',
        'utils.helper_functions',
        'utils.logger'
    ]
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"+ Successfully imported {module_name}")
        except Exception as error:
            print(f"- Error importing {module_name}: {error}")
            print(f"  - Error type: {type(error).__name__}")
            if isinstance(error, ImportError):
                parts = module_name.split('.')
                if len(parts) > 1:
                    parent_module = '.'.join(parts[:-1])
                    try:
                        parent = importlib.import_module(parent_module)
                        print(f"  - Contents of {parent_module}: {dir(parent)}")
                    except ImportError:
                        print(f"  - Could not import parent module {parent_module}")
    
    print("Import check complete")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="DreamerV3 Training")
    parser.add_argument("--config", type=str, default="config/configuration.yaml", 
                        help="Path to configuration YAML file")
    parser.add_argument("--log_directory", type=str, default="logs", 
                        help="Directory for logs and checkpoints")
    parser.add_argument("--task_name", type=str, default="CartPole-v1", 
                        help="Task identifier for the environment")
    parser.add_argument("--render_during_evaluation", action="store_true", 
                        help="Render environment during evaluation")
    parser.add_argument("--enable_debugging", action="store_true", 
                        help="Enable debug output")
    parser.add_argument("--checkpoint", type=str, default="", 
                        help="Path to checkpoint file to load")
    parser.add_argument("--total_training_steps", type=int, default=0,
                        help="Total training steps (0 to use config default)")
    parser.add_argument("--train_ratio", type=int, default=0,
                        help="Training updates per environment step (0 to use config default)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size (0 to use config default)")
    return parser.parse_args()

def load_yaml_configuration(config_path: str) -> Dict:
    """
    Load configuration from a YAML file and ensure numeric values are correctly typed.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        return {}
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        numeric_parameters = [
            'model_learning_rate', 'actor_lr', 'critic_lr', 
            'optimizer_epsilon', 'gradient_clip_value', 'weight_decay_value',
            'actor_eps', 'critic_eps', 'actor_grad_clip', 'critic_grad_clip',
            'actor_entropy', 'actor_temperature', 'actor_unimix_ratio',
            'critic_slow_target_update_fraction', 'discount_factor',
            'discount_lambda', 'dynamics_loss_scale', 'representation_loss_scale',
            'kl_free', 'decoder_loss_scale', 'reward_head_loss_scale',
            'continuation_head_loss_scale', 'unimix_ratio', 'return_ema_alpha',
            'reward_ema_alpha', 'train_ratio', 'sequence_length', 'batch_size'
        ]
        
        for param in numeric_parameters:
            if param in config and config[param] is not None:
                try:
                    if isinstance(config[param], str):
                        if '.' in config[param] or 'e' in config[param].lower():
                            config[param] = float(config[param])
                        else:
                            config[param] = int(config[param])
                    elif isinstance(config[param], (int, float)):
                        if param in ['model_learning_rate', 'actor_lr', 'critic_lr',
                                    'actor_entropy', 'actor_temperature', 'actor_unimix_ratio',
                                    'discount_factor', 'discount_lambda', 'return_ema_alpha']:
                            config[param] = float(config[param])
                except ValueError:
                    logger.warning(f"Could not convert {param}={config[param]} to numeric type. Using default.")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as error:
        logger.error(f"Error loading configuration file: {error}")
        return {}

def count_episode_steps(directory: str) -> int:
    """
    Count total steps across all episodes in a directory.
    
    Args:
        directory: Path to the episode directory
        
    Returns:
        Total number of steps
    """
    directory_path = pathlib.Path(directory)
    total_steps = 0
    
    for file in directory_path.glob("*.npz"):
        try:
            data = np.load(file)
            total_steps += len(data["reward"])
        except Exception:
            continue
    return total_steps

def load_checkpoint(agent: DreamerAgent, logger_object: Logger, path: pathlib.Path, 
                    device: str, enable_debugging: bool = False) -> bool:
    if not path.exists():
        if enable_debugging:
            logger.debug(f"No checkpoint found at {path}")
        return False
        
    try:
        checkpoint = torch.load(path, map_location=device)
        agent_state_dict = checkpoint["agent_state_dict"]
        
        # Filter out incompatible keys
        current_state_dict = agent.state_dict()
        compatible_state_dict = {k: v for k, v in agent_state_dict.items() if k in current_state_dict and v.shape == current_state_dict[k].shape}
        if len(compatible_state_dict) < len(current_state_dict):
            logger.warning(f"Checkpoint contains {len(agent_state_dict)} keys, but only {len(compatible_state_dict)} match current model. Training from scratch.")
            return False
        
        agent.load_state_dict(compatible_state_dict, strict=False)
        
        if "current_step" in checkpoint:
            agent.current_step = checkpoint["current_step"]
            logger_object.global_step = agent.current_step * getattr(agent.configuration, "action_repeat", 1)
        
        if "optimizer_state" in checkpoint and checkpoint["optimizer_state"]:
            if hasattr(agent.world_model, 'model_optimizer'):
                agent.world_model.model_optimizer.load_state_dict(checkpoint["optimizer_state"].get("world_model", {}))
            if hasattr(agent.task_behavior, 'actor_optimizer'):
                agent.task_behavior.actor_optimizer.load_state_dict(checkpoint["optimizer_state"].get("actor", {}))
            if hasattr(agent.task_behavior, 'value_optimizer'):
                agent.task_behavior.value_optimizer.load_state_dict(checkpoint["optimizer_state"].get("value", {}))
        
        logger.info(f"Loaded compatible parts of checkpoint from {path} at step {agent.current_step}")
        return True
    except Exception as error:
        logger.error(f"Error loading checkpoint: {error}")
        return False

def save_checkpoint(agent: DreamerAgent, path: pathlib.Path, enable_debugging: bool = False) -> None:
    """
    Save an agent checkpoint to disk.
    
    Args:
        agent: DreamerAgent instance to save
        path: Path to save the checkpoint file
        enable_debugging: Whether to log debug information
    """
    try:
        checkpoint_data = {
            "agent_state_dict": agent.state_dict(),
            "current_step": agent.current_step,
            "optimizer_state": agent.collect_optimizer_states()
        }
        
        torch.save(checkpoint_data, path)
        if enable_debugging:
            logger.debug(f"Checkpoint saved to {path}")
    except Exception as error:
        logger.error(f"Error saving checkpoint: {error}")

def plot_training_history(loss_history: Dict[str, List], save_path: pathlib.Path, enable_debugging: bool = False) -> None:
    """
    Plot training loss history and save to a file.
    
    Args:
        loss_history: Dictionary of loss metrics over time
        save_path: Path to save the plot
        enable_debugging: Whether to log debug information
    """
    try:
        plt.figure(figsize=(12, 8))
        plot_groups = {
            "World Model Losses": ["world_model_loss", "decoder_loss", "kl_loss", "reward_loss", "continuation_loss"],
            "Actor-Critic Losses": ["actor_loss", "value_loss", "policy_entropy"],
            "Gradient Statistics": ["actor_gradient_average", "value_gradient_average", "world_model_gradient_average"]
        }
        fig, axes = plt.subplots(len(plot_groups), 1, figsize=(12, 4 * len(plot_groups)), sharex=True)
        for i, (group_name, metrics_list) in enumerate(plot_groups.items()):
            ax = axes[i] if len(plot_groups) > 1 else axes
            for name in metrics_list:
                if name in loss_history and loss_history[name]:
                    values = loss_history[name]
                    window_size = min(len(values) // 10 + 1, 100)
                    if window_size > 1 and len(values) > window_size:
                        smoothed = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
                        ax.plot(smoothed, label=name)
                    else:
                        ax.plot(values, label=name)
            ax.set_title(group_name)
            ax.set_ylabel("Value")
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Training Steps") if len(plot_groups) > 1 else axes.set_xlabel("Training Steps")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        if enable_debugging:
            logger.debug(f"Loss history plot saved to {save_path}")
    except Exception as error:
        logger.error(f"Error plotting training history: {error}")

def evaluate_agent(agent: DreamerAgent, environment: SingleEnvironment, logger_object: Logger, 
                   number_of_episodes: int = 10, render_during_evaluation: bool = False) -> float:
    logger.info(f"Evaluating agent over {number_of_episodes} episodes...")
    total_reward = 0.0
    episode_lengths = []
    all_rewards = []
    
    agent.eval()
    
    with torch.no_grad():
        for episode in range(number_of_episodes):
            observation = environment.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            state = None
            
            while not done:
                policy_output, state = agent(observation, state, training=False)
                action = policy_output["action"]
                if action.ndim > 1:
                    action = action[0]
                
                next_observation, reward, done, info = environment.step(action)
                if render_during_evaluation:
                    environment.render()
                
                episode_reward += reward
                step_count += 1
                observation = next_observation
                
                if step_count >= environment.max_episode_steps:
                    break
            
            total_reward += episode_reward
            episode_lengths.append(step_count)
            all_rewards.append(episode_reward)
            
            logger_object.log_scalar(f"evaluation/episode_{episode + 1}_reward", episode_reward)
            logger_object.log_scalar(f"evaluation/episode_{episode + 1}_length", step_count)
            logger_object.log_episode_return(episode_reward)
            
            logger.info(f"Evaluation episode {episode + 1}/{number_of_episodes}: reward={episode_reward:.2f}, length={step_count}")
    
    average_reward = total_reward / number_of_episodes
    average_length = sum(episode_lengths) / len(episode_lengths)
    reward_std = np.std(all_rewards)
    
    logger_object.log_scalar("evaluation/average_reward", average_reward)
    logger_object.log_scalar("evaluation/average_length", average_length)
    logger_object.log_scalar("evaluation/reward_standard_deviation", reward_std)
    logger_object.write()
    
    agent.train()
    logger.info(f"Evaluation results: Avg Reward={average_reward:.2f} ± {reward_std:.2f}, Avg Length={average_length:.2f}")
    
    return average_reward

def print_loss_status(iteration: int, metrics: Dict[str, float], previous_values: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Print formatted loss values with change indicators.
    
    Args:
        iteration: Current training iteration
        metrics: Dictionary of current metrics
        previous_values: Dictionary of previous metric values for comparison
    
    Returns:
        Updated previous_values dictionary
    """
    if previous_values is None:
        previous_values = {}
    
    key_metrics = [
        "world_model_loss", "actor_loss", "value_loss", "kl_loss",
        "reward_loss", "decoder_loss", "continuation_loss", 
        "policy_entropy", "return_scale"
    ]
    
    available_metrics = [key for key in key_metrics if key in metrics]
    
    if not available_metrics:
        print(f"\033[93m[Iter {iteration}] No metrics available\033[0m")
        return previous_values
    
    parts = []
    for key in available_metrics:
        value = metrics[key]
        prev_value = previous_values.get(key)
        
        if prev_value is not None:
            diff = abs(value - prev_value)
            if diff > 1e-6:
                is_better = value > prev_value if key == "policy_entropy" else value < prev_value
                change_str = f"\033[92m{value:.4f} ↓\033[0m" if is_better else f"\033[91m{value:.4f} ↑\033[0m"
            else:
                change_str = f"{value:.4f} ="
        else:
            change_str = f"{value:.4f}"
        
        parts.append(f"{key}: {change_str}")
        previous_values[key] = value
    
    display_string = " | ".join(parts)
    indicator = "\033[94m[LEARNING ACTIVE]\033[0m" if any("↓" in s or "↑" in s for s in parts) else "\033[93m[STABLE]\033[0m"
    print(f"\033[1m[Iter {iteration}] {indicator}\033[0m {display_string}")
    return previous_values

def visualize_gradients(agent: DreamerAgent, iteration: int, log_directory: pathlib.Path) -> None:
    """
    Create a gradient flow visualization for debugging gradient issues.
    
    Args:
        agent: DreamerAgent instance
        iteration: Current training iteration
        log_directory: Directory to save the visualization
    """
    try:
        plt.figure(figsize=(12, 8))
        gradient_means = []
        layer_names = []
        
        for name, module in [('world_model', agent.world_model), ('task_behavior', agent.task_behavior)]:
            for param_name, param in module.named_parameters():
                if param.grad is not None:
                    gradient_means.append(param.grad.abs().mean().item())
                    layer_names.append(f"{name}.{param_name}")
        
        if not gradient_means:
            logger.warning("No gradients found for visualization")
            return
        
        plt.barh(np.arange(len(gradient_means)), gradient_means)
        plt.yticks(np.arange(len(gradient_means)), layer_names)
        plt.xlabel('Mean Absolute Gradient')
        plt.title(f'Gradient Flow at Iteration {iteration}')
        plt.grid(True, alpha=0.3)
        
        for i, value in enumerate(gradient_means):
            if value > 0:
                plt.text(value, i, f"{value:.1e}", va='center')
        
        plt.tight_layout()
        plt.savefig(log_directory / f"gradient_flow_{iteration}.png")
        plt.close()
        
        logger.info(f"Gradient visualization saved for iteration {iteration}")
    except Exception as error:
        logger.error(f"Error creating gradient visualization: {error}")

def sanity_check_model(agent: DreamerAgent, environment: SingleEnvironment) -> bool:
    logger.info("Running model sanity check...")
    try:
        batch_size = 4
        sequence_length = 10
        
        batch = {
            "image": torch.rand(batch_size, sequence_length, 3, 64, 64, device=agent.device),
            "action": torch.randint(0, environment.action_space.n, (batch_size, sequence_length), device=agent.device),  # Long tensor
            "reward": torch.rand(batch_size, sequence_length, device=agent.device),
            "is_first": torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=agent.device),
            "is_terminal": torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=agent.device),
            "is_last": torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=agent.device),
            "discount": torch.ones(batch_size, sequence_length, device=agent.device),
            "continuation": torch.ones(batch_size, sequence_length, device=agent.device)
        }
        batch["is_first"][:, 0] = True
        batch["is_last"][:, -1] = True
        batch["is_terminal"][:, -1] = True
        batch["discount"][:, -1] = 0.0
        batch["continuation"][:, -1] = 0.0
        
        metrics1 = agent.train_step(batch)
        metrics2 = agent.train_step(batch)
        
        loss_keys = [key for key in metrics1.keys() if 'loss' in key]
        changed = False
        
        for key in loss_keys:
            if key in metrics1 and key in metrics2:
                diff = abs(metrics2[key] - metrics1[key])
                if diff > 1e-5:
                    changed = True
                    logger.info(f"Loss {key} changed by {diff:.6f}")
        
        if not changed:
            logger.error("Sanity check failed: Losses did not change between steps!")
            return False
        
        logger.info("Sanity check passed")
        return True
    except Exception as error:
        logger.error(f"Sanity check failed: {error}")
        return False

def check_world_model_dimensions(config_namespace: argparse.Namespace) -> Dict[str, int]:
    """
    Verify consistency of world model dimensions.
    
    Args:
        config_namespace: Configuration namespace object
        
    Returns:
        Dictionary of key dimensions
    """
    print("\n=== World Model Dimension Check ===")
    
    discrete_latent_num = getattr(config_namespace, "discrete_latent_num", 32)
    discrete_latent_size = getattr(config_namespace, "discrete_latent_size", 32)
    stochastic_dimension = discrete_latent_num * discrete_latent_size
    deterministic_dimension = getattr(config_namespace, "dynamics_deterministic_dimension", 4096)
    feature_dimension = deterministic_dimension + stochastic_dimension
    image_size = getattr(config_namespace, "image_size", 64)
    image_channels = 3
    flat_image_dimension = image_channels * image_size * image_size
    encoder_output_dimension = getattr(config_namespace, "encoder_output_dimension", 1024)
    units = getattr(config_namespace, "units", 512)
    
    config_namespace.feature_dimension = feature_dimension
    
    print(f"Discrete latent: {discrete_latent_num} vars × {discrete_latent_size} size = {stochastic_dimension}")
    print(f"Deterministic dimension: {deterministic_dimension}")
    print(f"Feature dimension: {feature_dimension}")
    print(f"Image: {image_channels}×{image_size}×{image_size} = {flat_image_dimension}")
    print(f"Encoder output dimension: {encoder_output_dimension}")
    print(f"Hidden units: {units}")
    print("=== Dimension check complete ===\n")
    
    return {
        "stochastic_dimension": stochastic_dimension,
        "deterministic_dimension": deterministic_dimension,
        "feature_dimension": feature_dimension,
        "flat_image_dimension": flat_image_dimension
    }

def main() -> None:
    """
    Main entry point for DreamerV3 training.
    Handles configuration, environment setup, agent training, evaluation, and logging.
    """
    args = parse_arguments()
    config = load_yaml_configuration(args.config)
    config["log_directory"] = args.log_directory
    config["task_name"] = args.task_name
    config["render_during_evaluation"] = args.render_during_evaluation
    config["enable_debugging"] = args.enable_debugging
    if args.total_training_steps > 0:
        config["total_training_steps"] = args.total_training_steps
    if args.train_ratio > 0:
        config["train_ratio"] = args.train_ratio
    if args.batch_size > 0:
        config["batch_size"] = args.batch_size
    
    required_parameters = [
        ('model_learning_rate', 1e-4), ('actor_lr', 3e-5), ('critic_lr', 3e-5),
        ('optimizer_epsilon', 1e-8), ('gradient_clip_value', 1000.0),
        ('actor_grad_clip', 1000.0), ('critic_grad_clip', 1000.0),
        ('weight_decay_value', 0.0), ('train_ratio', 512),
        ('sequence_length', 50), ('batch_size', 16)
    ]
    
    for param, default_value in required_parameters:
        if param not in config or not isinstance(config[param], (int, float)):
            config[param] = default_value
    
    config_namespace = argparse.Namespace(**config)
    check_world_model_dimensions(config_namespace)
    
    logging.basicConfig(
        level=logging.DEBUG if config.get("enable_debugging", False) else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(os.path.join(args.log_directory, "dreamer.log")), logging.StreamHandler()]
    )
    
    set_random_seed(config.get("random_seed", 42))
    if config.get("use_deterministic_mode", True):
        enable_deterministic_mode()
    
    device = "cuda" if torch.cuda.is_available() and config.get("computation_device", "cuda") == "cuda" else "cpu"
    config["computation_device"] = device
    
    log_directory = pathlib.Path(config["log_directory"]).expanduser()
    training_episode_directory = pathlib.Path(config.get("training_episode_directory", "episodes/train")).expanduser()
    evaluation_episode_directory = pathlib.Path(config.get("evaluation_episode_directory", "episodes/eval")).expanduser()
    
    for directory in (log_directory, training_episode_directory, evaluation_episode_directory):
        directory.mkdir(parents=True, exist_ok=True)
    
    initial_steps = count_episode_steps(training_episode_directory) * config.get("action_repeat", 1)
    logger_object = Logger(log_directory=log_directory, global_step=initial_steps)
    logger_object.log_hyperparameters(config)
    logger_object.configuration = config_namespace
    
    try:
        environment_instance = SingleEnvironment(
            task_name=config["task_name"],
            action_repeat=config.get("action_repeat", 1),
            seed=config.get("random_seed", 42),
            image_size=config.get("image_size", 64),
            enable_debugging=config.get("enable_debugging", False)
        )
    except Exception as error:
        logger.error(f"Failed to create environment: {error}")
        raise
    
    logger.info(f"Environment: {config['task_name']}")
    logger.info(f"Observation space: {environment_instance.observation_space}")
    logger.info(f"Action space: {environment_instance.action_space}")
    
    training_episodes = load_episode_data(str(training_episode_directory), limit=config.get("buffer_capacity", 1000000))
    minimum_steps = config.get("minimum_steps_to_start_training", 1000)
    total_steps = sum(len(ep["reward"]) for ep in training_episodes.values() if "reward" in ep)
    
    if total_steps < minimum_steps:
        logger.info(f"Found {total_steps} steps, need at least {minimum_steps}")
        from agent.random_explorer import RandomExplorer
        try:
            random_explorer = RandomExplorer(config_namespace, environment_instance.action_space)
        except Exception as error:
            logger.error(f"Failed to create random explorer: {error}")
            raise
        
        prefill_steps = config.get("prefill_steps", minimum_steps - total_steps)
        logger.info(f"Prefilling to reach at least {prefill_steps} random steps...")
        
        total_steps_taken, total_episodes_completed, _, _ = simulate_episode(
            random_explorer, 
            environment_instance, 
            num_episodes=max(1, prefill_steps // 100), 
            directory=str(training_episode_directory)  # Pass training_episode_directory explicitly
        )
        training_episodes = load_episode_data(str(training_episode_directory), limit=config.get("buffer_capacity", 1000000))
        total_steps = sum(len(ep["reward"]) for ep in training_episodes.values() if "reward" in ep)
        
        logger.info(f"Prefilled with {total_steps_taken} steps in {total_episodes_completed} episodes")
        if total_steps < minimum_steps:
            logger.warning(f"Prefill only reached {total_steps} steps, less than {minimum_steps}")
    
    logger.info(f"Loaded {total_steps} training steps into dataset")
    
    training_dataset = create_dataset(training_episodes, config_namespace)
    
    try:
        agent = DreamerAgent(
            observation_space=environment_instance.observation_space,
            action_space=environment_instance.action_space,
            configuration=config_namespace,
            logger_object=logger_object,
            dataset=training_dataset
        )
        agent.to(device)
    except Exception as error:
        logger.error(f"Failed to create DreamerAgent: {error}")
        raise
    
    logger_object.log_model_parameters(agent)
    
    if not sanity_check_model(agent, environment_instance):
        logger.warning("Model failed sanity check")
    
    checkpoint_file = log_directory / "checkpoint_latest.pt"
    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint_file = checkpoint_path
    
    if load_checkpoint(agent, logger_object, checkpoint_file, device, config.get("enable_debugging", False)):
        logger.info(f"Resumed training from step {agent.current_step}")
    else:
        logger.info("Starting training from scratch")
    
    simulation_state = None
    iteration = 0
    start_time = time.time()
    best_checkpoint_file = log_directory / "checkpoint_best.pt"
    best_evaluation_reward = float('-inf')
    
    loss_history = {
        "actor_loss": [], "value_loss": [], "kl_loss": [],
        "decoder_loss": [], "reward_loss": [], "world_model_loss": [],
        "policy_entropy": [], "actor_gradient_average": [], "value_gradient_average": [],
        "world_model_gradient_average": []
    }
    
    previous_losses = {}
    total_training_steps = config["total_training_steps"]
    steps_per_iteration = config.get("steps_per_iteration", 1)
    loss_print_interval = config.get("loss_print_interval", 10)  # Reduced for more frequent logging
    train_ratio = config["train_ratio"]
    
    try:
        while agent.current_step < total_training_steps:
            iteration_start_time = time.time()
            
            steps_taken, episodes_completed, done, simulation_state = simulate_episode(
                agent, 
                environment_instance, 
                steps_per_iteration, 
                directory=str(training_episode_directory)
            )
            environment_steps = steps_taken * config.get("action_repeat", 1)
            agent.current_step += environment_steps
            logger_object.global_step = agent.current_step
            
            total_updates = steps_taken * train_ratio
            update_metrics_sum = {}
            metrics_count = 0
            
            for _ in range(total_updates):
                try:
                    batch = next(training_dataset)
                    batch_metrics = agent.train_step(batch)
                    if batch_metrics:
                        metrics_count += 1
                        for key, value in batch_metrics.items():
                            update_metrics_sum[key] = update_metrics_sum.get(key, 0.0) + float(value)
                except Exception as error:
                    logger.warning(f"Training update failed: {error}")
            
            if metrics_count > 0:
                update_metrics_average = {key: value / metrics_count for key, value in update_metrics_sum.items()}
                logger_object.log_metrics(update_metrics_average, compute_fps=True, debug=config.get("enable_debugging", False))
                
                # Enhanced logging of metrics
                logger.info(f"Iteration {iteration} - Averaged Metrics: {update_metrics_average}")
                
                if all(update_metrics_average.get(k, 0.0) == 0.0 for k in ["actor_loss", "value_loss", "world_model_loss", "kl_loss"]):
                    logger.warning("All losses are zero!")
                    visualize_gradients(agent, iteration, log_directory)
                
                previous_losses = print_loss_status(iteration, update_metrics_average, previous_losses)
                for key in loss_history:
                    if key in update_metrics_average:
                        loss_history[key].append(update_metrics_average[key])
            
            if iteration % 100 == 0 or iteration == 0:
                save_checkpoint(agent, checkpoint_file, config.get("enable_debugging", False))
                if iteration % 1000 == 0:
                    save_checkpoint(agent, log_directory / f"checkpoint_{iteration}.pt", config.get("enable_debugging", False))
                visualize_gradients(agent, iteration, log_directory)
            
            if agent.current_step % config.get("evaluation_interval_steps", 10000) < environment_steps:
                average_evaluation_reward = evaluate_agent(
                    agent,
                    environment_instance,
                    logger_object,
                    number_of_episodes=config.get("evaluation_number_of_episodes", 10),
                    render_during_evaluation=config.get("render_during_evaluation", False)
                )
                
                if average_evaluation_reward > best_evaluation_reward:
                    best_evaluation_reward = average_evaluation_reward
                    save_checkpoint(agent, best_checkpoint_file, config.get("enable_debugging", False))
                    logger.info(f"New best model saved! Reward: {best_evaluation_reward:.2f}")
                
                elapsed_minutes = (time.time() - start_time) / 60
                progress = agent.current_step / total_training_steps * 100
                logger.info(f"Progress: {progress:.1f}% | Time: {elapsed_minutes:.2f} min | Best reward: {best_evaluation_reward:.2f}")
                plot_training_history(loss_history, log_directory / "training_loss_history.png", config.get("enable_debugging", False))
            
            iteration_time = time.time() - iteration_start_time
            fps = steps_taken / max(iteration_time, 1e-6)
            logger_object.log_scalar("performance/frames_per_second", fps)
            logger_object.log_scalar("performance/updates_per_second", total_updates / max(iteration_time, 1e-6))
            
            if iteration % loss_print_interval == 0:
                logger.info(f"Iter {iteration}: Steps={agent.current_step}/{total_training_steps}, Updates={total_updates}, Time={iteration_time:.3f}s, FPS={fps:.1f}")
            
            iteration += 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as error:
        logger.error(f"Training failed: {error}")
        traceback.print_exc()
    finally:
        average_evaluation_reward = evaluate_agent(
            agent,
            environment_instance,
            logger_object,
            number_of_episodes=config.get("evaluation_number_of_episodes", 10),
            render_during_evaluation=config.get("render_during_evaluation", False)
        )
        
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"\n=== Training Summary ===")
        logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        logger.info(f"Total iterations: {iteration}")
        logger.info(f"Final step count: {agent.current_step}")
        logger.info(f"Final evaluation reward: {average_evaluation_reward:.2f}")
        logger.info(f"Best evaluation reward: {best_evaluation_reward:.2f}")
        
        save_checkpoint(agent, log_directory / "checkpoint_final.pt", config.get("enable_debugging", False))
        plot_training_history(loss_history, log_directory / "training_loss_history.png", config.get("enable_debugging", False))
        
        environment_instance.close()
        logger_object.close()

if __name__ == "__main__":
    check_imports()
    main()