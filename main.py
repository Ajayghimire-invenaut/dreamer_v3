# Entry point for running the DreamerV3 system
import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import torch
from application.trainer import DreamerTrainer
from application.config_handler import ConfigHandler

def parse_arguments():
    """
    Parse command-line arguments to configure the training run.
    Matches NM512/dreamerv3-torch's argument parsing style.
    """
    parser = argparse.ArgumentParser(description="DreamerV3 Training")
    parser.add_argument("--task", type=str, default="walker_walk",
                        help="Task name (e.g., walker_walk)")
    parser.add_argument("--domain", type=str, default="walker",
                        help="Domain name (e.g., walker)")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for logs")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom experiment name (defaults to timestamp)")
    parser.add_argument("--max_steps", type=int, default=1000000,
                        help="Maximum training steps (default 1M; official is ~100M)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file to resume training (optional)")
    return parser.parse_args()

def main():
    """Initialize and run the DreamerV3 training process with checkpointing."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize configuration
    config_handler = ConfigHandler()
    config = config_handler.get_config()
    
    # Update config with command-line arguments
    config_updates = {
        "task_name": args.task.split("_")[1] if "_" in args.task else args.task,
        "domain_name": args.domain,
        "log_dir": args.log_dir,
        "experiment_name": args.experiment_name or args.task,
        "max_steps": args.max_steps
    }
    config_handler.update_config(config_updates)
    config = config_handler.get_config()
    
    # Print configuration for verification
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = DreamerTrainer(config, device)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Run training
    trainer.train()

if __name__ == "__main__":
    main()