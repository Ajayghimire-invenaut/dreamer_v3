# DreamerV3 - PyTorch Implementation

This repository implements DreamerV3, a model-based reinforcement learning algorithm, in PyTorch. It is based on the paper ["Mastering Diverse Domains through World Models"](https://arxiv.org/abs/2301.04104). This implementation targets the DeepMind Control Suite (DMC) `walker_walk` task and is structured using a Domain-Driven Design (DDD) approach.

## Features

- **World Model**: Recurrent State Space Model (RSSM) for environment dynamics prediction
- **Actor-Critic**: Imagination-based policy and value learning
- **Environment**: Supports DMC tasks with 64×64×3 pixel inputs
- **Logging**: TensorBoard integration for monitoring training progress

## Requirements

- **OS**: Windows 10/11 (CPU-only tested) or Linux (recommended)
- **Python**: 3.11
- **RAM**: ≥ 4 GB (for `buffer_capacity=25000`)
- **Hardware**: CPU (GPU optional)

## Installation

```bash
git clone <repository-url>
cd dreamer_version_3
python -m venv venv
venv\Scripts\activate  # On Windows
# On Linux/Mac use: source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Requirements File

Create a `requirements.txt` file with the following dependencies:

```
# Core PyTorch dependencies (CPU-only)
torch==2.0.1+cpu
torchvision==0.15.2+cpu

# Environment and RL dependencies
tensorboard==2.17.1
gym==0.26.2
mujoco==2.3.5
dm_control==1.0.9
moviepy==1.0.3

# Atari-specific (optional)
atari-py==0.2.9
opencv-python==4.10.0.84

# Utilities
numpy==1.26.4
pyopengl==3.1.7
```

## Quick Start

### Train the Model

```bash
python main.py
```

To run 100M steps for official results:

```bash
python main.py --max_steps 100000000
```

Test with fewer steps first (e.g., 10M):

```bash
python main.py --max_steps 10000000
```

### Monitor with TensorBoard

```bash
tensorboard --logdir ./logs
```

Visit http://localhost:6006 in your browser.

## Configuration Highlights

Key hyperparameters in `application/config_handler.py`:

```yaml
collect_steps: 500      # Steps to collect before training
batch_size: 16          # Number of sequences per batch
batch_length: 50        # Sequence length
horizon: 15             # Imagination horizon
buffer_capacity: 25000  # Replay buffer size
world_lr: 0.0003        # World model learning rate
actor_lr: 8e-5          # Actor learning rate
critic_lr: 8e-5         # Critic learning rate
kl_free: 1.0            # Free nats for KL divergence
```

## Project Structure

```
dreamer_version_3/
├── core/
│   ├── world_domain/
│   │   ├── dynamics.py           # RSSM implementation
│   │   ├── encoder.py            # Image encoding
│   │   ├── decoder.py            # Image reconstruction
│   │   └── reward_head.py        # Reward prediction
│   ├── behavior_domain/
│   │   ├── actor.py              # Policy network
│   │   ├── critic.py             # Value network
│   │   └── exploration.py        # Exploration strategies
│   ├── utilities/
│   │   ├── data_tools.py         # Data processing utilities
│   │   └── replay_buffer.py      # Experience replay
├── infrastructure/
│   ├── environment_manager.py    # DMC environment wrapper
│   └── logger.py                 # Tensorboard logging
├── application/
│   ├── trainer.py                # Training loop implementation
│   └── config_handler.py         # Configuration management
├── main.py                       # Entry point
└── requirements.txt              # Dependencies
```

## Technical Implementation Details

### World Model

The world model consists of:

- **Encoder**: Convolutional neural network that embeds 64×64×3 images
- **RSSM**: Recurrent state-space model with deterministic and stochastic components
- **Decoder**: Transposed convolutional network for image reconstruction
- **Reward Predictor**: MLP that predicts rewards from latent states

### Actor-Critic Architecture

- **Actor**: Outputs continuous action distribution (tanh-transformed normal)
- **Critic**: Estimates expected returns from imagined trajectories
- **Imagination**: Rollouts in latent space for policy improvement

### Training Process

1. **Collect experience**: Interact with environment using current policy
2. **World model learning**: Train RSSM, encoder, decoder and reward predictor
3. **Behavior learning**: Train actor and critic on imagined trajectories
4. **Repeat**: Alternate between data collection and model updates

## Tips for Successful Training

- **Memory Issues**: If you encounter memory errors, lower `buffer_capacity` to 10000
- **Reward Sparsity**: Increase `collect_steps` or add maximum episode length if rewards are sparse
- **Computation Speed**: Start with fewer steps on CPU before scaling up to full training
- **State Representation**: The RSSM's latent state quality is crucial for good performance

## Performance Target

🎯 The goal is to achieve 800–1000 episode return on the `walker_walk` task after 100M environment steps.

## License

[MIT License](LICENSE)

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```