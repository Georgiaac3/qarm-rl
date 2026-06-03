# Octopus Arm Reinforcement Learning

A simple RL environment where an agent learns to control a multi-segment arm to reach a target.

## Overview

The agent controls an octopus-like arm with multiple segments and must learn to reach a randomly placed target in 30 steps (or less).

### Features

- **Multi-segment arm**: Configurable (default: 3 segments)
- **Angle-based control**: Direct control of segment angles
- **Simple goal**: Reach target in 30 steps
- **Stable Baselines3 compatible**: PPO, SAC, DDPG
- **Parallel training**: 4-8x speedup with vectorized environments

## Quick Start

### Training

```bash
# Fast training (PPO, 8 parallel envs)
python3 train.py --timesteps 100000 --num_envs 8

# With SAC algorithm
python3 train.py --algorithm SAC --timesteps 100000 --num_envs 8

# With 5 segments
python3 train.py --num_segments 5 --timesteps 100000 --num_envs 8
```

### Evaluation

```bash
# Test trained model
python3 evaluate.py ./checkpoints/octopus_arm_ppo_final.zip --episodes 10

# No rendering (faster)
python3 evaluate.py ./checkpoints/octopus_arm_ppo_final.zip --no_render --episodes 20
```

## Environment Details

### Action Space
- **Type**: Box(-1.0, 1.0, (num_segments,))
- **Meaning**: Angle change for each segment [-1=left, +1=right]

### Observation Space
- **Type**: Box(-3.0, 3.0, (5,))
- **Content**: [tip_x, tip_y, target_x, target_y, distance]

### Reward

- **+10**: Target reached (distance < 0.1)
- **-distance**: Distance-based shaping
- **-0.01/step**: Efficiency penalty
- **-5.0**: Timeout (30 steps exceeded)



