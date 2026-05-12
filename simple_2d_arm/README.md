# Simple 2D Arm - RL Validation

## Objective

Fast validation that **RL can correct weak IK solutions** using a simple 2-DOF arm reaching task.

Quickly verify the hybrid approach works before deploying to the real robot.

---

## Architecture

### Files

| File | Purpose |
|------|---------|
| `kinematics_2d.py` | 2D arm forward/inverse kinematics |
| `env_2d.py` | Gymnasium environment for reaching task |
| `train_2d.py` | SAC training (50k timesteps, ~10 min) |
| `visualize_2d.py` | Evaluation & trajectory visualization |

### Environment

- **Observation**: `[target_x, target_y, q1, q2, ik_q1_dot, ik_q2_dot]`
- **Action**: Residual IK corrections `[δq1_dot, δq2_dot]` ([-1, 1] range)
- **Reward**: `-distance + 10.0 if distance < 0.05m`
- **Dynamics**: `q_final_dot = ik_q_dot + action * 0.5` then integrate

---

## Quick Start

### Installation
```bash
cd simple_2d_arm
pip install -r requirements.txt
```

### Test Without Training

If you already have a trained model:

```bash
python visualize_2d.py
```

**Output**:
- Terminal: Mean distance, std dev, success rate
- Plot: End-effector trajectory with target

Expected on first run without a model: Error message → proceed to training

### Train a New Model

```bash
python train_2d.py
```

**Output**:
- `models_sac/final_model.zip` → Final trained SAC agent
- `models_sac/best_model/` → Best checkpoint found
- `models_sac/checkpoints/` → Intermediate checkpoints
- Training progress printed to terminal

**Time**: ~10 min on CPU, ~1-2 min on GPU

### Evaluate Trained Model

After training, run:

```bash
python visualize_2d.py
```

**Output**:
- Terminal metrics (distance, success rate)
- Matplotlib figure showing trajectory

---

## Results

After training 50k timesteps on the 2D reaching task:

**With IK + RL**:
- Mean error: ~0.02-0.05 m (depending on random initialization)
- Success rate: 80-95% (within 5cm of target)
- Converges efficiently thanks to weak IK prior

The RL agent learns to correct IK limitations and improve end-effector precision.

---

## When to Use This

- ✅ **Before deploying to real robot**: Run this toy example to verify training works
- ✅ **Debug your RL setup**: If QArm training fails, check if basic RL works here
- ✅ **Prototype ideas**: Test new reward functions, network architectures, etc.
- ❌ **Not for**: Solving the actual throwing task (use main QArm project)

---

## Directory Structure

```
simple_2d_arm/
├── kinematics_2d.py          # Forward & inverse kinematics
├── env_2d.py                 # Gymnasium environment
├── train_2d.py               # SAC training (50k steps)
├── visualize_2d.py           # Evaluation & visualization
├── requirements.txt          # Dependencies
└── README.md                 # This file

models_sac/                   # (Created after training)
├── final_model.zip           # Final model
├── best_model/
│   └── best_model.zip        # Best model checkpoint
└── checkpoints/
    ├── sac_2d_10000_steps.zip
    └── ...
```

---

## Troubleshooting

### Model not found error
```
✗ Model not found. Run: python train_2d.py
```
→ Train first with `python train_2d.py`, then evaluate

### Import errors
```bash
pip install gymnasium stable-baselines3 matplotlib numpy
```

### Slow training
- Expected: ~10 min on CPU
- Use GPU for faster training
- Reduce `total_timesteps` in `train_2d.py` to 10k for quick test


## Note:
Model created - Policy: SACPolicy(
  (actor): Actor(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (latent_pi): Sequential(
      (0): Linear(in_features=6, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
    )
    (mu): Linear(in_features=64, out_features=2, bias=True)
    (log_std): Linear(in_features=64, out_features=2, bias=True)
  )
  (critic): ContinuousCritic(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (qf0): Sequential(
      (0): Linear(in_features=8, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=1, bias=True)
    )
    (qf1): Sequential(
      (0): Linear(in_features=8, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=1, bias=True)
    )
  )
  (critic_target): ContinuousCritic(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (qf0): Sequential(
      (0): Linear(in_features=8, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=1, bias=True)
    )
    (qf1): Sequential(
      (0): Linear(in_features=8, out_features=64, bias=True)
      (1): ReLU()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU()
      (4): Linear(in_features=64, out_features=1, bias=True)
    )
  )
)