# Simple 2D Arm Validation Experiment

## Objective

Validate that your **hybrid approach (analytical IK + residual RL correction)** is effective by testing it on a simplified problem with **known optimal behavior**.

This mini-project:
- ✅ Implements a **2-DOF planar arm** with **analytical kinematics**
- ✅ Creates a **reaching task** where the solution is well-understood
- ✅ Compares: **IK-only baseline** vs **IK + RL residual correction**
- ✅ Trains quickly (~5-10 min) to validate the approach

---

## Architecture

### 1. **Kinematics Module** (`kinematics_2d.py`)
- 2-DOF arm: Shoulder + Elbow
- **Forward Kinematics**: Analytically exact
- **Inverse Kinematics**: Closed-form solution (law of cosines)
- **Jacobian**: For velocity mapping

### 2. **Gymnasium Environment** (`env_2d.py`)
- **State**: `[target_x, target_y, q1, q2, ik_q1_dot, ik_q2_dot]`
- **Actions**: Residual corrections `[δq1_dot, δq2_dot]`
- **Reward**: `-distance_to_target + bonus_if_close`
- **Final Command**: `q_dot_final = q_dot_IK + action * scale`

### 3. **Training Script** (`train_2d.py`)
- Uses **Stable-Baselines3 SAC** (same as your main project)
- Trains for 50k timesteps (~10 min on CPU)
- Saves checkpoints and best model

### 4. **Evaluation & Visualization** (`visualize_2d.py`)
- Compares IK-only vs IK+RL on 30 test episodes
- Plots distance distributions
- Visualizes trajectories
- Quantifies improvement %

---

## Quick Start

### Installation
```bash
cd simple_2d_arm
pip install -r requirements.txt
```

### Step 1: Train the Model
```bash
python train_2d.py
```
**Output**: 
- `models_sac/best_model/` → trained SAC agent
- `models_sac/checkpoints/` → intermediate checkpoints
- `models_sac/tensorboard/` → training curves

### Step 2: Validate & Visualize
```bash
python visualize_2d.py
```
**Output**:
- Terminal summary of results
- `validation_results/comparison.png` → distances comparison
- `validation_results/trajectory_ik_only.png` → baseline trajectory
- `validation_results/trajectory_ik_rl.png` → improved trajectory

---

## Expected Results

### Baseline (IK Only)
- No learning, just inverse kinematics
- Error plateau: ≈ 0.10-0.15 m depending on random targets
- Success rate: ≈ 40-60% (within 5cm)

### Hybrid (IK + RL)
- RL learns residual corrections
- Error plateau: ≈ 0.02-0.05 m (significantly better)
- Success rate: ≈ 80-95% (large improvement)

### Key Insight
✅ **RL learns to correct IK errors & improve precision**

This validates that:
1. Your architecture works correctly
2. Residual learning is effective
3. Training converges quickly with good prior

---

## How This Relates to Your QArm Project

| Aspect | 2D Validation | Your QArm Project |
|--------|---------------|-------------------|
| **Kinematics** | Analytical (exact) | Placeholder → needs proper forward kinematics |
| **Task** | Simple reaching | Complex throwing |
| **Model** | 2 DOF | 4 DOF + gripper |
| **Validation** | Quickly verify approach | Full system on real robot |
| **Goal** | Prove hybrid works | Achieve throwing accuracy |

---


## Files Overview

```
simple_2d_arm/
├── kinematics_2d.py      # 2D arm kinematics (exact analyt solution)
├── env_2d.py             # Gymnasium environment
├── train_2d.py           # SAC training script
├── visualize_2d.py       # Evaluation & visualization
├── requirements.txt      # Dependencies
└── README.md             # This file

models_sac/              # Generated after training
├── best_model/          # Best SAC model
├── checkpoints/         # Intermediate models
└── tensorboard/         # Training logs

validation_results/      # Generated after evaluation
├── comparison.png       # Results comparison plots
├── trajectory_ik_only.png
└── trajectory_ik_rl.png
```

---

## Troubleshooting

### 1. Import errors for `gymnasium`
```bash
# Gymnasium replaced gym in 2023+
pip install gymnasium --upgrade
```

### 2. Slow training on CPU
- Expected: ~10 min for 50k steps on CPU
- Use GPU if available (installs torch with CUDA)
- Reduce `total_timesteps` in `train_2d.py` for quick test

### 3. Model not converging
- Check learning rate (try 1e-4 to 1e-3)
- Verify reward function is reasonable (check `_compute_reward()`)
- Increase replay buffer size