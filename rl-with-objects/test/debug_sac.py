#!/usr/bin/env python3
"""Debug SAC model wrapper."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from environment.bin_picking_env import BinPickingEnv
from config import get_config

config = get_config()
device = config["model"]["device"]

# Load SAC model
model_path = Path(__file__).parent.parent / "training" / "checkpoints" / "rl_policy_10000_steps.zip"
print(f"Loading SAC model from {model_path}...")
sac_model = SAC.load(str(model_path), device=device)
print(f"✓ Loaded SAC model")

# Create environment
env = BinPickingEnv(
    image_height=config['env']['image_height'],
    image_width=config['env']['image_width'],
    max_steps=config['env']['max_steps'],
    n_objects=config['env']['n_objects'],
)

obs, info = env.reset()
print(f"\n✓ Reset environment")
print(f"  Observation keys: {obs.keys()}")
for key in obs:
    print(f"    {key}: shape={obs[key].shape}, dtype={obs[key].dtype}, min={obs[key].min():.3f}, max={obs[key].max():.3f}")

# Test prediction
print(f"\nTesting SAC prediction...")
action, _ = sac_model.predict(obs, deterministic=True)
print(f"✓ Got action: {action}")
print(f"  action[0] (grasp_x): {action[0]:.3f}")
print(f"  action[1] (grasp_y): {action[2]:.3f}")
print(f"  action[2] (vel_x):   {action[2]:.3f}")
print(f"  action[3] (vel_y):   {action[3]:.3f}")

# Try several predictions
print(f"\nTesting 5 predictions:")
for i in range(5):
    action, _ = sac_model.predict(obs, deterministic=True)
    print(f"  Step {i+1}: action={action}")

print(f"\n✓ SAC model working correctly")
