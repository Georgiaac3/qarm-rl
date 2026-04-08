"""
Quick setup + test script
Verify all imports work before training
"""

import sys
from pathlib import Path

print("=" * 80)
print("2D ARM VALIDATION - SETUP TEST")
print("=" * 80)

# Test imports
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ numpy")
except ImportError as e:
    print(f"   ✗ numpy: {e}")

try:
    import gymnasium as gym
    print("   ✓ gymnasium")
except ImportError as e:
    print(f"   ✗ gymnasium: {e}")

try:
    from stable_baselines3 import SAC
    print("   ✓ stable-baselines3")
except ImportError as e:
    print(f"   ✗ stable-baselines3: {e}")

try:
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib")
except ImportError as e:
    print(f"   ✗ matplotlib: {e}")

# Test local modules
print("\n2. Testing local modules...")
try:
    from kinematics_2d import Arm2D
    print("   ✓ kinematics_2d.Arm2D")
except ImportError as e:
    print(f"   ✗ kinematics_2d: {e}")

try:
    from env_2d import Arm2DEnv
    print("   ✓ env_2d.Arm2DEnv")
except ImportError as e:
    print(f"   ✗ env_2d: {e}")

# Test basic functionality
print("\n3. Testing basic functionality...")
try:
    arm = Arm2D(l1=0.5, l2=0.5)
    q = np.array([0.0, 0.0])
    ee = arm.forward_kinematics(q)
    print(f"   ✓ Forward kinematics: q={q} → ee={ee}")
except Exception as e:
    print(f"   ✗ Forward kinematics: {e}")

try:
    env = Arm2DEnv(max_steps=10)
    obs, info = env.reset()
    print(f"   ✓ Environment reset: obs.shape={obs.shape}")
except Exception as e:
    print(f"   ✗ Environment: {e}")

try:
    action = np.array([0.0, 0.0])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   ✓ Environment step: reward={reward:.4f}, distance={info['distance']:.4f}")
except Exception as e:
    print(f"   ✗ Environment step: {e}")

env.close()

print("\n" + "=" * 80)
print("✓ SETUP TEST COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. python train_2d.py          # Train SAC agent (~10 min)")
print("  2. python visualize_2d.py      # Evaluate & visualize results")
print("=" * 80)
