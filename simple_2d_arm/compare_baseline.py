"""Compare IK only vs IK+RL performance."""
import numpy as np
from pathlib import Path
from env_2d import Arm2DEnv
from stable_baselines3 import SAC


def evaluate_ik_only(env, num_episodes=30):
    """Evaluate IK baseline (no RL)."""
    distances = []
    success = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = np.array([0.0, 0.0])  # No RL correction
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            distance = info["distance"]
        
        distances.append(distance)
        if distance < 0.05:
            success += 1
    
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "success_rate": success / num_episodes,
        "distances": np.array(distances),
    }


def evaluate_ik_plus_rl(model, env, num_episodes=30):
    """Evaluate IK + RL."""
    distances = []
    success = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            distance = info["distance"]
        
        distances.append(distance)
        if distance < 0.05:
            success += 1
    
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "success_rate": success / num_episodes,
        "distances": np.array(distances),
    }


# Load model
model_path = Path("models_sac/final_model.zip")
if not model_path.exists():
    print("Model not found")
    exit()

model = SAC.load(str(model_path))
env = Arm2DEnv(max_steps=100)

# Evaluate both
print("\n" + "=" * 70)
print("BASELINE vs IMPROVED COMPARISON")
print("=" * 70)

ik_results = evaluate_ik_only(env, num_episodes=30)
rl_results = evaluate_ik_plus_rl(model, env, num_episodes=30)

print("\nIK ONLY (Baseline):")
print(f"  Mean distance: {ik_results['mean']:.4f} m")
print(f"  Success rate:  {ik_results['success_rate']*100:.1f}%")

print("\nIK + RL (Improved):")
print(f"  Mean distance: {rl_results['mean']:.4f} m")
print(f"  Success rate:  {rl_results['success_rate']*100:.1f}%")

improvement = (rl_results['success_rate'] - ik_results['success_rate']) * 100
print(f"\nRL CONTRIBUTION: {improvement:.1f}% success rate")
print(f"Error reduction: {(1 - rl_results['mean']/ik_results['mean'])*100:.1f}%")
print("=" * 70 + "\n")

env.close()
