"""Simple evaluation and visualization script for 2D Arm RL."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from env_2d import Arm2DEnv
from stable_baselines3 import SAC


def evaluate_ik_only(env: Arm2DEnv, num_episodes: int = 20):
    """Baseline: IK without RL correction."""
    distances = []
    success_count = 0

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
            success_count += 1

    return {
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "success_rate": success_count / num_episodes,
        "distances": np.array(distances),
    }


def evaluate_ik_plus_rl(model: SAC, env: Arm2DEnv, num_episodes: int = 20, deterministic: bool = True):
    """IK + RL: Inverse kinematics with learned residual corrections."""
    distances = []
    success_count = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            distance = info["distance"]

        distances.append(distance)
        if distance < 0.05:
            success_count += 1

    return {
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "success_rate": success_count / num_episodes,
        "distances": np.array(distances),
    }


def visualize_trajectory(env, model=None, use_rl=True, num_steps=100):
    """Visualize single trajectory."""
    obs, info = env.reset()
    target = info["target"]
    trajectory = []
    q_history = []

    for _ in range(num_steps):
        if use_rl and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0])  # IK only
        
        ee_pos = env.arm.forward_kinematics(env.q)
        trajectory.append(ee_pos.copy())
        q_history.append(env.q.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    trajectory = np.array(trajectory)
    q_final = q_history[-1]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="Trajectory")
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start")
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=10, label="End")
    ax.plot(target[0], target[1], "r*", markersize=20, label="Target")

    # Workspace
    circle = patches.Circle(
        (0, 0), env.arm.l1 + env.arm.l2, fill=False, linestyle="--", color="gray", alpha=0.3
    )
    ax.add_patch(circle)

    # Arm config
    elbow = np.array([env.arm.l1 * np.cos(q_final[0]), env.arm.l1 * np.sin(q_final[0])])
    ee = env.arm.forward_kinematics(q_final)
    ax.plot([0, elbow[0]], [0, elbow[1]], "k-", linewidth=3, alpha=0.5)
    ax.plot([elbow[0], ee[0]], [elbow[1], ee[1]], "k-", linewidth=3, alpha=0.5)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    title = "IK + RL" if use_rl else "IK Only"
    ax.set_title(f"2D Arm Trajectory ({title})")
    plt.tight_layout()
    plt.show()


def main():
    """Main evaluation script."""
    print("\n" + "=" * 70)
    print("2D ARM EVALUATION - IK vs IK+RL")
    print("=" * 70 + "\n")

    model_path = Path("models_sac/final_model.zip")
    if not model_path.exists():
        model_path = Path("models_sac/best_model/best_model.zip")

    if not model_path.exists():
        print("✗ Model not found. Run: python train_2d.py")
        return

    print(f"✓ Loading model from {model_path}\n")
    model = SAC.load(str(model_path))

    eval_env = Arm2DEnv(max_steps=100)
    
    # Evaluate both approaches
    ik_results = evaluate_ik_only(eval_env, num_episodes=30)
    rl_results = evaluate_ik_plus_rl(model, eval_env, num_episodes=30)

    print("-" * 70)
    print("IK ONLY (Baseline)")
    print("-" * 70)
    print(f"Mean distance:  {ik_results['mean_distance']:.4f} m")
    print(f"Std deviation:  {ik_results['std_distance']:.4f} m")
    print(f"Success rate:   {ik_results['success_rate']*100:.1f}%")
    
    print("\n" + "-" * 70)
    print("IK + RL (Learned)")
    print("-" * 70)
    print(f"Mean distance:  {rl_results['mean_distance']:.4f} m")
    print(f"Std deviation:  {rl_results['std_distance']:.4f} m")
    print(f"Success rate:   {rl_results['success_rate']*100:.1f}%")
    
    print("\n" + "-" * 70)
    print("COMPARISON")
    print("-" * 70)
    improvement = (rl_results['success_rate'] - ik_results['success_rate']) * 100
    distance_improvement = (1 - rl_results['mean_distance'] / ik_results['mean_distance']) * 100
    print(f"Success delta:  {improvement:+.1f}%")
    print(f"Distance delta: {distance_improvement:+.1f}%")
    print("=" * 70 + "\n")

    # Visualize both approaches
    print("Visualizing IK only...")
    visualize_trajectory(eval_env, model=None, use_rl=False)
    
    print("\nVisualizing IK + RL...")
    visualize_trajectory(eval_env, model=model, use_rl=True)


if __name__ == "__main__":
    main()
