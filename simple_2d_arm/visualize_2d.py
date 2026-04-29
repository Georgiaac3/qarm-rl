"""
Evaluation and Visualization script for 2D Arm RL
Compares: Analytical IK vs IK + RL correction
"""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from env_2d import Arm2DEnv
from kinematics_2d import Arm2D
from stable_baselines3 import SAC


def evaluate_ik_only(env: Arm2DEnv, num_episodes: int = 20) -> dict:
    """
    Evaluate performance using only inverse kinematics (no RL correction).

    This is the "analytical baseline" - what your hybrid approach improves upon.
    """
    print(f"\nEvaluating IK-only baseline ({num_episodes} episodes)...")

    distances = []
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_distance = float("inf")

        while not done:
            # Use only IK velocities (no RL correction)
            # In the env, this is: action = [0, 0] (no correction)
            action = np.array([0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_distance = info["distance"]

        distances.append(ep_distance)
        if ep_distance < 0.05:
            success_count += 1

    return {
        "distances": np.array(distances),
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "success_rate": success_count / num_episodes,
    }


def evaluate_ik_plus_rl(
    model: SAC,
    env: Arm2DEnv,
    num_episodes: int = 20,
    deterministic: bool = True,
) -> dict:
    """
    Evaluate performance using IK + RL residual correction.
    """
    print(f"\nEvaluating IK + RL ({num_episodes} episodes)...")

    distances = []
    success_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_distance = float("inf")

        while not done:
            # RL agent computes correction
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_distance = info["distance"]

        distances.append(ep_distance)
        if ep_distance < 0.05:
            success_count += 1

    return {
        "distances": np.array(distances),
        "mean_distance": np.mean(distances),
        "std_distance": np.std(distances),
        "min_distance": np.min(distances),
        "max_distance": np.max(distances),
        "success_rate": success_count / num_episodes,
    }


def plot_comparison(ik_results: dict, rl_results: dict, save_path: str = None):
    """Plot results comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Distance distribution
    ax = axes[0]
    ax.boxplot([ik_results["distances"], rl_results["distances"]], labels=["IK Only", "IK + RL"])
    ax.set_ylabel("Final Distance to Target (m)")
    ax.set_title("Distance Distribution")
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(ik_results["distances"], alpha=0.6, label="IK Only", bins=10)
    ax.hist(rl_results["distances"], alpha=0.6, label="IK + RL", bins=10)
    ax.set_xlabel("Final Distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Distance Histogram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Summary metrics
    ax = axes[2]
    ax.axis("off")

    summary_text = f"""
    ═══════════════════════════════════════
    VALIDATION RESULTS
    ═══════════════════════════════════════

    ANALYTICAL BASELINE (IK Only):
      Mean Distance:     {ik_results['mean_distance']:.4f} m
      Std Dev:           {ik_results['std_distance']:.4f} m
      Min Distance:      {ik_results['min_distance']:.4f} m
      Success Rate:      {ik_results['success_rate']*100:.1f}%

    HYBRID APPROACH (IK + RL):
      Mean Distance:     {rl_results['mean_distance']:.4f} m
      Std Dev:           {rl_results['std_distance']:.4f} m
      Min Distance:      {rl_results['min_distance']:.4f} m
      Success Rate:      {rl_results['success_rate']*100:.1f}%

    IMPROVEMENT:
      Mean Distance ↓:   {(1 - rl_results['mean_distance']/ik_results['mean_distance'])*100:.1f}%
      Success Rate ↑:    {(rl_results['success_rate'] - ik_results['success_rate'])*100:.1f}%
    """

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Plot saved to {save_path}")

    plt.show()


def visualize_trajectory(
    env: Arm2DEnv,
    model: SAC = None,
    use_rl: bool = True,
    num_steps: int = 100,
    save_path: str = None,
):
    """Visualize single trajectory with arm animation."""
    obs, info = env.reset()
    target = info["target"]

    # Collect trajectory
    trajectory = []
    q_history = []

    for step in range(num_steps):
        if use_rl and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = np.array([0.0, 0.0])

        ee_pos = env.arm.forward_kinematics(env.q)
        trajectory.append(ee_pos.copy())
        q_history.append(env.q.copy())

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    trajectory = np.array(trajectory)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="End-effector trajectory")
    ax.plot(trajectory[0, 0], trajectory[0, 1], "go", markersize=10, label="Start")
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", markersize=10, label="End")
    ax.plot(target[0], target[1], "r*", markersize=20, label="Target")

    # Plot workspace limit
    circle = patches.Circle(
        (0, 0), env.arm.l1 + env.arm.l2, fill=False, linestyle="--", color="gray", alpha=0.3
    )
    ax.add_patch(circle)

    # Plot arm in final configuration
    q_final = q_history[-1]
    elbow_pos = np.array([env.arm.l1 * np.cos(q_final[0]), env.arm.l1 * np.sin(q_final[0])])
    ee_pos = env.arm.forward_kinematics(q_final)

    ax.plot([0, elbow_pos[0]], [0, elbow_pos[1]], "k-", linewidth=3, alpha=0.5)
    ax.plot([elbow_pos[0], ee_pos[0]], [elbow_pos[1], ee_pos[1]], "k-", linewidth=3, alpha=0.5)
    ax.plot(0, 0, "ko", markersize=8)
    ax.plot(elbow_pos[0], elbow_pos[1], "ko", markersize=6)
    ax.plot(ee_pos[0], ee_pos[1], "ko", markersize=6)

    # Distance annotation
    final_distance = np.linalg.norm(ee_pos - target)
    ax.text(
        0.5,
        -0.95,
        f"Final distance: {final_distance:.4f} m",
        transform=ax.transAxes,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    title = "IK + RL" if use_rl else "IK Only"
    ax.set_title(f"2D Arm Trajectory ({title})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✓ Trajectory saved to {save_path}")

    plt.show()


def main():
    """Main evaluation script."""
    print("=" * 80)
    print("2D ARM VALIDATION EXPERIMENT")
    print("Comparing: Analytical IK vs IK + RL")
    print("=" * 80)

    # Load model - try final_model first, then best_model
    model_path = Path("models_sac/final_model.zip")
    if not model_path.exists():
        model_path = Path("models_sac/best_model/best_model.zip")

    if not model_path.exists():
        print(f"✗ Model not found at {model_path}")
        print("  Run 'python train_2d.py' first to train the model")
        return

    print(f"✓ Loading model from {model_path}")
    model = SAC.load(str(model_path))

    # Create evaluation environment
    eval_env = Arm2DEnv(max_steps=100)

    # Evaluate both approaches
    ik_results = evaluate_ik_only(eval_env, num_episodes=30)
    rl_results = evaluate_ik_plus_rl(model, eval_env, num_episodes=30)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nIK Only (Baseline):")
    print(
        f"  Mean distance: {ik_results['mean_distance']:.4f} m (±{ik_results['std_distance']:.4f})"
    )
    print(f"  Success rate:  {ik_results['success_rate']*100:.1f}%")

    print("\nIK + RL (Hybrid):")
    print(
        f"  Mean distance: {rl_results['mean_distance']:.4f} m (±{rl_results['std_distance']:.4f})"
    )
    print(f"  Success rate:  {rl_results['success_rate']*100:.1f}%")

    improvement = (1 - rl_results["mean_distance"] / ik_results["mean_distance"]) * 100
    print(f"\nImprovement: {improvement:.1f}% reduction in mean error")

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    save_dir = Path("validation_results")
    save_dir.mkdir(exist_ok=True)

    # Comparison plot
    plot_comparison(ik_results, rl_results, save_path=str(save_dir / "comparison.png"))

    # Trajectories
    print("\nVisualizing trajectories...")

    visualize_trajectory(
        eval_env,
        model=None,
        use_rl=False,
        num_steps=100,
        save_path=str(save_dir / "trajectory_ik_only.png"),
    )

    visualize_trajectory(
        eval_env,
        model=model,
        use_rl=True,
        num_steps=100,
        save_path=str(save_dir / "trajectory_ik_rl.png"),
    )

    print("\n" + "=" * 80)
    print("✓ VALIDATION COMPLETE")
    print("=" * 80)
    eval_env.close()


if __name__ == "__main__":
    main()
