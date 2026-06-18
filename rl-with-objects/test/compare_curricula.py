"""
Utility script to compare curriculum learning vs. baseline training results.

Usage:
    python3 rl-with-objects/test/compare_curricula.py
    python3 rl-with-objects/test/compare_curricula.py --plot
    python3 rl-with-objects/test/compare_curricula.py --detailed
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_evaluations(log_dir):
    """Load evaluation results from npz file."""
    eval_file = Path(log_dir) / "evaluations.npz"

    if not eval_file.exists():
        return None

    try:
        data = np.load(eval_file, allow_pickle=True)
        return {
            "timesteps": data["timesteps"],
            "results": data["results"],  # Shape: (n_eval, n_episodes)
            "ep_lengths": data.get("ep_lengths", None),
        }
    except Exception as e:
        print(f"Error loading {eval_file}: {e}")
        return None


def print_header(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title:^80}")
    print(f"{'='*80}\n")


def compare_results():
    """Compare curriculum vs baseline results."""

    checkpoint_dir = Path(__file__).parent.parent / "training" / "checkpoints"

    if not checkpoint_dir.exists():
        print("❌ Checkpoint directory not found!")
        print(f"   Expected: {checkpoint_dir}")
        return

    # Look for WITH_CURRICULUM and BASELINE
    curriculum_path = checkpoint_dir / "WITH_CURRICULUM" / "logs"
    baseline_path = checkpoint_dir / "BASELINE" / "logs"

    print_header("CURRICULUM LEARNING COMPARISON")

    # Try to load results
    print("📂 Looking for results...")

    curriculum_data = load_evaluations(curriculum_path) if curriculum_path.exists() else None
    baseline_data = load_evaluations(baseline_path) if baseline_path.exists() else None

    if curriculum_data is None and baseline_data is None:
        print("❌ No evaluation results found yet!")
        print(f"\n   Curriculum logs:  {curriculum_path}")
        print(f"   Baseline logs:    {baseline_path}")
        print("\n   Run training first:")
        print("   - python3 training/train_curr.py")
        print("   - python3 training/train_curr.py --no-curriculum")
        return

    # Extract results
    results = {}

    if curriculum_data:
        print(f"✅ WITH_CURRICULUM found ({len(curriculum_data['results'])} evaluations)")
        results["curriculum"] = curriculum_data
    else:
        print(f"❌ WITH_CURRICULUM not found")

    if baseline_data:
        print(f"✅ BASELINE found ({len(baseline_data['results'])} evaluations)")
        results["baseline"] = baseline_data
    else:
        print(f"❌ BASELINE not found")

    if not results:
        return

    # Analyze results
    print_header("RESULTS ANALYSIS")

    for name, data in results.items():
        print(f"\n{name.upper()}")
        print("-" * 40)

        # Per-evaluation stats
        timesteps = data["timesteps"]
        rewards = data["results"]

        # Handle 2D array (multiple eval runs)
        if rewards.ndim == 2:
            means = np.mean(rewards, axis=1)
            stds = np.std(rewards, axis=1)
            n_episodes_per_eval = rewards.shape[1]
        else:
            means = rewards
            stds = np.zeros_like(means)
            n_episodes_per_eval = 1

        # Stats
        best_mean = np.max(means)
        worst_mean = np.min(means)
        avg_mean = np.mean(means)
        final_mean = means[-1] if len(means) > 0 else 0

        print(f"  Evaluations:         {len(timesteps)}")
        print(f"  Episodes per eval:   {n_episodes_per_eval}")
        print(f"  Training steps:      {timesteps[-1] if len(timesteps) > 0 else 0:,}")
        print(f"\n  Mean Reward:")
        print(f"    Best:              {best_mean:>7.2f}")
        print(f"    Worst:             {worst_mean:>7.2f}")
        print(f"    Average:           {avg_mean:>7.2f}")
        print(f"    Final:             {final_mean:>7.2f}")
        print(f"\n  Improvement (final vs worst):")
        print(f"    Delta:             {final_mean - worst_mean:>7.2f}")
        print(f"    % Gain:            {(final_mean - worst_mean) / abs(worst_mean) * 100:>7.1f}%")

    # Comparative stats
    if len(results) == 2:
        print_header("COMPARATIVE ANALYSIS")

        curriculum = results.get("curriculum")
        baseline = results.get("baseline")

        if curriculum and baseline:
            # Get final means
            curr_rewards = curriculum["results"]
            base_rewards = baseline["results"]

            if curr_rewards.ndim == 2:
                curr_means = np.mean(curr_rewards, axis=1)
                curr_final = curr_means[-1] if len(curr_means) > 0 else 0
            else:
                curr_final = curr_rewards[-1] if len(curr_rewards) > 0 else 0

            if base_rewards.ndim == 2:
                base_means = np.mean(base_rewards, axis=1)
                base_final = base_means[-1] if len(base_means) > 0 else 0
            else:
                base_final = base_rewards[-1] if len(base_rewards) > 0 else 0

            improvement = curr_final - base_final
            improvement_pct = (improvement / abs(base_final)) * 100 if base_final != 0 else 0

            print(f"\nCurriculum vs Baseline (final rewards):")
            print(f"  Curriculum:        {curr_final:>7.2f}")
            print(f"  Baseline:          {base_final:>7.2f}")
            print(f"  Improvement:       {improvement:>7.2f} ({improvement_pct:>6.1f}%)")

            if improvement > 0:
                print(f"\n  ✅ Curriculum learning is BETTER by {improvement_pct:.1f}%")
            else:
                print(f"\n  ❌ Baseline is better (maybe tuning needed?)")

            # Phase analysis
            curr_timesteps = curriculum["timesteps"]
            if len(curr_timesteps) >= 3:
                print("\n  Curriculum Learning Phases:")
                print(
                    f"    Phase 1 (0-50k):    {curr_means[0] if len(curr_means) > 0 else 0:.2f} reward"
                )
                print(
                    f"    Phase 2 (50-150k):  {np.mean(curr_means[1:3]) if len(curr_means) > 1 else 0:.2f} reward"
                )
                print(f"    Phase 3 (150k+):    {curr_means[-1]:.2f} reward")

    print_header("RECOMMENDATIONS")

    if len(results) == 1:
        print("✅ You have one training completed.")
        print("   Run the other configuration to compare:")
        if "curriculum" in results:
            print("   → python3 training/train_curr.py --no-curriculum")
        else:
            print("   → python3 training/train_curr.py")
    elif len(results) == 2:
        curr_final = np.mean(results["curriculum"]["results"][-1]) if "curriculum" in results else 0
        base_final = np.mean(results["baseline"]["results"][-1]) if "baseline" in results else 0

        if curr_final > base_final * 1.1:  # 10% better
            print("✅ Curriculum learning works well for your setup!")
            print("   Use: python3 training/train_curr.py")
        elif curr_final > base_final:
            print("✅ Curriculum learning shows marginal improvement.")
            print("   Consider tuning reward_shaper.py for better results.")
        else:
            print("⚠️  Baseline seems better.")
            print("   Check:")
            print("   1. Adjust reward multipliers in reward_shaper.py")
            print("   2. Verify environment difficulty is reasonable")
            print("   3. Try longer training (--steps 500000)")

    print()


def plot_results():
    """Plot curriculum vs baseline if matplotlib available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed. Install with:")
        print("   pip3 install matplotlib")
        return

    checkpoint_dir = Path(__file__).parent.parent / "training" / "checkpoints"

    curriculum_data = load_evaluations(checkpoint_dir / "WITH_CURRICULUM" / "logs")
    baseline_data = load_evaluations(checkpoint_dir / "BASELINE" / "logs")

    if not curriculum_data and not baseline_data:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Rewards over time
    ax = axes[0]
    if curriculum_data:
        rewards = curriculum_data["results"]
        if rewards.ndim == 2:
            means = np.mean(rewards, axis=1)
            stds = np.std(rewards, axis=1)
        else:
            means = rewards
            stds = np.zeros_like(means)

        timesteps = curriculum_data["timesteps"]
        ax.plot(timesteps, means, "b-o", label="Curriculum", linewidth=2)
        ax.fill_between(timesteps, means - stds, means + stds, alpha=0.2, color="b")

    if baseline_data:
        rewards = baseline_data["results"]
        if rewards.ndim == 2:
            means = np.mean(rewards, axis=1)
            stds = np.std(rewards, axis=1)
        else:
            means = rewards
            stds = np.zeros_like(means)

        timesteps = baseline_data["timesteps"]
        ax.plot(timesteps, means, "r-s", label="Baseline", linewidth=2)
        ax.fill_between(timesteps, means - stds, means + stds, alpha=0.2, color="r")

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Training Progress: Curriculum vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Distribution comparison
    ax = axes[1]
    if curriculum_data and baseline_data:
        curr_rewards = curriculum_data["results"]
        base_rewards = baseline_data["results"]

        if curr_rewards.ndim == 2:
            curr_final = np.mean(curr_rewards[-1])
        else:
            curr_final = curr_rewards[-1]

        if base_rewards.ndim == 2:
            base_final = np.mean(base_rewards[-1])
        else:
            base_final = base_rewards[-1]

        ax.bar(
            ["Curriculum", "Baseline"],
            [curr_final, base_final],
            color=["blue", "red"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.set_ylabel("Final Mean Reward", fontsize=12)
        ax.set_title("Final Performance", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(checkpoint_dir / "comparison.png", dpi=150)
    print(f"✅ Plot saved to: {checkpoint_dir}/comparison.png")
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compare curriculum learning vs baseline training")
    parser.add_argument(
        "--plot", action="store_true", help="Generate comparison plot (requires matplotlib)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed phase-by-phase analysis"
    )

    args = parser.parse_args()

    compare_results()

    if args.plot:
        print_header("GENERATING PLOT")
        plot_results()
