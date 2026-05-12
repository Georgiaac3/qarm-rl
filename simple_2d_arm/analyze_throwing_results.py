"""
Detailed Analysis of 2D Throwing Task Results
Analyze what worked, what didn't, and improvements
"""

import matplotlib.pyplot as plt
import numpy as np


def analyze_throwing_results(eval_file="models_sac_throwing/eval_logs/evaluations.npz"):
    """Analyze throwing task results in detail."""

    try:
        data = np.load(eval_file)
        timesteps = data["timesteps"]
        results = data["results"]  # Shape: (num_evals, num_episodes)
    except FileNotFoundError:
        print(f"❌ File not found: {eval_file}")
        print("   Did you run train_2d_throwing.py?")
        return

    print("\n" + "=" * 80)
    print("THROWING TASK - DETAILED RESULTS ANALYSIS")
    print("=" * 80 + "\n")

    # Basic stats
    mean_rews = np.mean(results, axis=1)
    std_rews = np.std(results, axis=1)
    min_rews = np.min(results, axis=1)
    max_rews = np.max(results, axis=1)

    # Print header
    print(f"Training Duration: {timesteps[0]:,} to {timesteps[-1]:,} steps")
    print(f"Evaluation Points: {len(timesteps)}")
    print(f"Episodes per Eval: {results.shape[1]}")
    print()

    # 1. PERFORMANCE OVERVIEW
    print("=" * 80)
    print("1. PERFORMANCE OVERVIEW")
    print("=" * 80)

    print(f"\nInitial (Step {timesteps[0]:,}):")
    print(f"  Mean: {mean_rews[0]:>8.2f}  | Std: {std_rews[0]:>8.2f}")
    print(f"  Range: [{min_rews[0]:>7.2f}, {max_rews[0]:>7.2f}]")

    print(f"\nFinal (Step {timesteps[-1]:,}):")
    print(f"  Mean: {mean_rews[-1]:>8.2f}  | Std: {std_rews[-1]:>8.2f}")
    print(f"  Range: [{min_rews[-1]:>7.2f}, {max_rews[-1]:>7.2f}]")

    if np.max(mean_rews) != mean_rews[-1]:
        best_idx = np.argmax(mean_rews)
        print(f"\nBest (Step {timesteps[best_idx]:,}):")
        print(f"  Mean: {mean_rews[best_idx]:>8.2f}  | Std: {std_rews[best_idx]:>8.2f}")

    # 2. IMPROVEMENT ANALYSIS
    print("\n" + "=" * 80)
    print("2. IMPROVEMENT ANALYSIS")
    print("=" * 80)

    improvement = mean_rews[-1] - mean_rews[0]
    improvement_pct = (improvement / abs(mean_rews[0])) * 100 if mean_rews[0] != 0 else 0

    print(f"\nAbsolute Improvement: {improvement:+.2f}")
    print(f"Percentage Improvement: {improvement_pct:+.1f}%")

    # Stability analysis
    stability_init = std_rews[0] / abs(mean_rews[0]) if mean_rews[0] != 0 else float("inf")
    stability_final = std_rews[-1] / abs(mean_rews[-1]) if mean_rews[-1] != 0 else float("inf")

    print(f"\nStability (Coefficient of Variation):")
    print(f"  Initial: {stability_init:.4f}")
    print(f"  Final:   {stability_final:.4f}")

    if stability_final < stability_init:
        print(f"  ✓ Improved by {(stability_init - stability_final) / stability_init * 100:.1f}%")
    else:
        print(f"  ❌ Worsened by {(stability_final - stability_init) / stability_init * 100:.1f}%")

    # 3. SUCCESS RATE ANALYSIS
    print("\n" + "=" * 80)
    print("3. SUCCESS METRICS (Hit = reward ≥ 50)")
    print("=" * 80)

    # Threshold for hitting: reward >= 50 means good trajectory
    hit_threshold = 50

    hits_per_eval = [np.sum(results[i] >= hit_threshold) for i in range(len(timesteps))]
    hit_rates = [hits / results.shape[1] * 100 for hits in hits_per_eval]

    print(f"\nHit Rate (reward ≥ {hit_threshold}):")
    print(f"  Step {timesteps[0]:>6,} | {hit_rates[0]:>5.1f}%")
    print(f"  Step {timesteps[-1]:>6,} | {hit_rates[-1]:>5.1f}%")

    if hit_rates[-1] > hit_rates[0]:
        print(f"  ✓ Improvement: +{hit_rates[-1] - hit_rates[0]:.1f}%")
    elif hit_rates[-1] < hit_rates[0]:
        print(f"  ❌ Degradation: {hit_rates[-1] - hit_rates[0]:.1f}%")

    # 4. CONVERGENCE ANALYSIS
    print("\n" + "=" * 80)
    print("4. CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Check if converging to optimum
    last_5_mean = np.mean(mean_rews[-5:])
    first_5_mean = np.mean(mean_rews[:5])

    print(f"\nFirst 5 checkpoints average: {first_5_mean:.2f}")
    print(f"Last 5 checkpoints average: {last_5_mean:.2f}")
    print(f"Improvement in last phase: {last_5_mean - first_5_mean:+.2f}")

    # Check convergence criterion: if variance is small and trend is flat
    recent_trend = np.polyfit(timesteps[-5:], mean_rews[-5:], 1)[0]
    print(f"\nRecent trend (slope): {recent_trend:.6f}")

    if abs(recent_trend) < 0.001:
        print("  ✓ Policy converged (flat trend)")
    elif recent_trend > 0:
        print("  ⚠ Still improving slightly")
    else:
        print("  ❌ Degrading at the end")

    # 5. VARIANCE PROGRESSION
    print("\n" + "=" * 80)
    print("5. VARIANCE PROGRESSION")
    print("=" * 80)

    print(f"\nInitial Std Dev: {std_rews[0]:.2f}")
    print(f"Final Std Dev: {std_rews[-1]:.2f}")
    print(f"Max Std Dev: {np.max(std_rews):.2f} at step {timesteps[np.argmax(std_rews)]:,}")
    print(f"Average Std Dev: {np.mean(std_rews):.2f}")

    if std_rews[-1] < std_rews[0]:
        print(f"\n✓ Variance decreased: {std_rews[0] - std_rews[-1]:.2f} reduction")
    else:
        print(f"\n❌ Variance increased: {std_rews[-1] - std_rews[0]:+.2f}")

    # 6. FAILURE ANALYSIS
    print("\n" + "=" * 80)
    print("6. FAILURE ANALYSIS (Low Rewards)")
    print("=" * 80)

    failure_threshold = -1.0  # Very bad performance
    failures_per_eval = [np.sum(results[i] < failure_threshold) for i in range(len(timesteps))]
    failure_rates = [fails / results.shape[1] * 100 for fails in failures_per_eval]

    print(f"\nFail Rate (reward < {failure_threshold}):")
    print(f"  Step {timesteps[0]:>6,} | {failure_rates[0]:>5.1f}%")
    print(f"  Step {timesteps[-1]:>6,} | {failure_rates[-1]:>5.1f}%")

    if failure_rates[-1] < failure_rates[0]:
        print(f"  ✓ Fewer failures: -{failure_rates[0] - failure_rates[-1]:.1f}%")
    else:
        print(f"  ❌ More failures: +{failure_rates[-1] - failure_rates[0]:.1f}%")

    # 7. DETAILED CHECKPOINT TABLE
    print("\n" + "=" * 80)
    print("7. CHECKPOINT-BY-CHECKPOINT BREAKDOWN")
    print("=" * 80)
    print()
    print(f"{'Step':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Hit%':<8} {'Fail%':<8}")
    print("-" * 80)

    for i, step in enumerate(timesteps):
        hit_pct = hit_rates[i]
        fail_pct = failure_rates[i]
        print(
            f"{step:<8,} {mean_rews[i]:<10.2f} {std_rews[i]:<10.2f} "
            f"{min_rews[i]:<10.2f} {max_rews[i]:<10.2f} {hit_pct:<8.1f} {fail_pct:<8.1f}"
        )

    # 8. KEY FINDINGS
    print("\n" + "=" * 80)
    print("8. KEY FINDINGS & DIAGNOSIS")
    print("=" * 80)

    findings = []

    # Check 1: Is it converging?
    if abs(recent_trend) < 0.001 and last_5_mean > first_5_mean:
        findings.append("✓ CONVERGING: Policy reached stable state")
    elif abs(recent_trend) < 0.001 and last_5_mean <= first_5_mean:
        findings.append("❌ CONVERGED TO BAD: Policy stable but poor rewards")
    elif recent_trend > 0.001:
        findings.append("⚠ STILL LEARNING: Policy still improving (needs more steps)")
    else:
        findings.append("❌ DEGRADING: Policy getting worse (possible overfitting/instability)")

    # Check 2: Is variance good?
    if stability_final < 0.5:
        findings.append("✓ STABLE POLICY: Low variance relative to reward")
    elif stability_final < 1.0:
        findings.append("⚠ MODERATE VARIANCE: Acceptable but could be better")
    else:
        findings.append("❌ UNSTABLE POLICY: High variance - policy is unpredictable")

    # Check 3: Success rate
    if hit_rates[-1] > 50:
        findings.append(f"✓ GOOD HIT RATE: {hit_rates[-1]:.1f}% successful throws")
    elif hit_rates[-1] > 20:
        findings.append(f"⚠ MODERATE SUCCESS: {hit_rates[-1]:.1f}% hit rate (can improve)")
    else:
        findings.append(f"❌ POOR SUCCESS: Only {hit_rates[-1]:.1f}% throws succeed")

    # Check 4: Exploration
    if std_rews[-1] < std_rews[0]:
        findings.append("✓ CONTROLLED EXPLORATION: Variance decreased over time")
    else:
        findings.append("❌ EXCESSIVE EXPLORATION: Variance stayed high/increased")

    print("\n" + "\n".join(findings))

    return {
        "timesteps": timesteps,
        "means": mean_rews,
        "stds": std_rews,
        "hit_rates": hit_rates,
        "stability": stability_final,
        "improvement": improvement,
    }


def plot_throwing_analysis(eval_file="models_sac_throwing/eval_logs/evaluations.npz"):
    """Create comprehensive analysis plots."""

    try:
        data = np.load(eval_file)
        timesteps = data["timesteps"]
        results = data["results"]
    except FileNotFoundError:
        return

    mean_rews = np.mean(results, axis=1)
    std_rews = np.std(results, axis=1)
    min_rews = np.min(results, axis=1)
    max_rews = np.max(results, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Throwing Task - Detailed Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Performance with confidence intervals
    ax = axes[0, 0]
    ax.plot(timesteps, mean_rews, "b-", linewidth=2, label="Mean", marker="o")
    ax.fill_between(
        timesteps, mean_rews - std_rews, mean_rews + std_rews, alpha=0.2, label="±1 Std"
    )
    ax.set_ylabel("Episode Return", fontsize=11)
    ax.set_title("Learning Curve with Uncertainty")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Min/Max range
    ax = axes[0, 1]
    ax.fill_between(timesteps, min_rews, max_rews, alpha=0.3, label="Min-Max")
    ax.plot(timesteps, mean_rews, "b-", linewidth=2, label="Mean")
    ax.set_ylabel("Episode Return", fontsize=11)
    ax.set_title("Performance Range")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Hit rates (assuming reward >= 50)
    ax = axes[1, 0]
    hit_rates = [np.sum(results[i] >= 50) / len(results[i]) * 100 for i in range(len(timesteps))]
    ax.bar(
        timesteps,
        hit_rates,
        width=timesteps[1] - timesteps[0] if len(timesteps) > 1 else 5000,
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    ax.set_ylabel("Hit Rate (%)", fontsize=11)
    ax.set_xlabel("Training Steps")
    ax.set_title("Success Rate Over Training")
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, axis="y")

    # Plot 4: Stability (coefficient of variation)
    ax = axes[1, 1]
    cv = std_rews / np.abs(mean_rews + 1e-6)
    ax.plot(timesteps, cv, "o-", linewidth=2, color="orange", markersize=8)
    ax.fill_between(timesteps, cv, alpha=0.3, color="orange")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Threshold")
    ax.set_ylabel("Coefficient of Variation", fontsize=11)
    ax.set_xlabel("Training Steps")
    ax.set_title("Policy Stability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("models_sac_throwing/detailed_throwing_analysis.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved detailed analysis: models_sac_throwing/detailed_throwing_analysis.png")


def main():
    """Run analysis."""
    results = analyze_throwing_results()
    plot_throwing_analysis()

    # Print improvement suggestions
    if results:
        print("\n" + "=" * 80)
        print("9. IMPROVEMENT RECOMMENDATIONS")
        print("=" * 80)

        stability = results["stability"]
        improvement = results["improvement"]

        print("\n⚡ Based on the analysis above, try these improvements:\n")

        improvements = []

        if stability > 1.0:
            improvements.append(
                "1. REDUCE EXPLORATION:\n"
                "   - Lower ent_coef from 'auto' to 0.05\n"
                "   - Reduces random exploration once good policy is found"
            )

        if improvement < 0.5:
            improvements.append(
                "2. INCREASE TRAINING DURATION:\n"
                "   - Extend total_timesteps from 100k to 200k\n"
                "   - Current policy needs more learning time"
            )

        improvements.append(
            "3. IMPROVE REWARD SHAPING:\n"
            "   - Currently: -distance + bonus\n"
            "   - Try: -distance^2 (quadratic penalty for larger errors)"
        )

        improvements.append(
            "4. ADJUST NETWORK SIZE:\n"
            "   - Current: [256, 256]\n"
            "   - Try: [512, 512] if not enough capacity\n"
            "   - Or [128, 128] if overfitting occurs"
        )

        improvements.append(
            "5. CURRICULUM LEARNING:\n"
            "   - Start with near targets (0.5-2.0m)\n"
            "   - Gradually increase range to 5.0m\n"
            "   - Progressive difficulty improves convergence"
        )

        improvements.append(
            "6. BATCH SIZE TUNING:\n"
            "   - Current: 256\n"
            "   - Try: 128 (less stable but faster)\n"
            "   - Or: 512 (more stable but slower)"
        )

        print("\n".join(improvements))

        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
