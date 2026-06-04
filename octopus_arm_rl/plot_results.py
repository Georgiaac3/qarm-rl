"""Plot training results and compare algorithms."""

from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, DDPG

from env import OctopusArmEnv
import re


def plot_training_results(checkpoint_dir="./checkpoints"):
    """Plot training results from SB3 logging."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Collect results by algorithm
    results = defaultdict(lambda: {"steps": [], "rewards": [], "models": []})
    
    # Find all checkpoint files
    checkpoints = sorted(checkpoint_dir.glob("octopus_arm_*.zip"))
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Evaluating models...")
    print()
    
    # Evaluate each checkpoint
    for checkpoint_path in checkpoints:
        model_name = checkpoint_path.stem
        
        # Extract algorithm and step number
        if "ppo" in model_name:
            algo = "PPO"
        elif "sac" in model_name:
            algo = "SAC"
        elif "ddpg" in model_name:
            algo = "DDPG"
        else:
            continue
        
        # Extract step number from filename
        try:
            if "_final" in model_name:
                continue
            else:
                match = re.search(r"_(\d+)_steps$", model_name)

                if match:
                    step = int(match.group(1))
                else:
                    print(f"Impossible d'extraire le step de {model_name}")
                    continue
        except:
            continue
        
        # Load and evaluate model
        try:
            if algo == "PPO":
                model = PPO.load(str(checkpoint_path))
            elif algo == "SAC":
                model = SAC.load(str(checkpoint_path))
            elif algo == "DDPG":
                model = DDPG.load(str(checkpoint_path))
            
            # Evaluate on 500 episodes
            env = OctopusArmEnv(num_segments=3)
            episode_rewards = []
            
            for _ in range(500):
                obs, _ = env.reset()
                ep_reward = 0
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                
                episode_rewards.append(ep_reward)
            
            avg_reward = np.mean(episode_rewards)
            
            results[algo]["steps"].append(step)
            results[algo]["rewards"].append(avg_reward)
            results[algo]["models"].append(checkpoint_path.name)
            
            print(f"  {model_name}: reward={avg_reward:.2f}")
        
        except Exception as e:
            print(f"  ⚠️  Error loading {model_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("Plotting results...")
    print("="*60 + "\n")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Reward over steps for each algorithm
    ax1 = axes[0]
    colors = {"PPO": "blue", "SAC": "green", "DDPG": "red"}
    
    for algo, color in colors.items():
        if results[algo]["steps"]:
            steps = np.array(results[algo]["steps"])
            rewards = np.array(results[algo]["rewards"])
            
            # Sort by steps
            sort_idx = np.argsort(steps)
            steps = steps[sort_idx]
            rewards = rewards[sort_idx]
            
            ax1.plot(steps, rewards, marker="o", label=algo, color=color, linewidth=2, markersize=6)
    
    ax1.set_xlabel("Training Timesteps", fontsize=12)
    ax1.set_ylabel("Mean Evaluation Reward (500 episodes)", fontsize=12)
    ax1.set_title(
        "Evaluation Reward vs Training Timesteps",
        fontsize=14,
        fontweight="bold"
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart comparing final performance
    ax2 = axes[1]
    
    algo_names = []
    final_rewards = []
    
    for algo in ["PPO", "SAC", "DDPG"]:
        if results[algo]["rewards"]:
            algo_names.append(algo)
            steps = np.array(results[algo]["steps"])
            rewards = np.array(results[algo]["rewards"])
            idx = np.argmax(steps)
            final_rewards.append(rewards[idx])
                
    if algo_names:
        bars = ax2.bar(algo_names, final_rewards, color=[colors[a] for a in algo_names], alpha=0.7, edgecolor="black", linewidth=2)
        
        # Add value labels on bars
        for bar, reward in zip(bars, final_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{reward:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight="bold")
        
        ax2.set_ylabel("Mean Evaluation Reward (500 episodes)", fontsize=12)
        ax2.set_title("Final Policy Performance", fontsize=14, fontweight="bold")
        ax2.set_ylim(bottom=0)
        ax2.grid(True, axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("training_results.png")
    plt.savefig(output_path)
    print(f"✅ Plot saved: {output_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    
    for algo in ["PPO", "SAC", "DDPG"]:
        if results[algo]["rewards"]:
            rewards = results[algo]["rewards"]
            print(f"\n{algo}:")
            print(f"  Initial reward: {rewards[0]:.2f}")
            print(f"  Final reward: {rewards[-1]:.2f}")
            print(f"  Improvement: {rewards[-1] - rewards[0]:.2f}")
            print(f"  Checkpoints: {len(rewards)}")


def compare_algorithms_detailed(checkpoint_dir="./checkpoints", num_episodes=10):
    """Detailed comparison of algorithms."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find final models
    final_models = {
        "PPO": checkpoint_dir / "octopus_arm_ppo_final.zip",
        "SAC": checkpoint_dir / "octopus_arm_sac_final.zip",
        "DDPG": checkpoint_dir / "octopus_arm_ddpg_final.zip",
    }
    
    print("\n" + "="*60)
    print("🔬 DETAILED ALGORITHM COMPARISON")
    print("="*60 + "\n")
    
    results = {}
    
    for algo, model_path in final_models.items():
        if not Path(model_path).exists():
            print(f"⚠️  {algo} model not found: {model_path}")
            continue
        
        print(f"Evaluating {algo}...")
        
        # Load model
        if algo == "PPO":
            model = PPO.load(str(model_path))
        elif algo == "SAC":
            model = SAC.load(str(model_path))
        elif algo == "DDPG":
            model = DDPG.load(str(model_path))
        
        # Evaluate
        env = OctopusArmEnv(num_segments=3)
        episode_rewards = []
        episode_distances = []
        targets_reached = 0
        timeouts = 0
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            ep_reward = 0
            ep_distances = []
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_distances.append(info["distance"])
                
                if info.get("timeout", False):
                    timeouts += 1
            
            episode_rewards.append(ep_reward)
            episode_distances.append(min(ep_distances))
            
            if min(ep_distances) < 0.1:
                targets_reached += 1
        
        results[algo] = {
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_distance": np.mean(episode_distances),
            "success_rate": targets_reached / num_episodes,
            "timeouts": timeouts / num_episodes,
        }
        
        print(f"  ✅ {algo}: avg_reward={results[algo]['avg_reward']:.2f}, "
              f"success={results[algo]['success_rate']:.1%}")
    
    # Print comparison table
    if results:
        print("\n" + "="*60)
        print("📊 COMPARISON TABLE")
        print("="*60)
        print(f"{'Algorithm':<12} {'Reward':>12} {'Min Dist':>12} {'Success':>12} {'Timeouts':>12}")
        print("-"*60)
        
        for algo in ["PPO", "SAC", "DDPG"]:
            if algo in results:
                r = results[algo]
                print(f"{algo:<12} {r['avg_reward']:>12.2f} {r['min_distance']:>12.4f} "
                      f"{r['success_rate']:>12.1%} {r['timeouts']:>12.1%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot and compare training results")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Path to checkpoint directory")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed comparison of final models")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for detailed evaluation")
    
    args = parser.parse_args()
    
    plot_training_results(args.checkpoint_dir)
    
    if args.detailed:
        compare_algorithms_detailed(args.checkpoint_dir, args.episodes)
