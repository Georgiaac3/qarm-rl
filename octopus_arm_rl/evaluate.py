"""Evaluation script for trained Octopus Arm agent."""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC, DDPG

from env import OctopusArmEnv


def evaluate_agent(model_path, num_episodes=10, num_segments=3, render=True, max_steps=None):
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
        num_segments: Number of arm segments
        render: Whether to render the environment
        max_steps: Override max steps per episode (None = use env default of 50)
    """
    # Load model (auto-detect algorithm from filename)
    model_name = Path(model_path).stem
    if "ppo" in model_name.lower():
        model = PPO.load(model_path)
    elif "sac" in model_name.lower():
        model = SAC.load(model_path)
    elif "ddpg" in model_name.lower():
        model = DDPG.load(model_path)
    else:
        # Try PPO by default
        model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = OctopusArmEnv(num_segments=num_segments, render_mode=render_mode)
    
    # Override max_steps if provided
    if max_steps is not None:
        env.max_steps = max_steps
        print(f"Using custom max_steps: {max_steps}")
    else:
        print(f"Using default max_steps: {env.max_steps} (5 second timer)")
    print()
    
    total_distances = []
    total_rewards = []
    targets_reached = 0
    timeouts = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()  # reset() returns (obs, info) in Gymnasium
        episode_reward = 0
        episode_distances = []
        episode_timeout = False
        
        done = False
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_distances.append(info["distance"])
            
            if info.get("timeout", False):
                episode_timeout = True
            
            if info["target_reached"]:
                targets_reached += 1
            
            if render:
                env.render()
        
        min_distance = min(episode_distances)
        total_distances.append(min_distance)
        total_rewards.append(episode_reward)
        
        # Format status
        status = "✅ SUCCESS" if info["target_reached"] else " TIMEOUT" if episode_timeout else "❌ FAILED"
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Min Dist={min_distance:.4f}, "
              f"Steps={info.get('steps_taken', '?')}/30 {status}")
        
        if episode_timeout:
            timeouts += 1
    
    # Print statistics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Min Distance: {np.mean(total_distances):.4f} ± {np.std(total_distances):.4f}")
    print(f"Targets Reached: {targets_reached}/{num_episodes} ✅")
    print(f"Timeouts: {timeouts}/{num_episodes} ⏱️  ")
    print(f"Success Rate: {targets_reached/num_episodes*100:.1f}%")
    print("="*50)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained Octopus Arm agent")
    parser.add_argument("model_path", type=str, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--num_segments", type=int, default=3,
                        help="Number of arm segments")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per episode (default: 50 with timer)")
    parser.add_argument("--no_render", action="store_true",
                        help="Don't render the environment")
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    evaluate_agent(
        args.model_path,
        num_episodes=args.episodes,
        num_segments=args.num_segments,
        render=not args.no_render,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()
