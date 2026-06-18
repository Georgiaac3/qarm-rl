"""
RL training script for bin picking policy with SAC (Soft Actor-Critic).

Uses Stable-Baselines3 to train a policy that directly optimizes for:
- Picking objects from bin (grasp rewards)
- Throwing them out (distance rewards)
- Clearing the bin efficiently

The policy learns end-to-end from pixels (RGB + Depth + Heatmap)
to pixel-wise grasping actions.

Usage:
    python3 training/train.py                      # Train for 100k steps
    python3 training/train.py --steps 500000       # Train for 500k steps
    python3 training/train.py --eval-freq 2000     # Evaluate every 2k steps
"""

import sys
from pathlib import Path
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.bin_picking_env import BinPickingEnv
from config import get_config


def train_rl(
    total_timesteps: int = 100_000,
    eval_freq: int = 5_000,
    save_freq: int = 10_000,
):
    """
    Train bin picking policy with SAC (Soft Actor-Critic).
    
    Args:
        total_timesteps: Total environment steps to train
        eval_freq: Frequency of evaluation episodes
        save_freq: Frequency of checkpoint saves
    """
    config = get_config(device="auto")
    
    print("\n" + "="*80)
    print("BIN PICKING WITH VARIABLE OBJECTS - RL TRAINING (SAC)")
    print("="*80)
    
    # Device
    device = config['model']['device']
    print(f"Device: {device}")
    
    # Create training environment
    print("\nInitializing environment...")
    env = BinPickingEnv(
        image_height=config['env']['image_height'],
        image_width=config['env']['image_width'],
        max_steps=config['env']['max_steps'],
        n_objects=config['env']['n_objects'],
        camera_fov=config['env'].get('camera_fov', 60.0),
    )
    
    # Wrap with Monitor for tracking
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = BinPickingEnv(
        image_height=config['env']['image_height'],
        image_width=config['env']['image_width'],
        max_steps=config['env']['max_steps'],
        n_objects=config['env']['n_objects'],
        camera_fov=config['env'].get('camera_fov', 60.0),
    )
    eval_env = Monitor(eval_env)
    
    # Create model directory
    model_dir = Path(__file__).parent / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print("MODEL CONFIGURATION")
    print(f"{'='*80}")
    print(f"Learning rate:           {config['training']['learning_rate']}")
    print(f"Batch size:              {config['training']['batch_size']}")
    print(f"Total timesteps:         {total_timesteps:,}")
    print(f"Eval frequency:          {eval_freq:,}")
    print(f"Checkpoint frequency:    {save_freq:,}")
    
    # Create SAC agent
    print("\nCreating SAC agent...")
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=config['training']['learning_rate'],
        batch_size=config['training']['batch_size'],
        buffer_size=50_000,
        learning_starts=1_000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device=device,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir),
        name_prefix="rl_policy",
        save_replay_buffer=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(model_dir / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=False,
    )
    
    # Train
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    # Save final model
    final_path = model_dir / "rl_policy_final.zip"
    model.save(str(final_path))
    print(f"\n✓ Final model saved to {final_path}")
    
    # Test on evaluation environment
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    
    test_episodes = 5
    episode_rewards = []
    
    for episode in range(test_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    print(f"\nMean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    
    env.close()
    eval_env.close()
    
    print(f"\n✓ Training complete!")
    print(f"Checkpoints saved to: {model_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train bin picking policy with SAC")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total training timesteps (default: 100k)")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                        help="Evaluation frequency (default: 5k steps)")
    parser.add_argument("--save-freq", type=int, default=10_000,
                        help="Checkpoint save frequency (default: 10k steps)")
    
    args = parser.parse_args()
    
    train_rl(
        total_timesteps=args.steps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )
