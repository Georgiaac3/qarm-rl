"""Training script for Octopus Arm environment."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from env import OctopusArmEnv


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train agent on Octopus Arm")
    parser.add_argument("--algorithm", type=str, default="PPO",
                        choices=["PPO", "SAC", "DDPG"],
                        help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100000,
                        help="Total timesteps to train")
    parser.add_argument("--num_envs", type=int, default=4,
                        help="Number of parallel environments")
    parser.add_argument("--num_segments", type=int, default=3,
                        help="Number of arm segments")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cuda", "cpu", "auto"],
                        help="Device to use for training")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard logs directory
    tb_log_dir = Path("./tensorboard")
    tb_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training {args.algorithm} agent on Octopus Arm")
    print(f"  - Timesteps: {args.timesteps}")
    print(f"  - Parallel environments: {args.num_envs}")
    print(f"  - Arm segments: {args.num_segments}")
    print(f"  - Output directory: {output_dir}")
    
    # Create vectorized environment with monitoring
    def make_env():
        env = OctopusArmEnv(num_segments=args.num_segments)
        env = Monitor(env)
        return env
    vec_env = make_vec_env(make_env, n_envs=args.num_envs)
    # Create algorithm
    algo_name = args.algorithm.lower()
    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=args.device,
            learning_rate=3e-4,
            n_steps=128,
            tensorboard_log=str(tb_log_dir),
        )
    elif args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=args.device,
            learning_rate=3e-4,
            tensorboard_log=str(tb_log_dir),
        )
    elif args.algorithm == "DDPG":
        model = DDPG(
            "MlpPolicy",
            vec_env,
            verbose=1,
            device=args.device,
            learning_rate=1e-3,
            tensorboard_log=str(tb_log_dir),
        )
    
    # Create callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir),
        name_prefix=f"octopus_arm_{algo_name}",
        save_replay_buffer=True,
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
        tb_log_name=algo_name,
    )

    # Save final model
    final_path = output_dir / f"octopus_arm_{algo_name}_final"
    model.save(str(final_path))
    print(f"\nTraining completed! Model saved to {final_path}")
    print(f"TensorBoard logs in: {tb_log_dir}")


if __name__ == "__main__":
    main()