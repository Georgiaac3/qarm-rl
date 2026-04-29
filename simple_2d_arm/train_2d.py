"""
Training script for 2D Arm RL
Uses Stable-Baselines3 SAC algorithm
"""

from pathlib import Path

import numpy as np
from env_2d import Arm2DEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


def make_env(seed=None):
    """Create and wrap environment."""
    env = Arm2DEnv(
        l1=0.5,
        l2=0.5,
        max_steps=100,
        target_radius=1.0,
        dt=0.05,
        ik_scale=1.0,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac(
    total_timesteps: int = 50_000,
    learning_rate: float = 3e-4,
    save_dir: str = "models_sac",
):
    """Train SAC agent on 2D arm."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Training SAC on 2D Arm (Validation Experiment)")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Create environments
    env = make_env(seed=42)
    eval_env = make_env(seed=43)

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        policy_kwargs={
            "net_arch": [64, 64],  # Two hidden layers
        },
        verbose=1,
        tensorboard_log=None,  # Disabled for quick startup (install tensorboard if needed)
    )

    print("Model created successfully")
    print(f"Policy: {model.policy}")
    print()

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path / "checkpoints",
        name_prefix="sac_2d",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path / "best_model",
        log_path=save_path / "eval_logs",
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Train
    print("Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True,
        )
        print("\n✓ Training completed successfully!")

        # Save final model
        final_path = save_path / "final_model"
        model.save(str(final_path))
        print(f"✓ Final model saved to {final_path}")

    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        final_path = save_path / "interrupted_model"
        model.save(str(final_path))
        print(f"✓ Model saved to {final_path}")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise

    finally:
        env.close()
        eval_env.close()

    return model, save_path


if __name__ == "__main__":
    model, save_path = train_sac(
        total_timesteps=50_000,
        learning_rate=3e-4,
    )
    print(f"\n✓ All models saved in: {save_path}")
