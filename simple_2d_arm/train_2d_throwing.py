from pathlib import Path

import numpy as np
from env_2d_throwing import Arm2DThrowingEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


class CurriculumCallback:
    """Adjust target distance difficulty during training."""

    def __init__(self, env, initial_range=(0.5, 1.5), final_range=(1.0, 5.0)):
        self.env = env
        self.initial_range = initial_range
        self.final_range = final_range

    def update_difficulty(self, timestep, total_timesteps):
        """Smoothly increase difficulty over training."""
        progress = timestep / total_timesteps

        min_dist = self.initial_range[0] + (self.final_range[0] - self.initial_range[0]) * progress
        max_dist = self.initial_range[1] + (self.final_range[1] - self.initial_range[1]) * progress

        self.env.target_distance_range = (min_dist, max_dist)


def make_env(seed=None, max_distance=None):
    env = Arm2DThrowingEnv(
        max_steps=200,
        target_distance_range=(0.5, 1.5) if max_distance is None else (0.5, max_distance),
        gravity=9.81,
        air_resistance=0.1,
        dt=0.01,
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac_throwing(
    total_timesteps: int = 50_000,
    learning_rate: float = 5e-5,  # Even lower for stability
    save_dir: str = "models_sac_throwing",
    use_curriculum: bool = True,
):
    """
    Train SAC agent on 2D throwing task.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("THROWING TASK TRAINING")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Curriculum learning: {use_curriculum}")
    print(f"Expected runtime: ~1-2 hours")
    print()

    # Create environments
    env = make_env(seed=42)
    eval_env = make_env(seed=43)

    # Create SAC model with improved hyperparameters
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=50_000,  # Smaller buffer = fresher data
        learning_starts=2_000,  # Learn a bit after random exploration
        batch_size=128,  # Smaller batches = more stable
        tau=0.01,  # Slower target network update
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef=0.05,  # FIXED VALUE: much lower than 'auto' for stability, otherwise it was chaotic
        target_entropy=-2.0,
        policy_kwargs={
            "net_arch": [128, 128],
            "activation_fn": lambda: __import__("torch.nn", fromlist=["ReLU"]).ReLU(),
        },
        verbose=1,
        device="auto",
        seed=42,
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path / "checkpoints",
        name_prefix="sac_throwing",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path / "best_model",
        log_path=save_path / "eval_logs",
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Curriculum learning
    curriculum = CurriculumCallback(env) if use_curriculum else None

    # Training
    print("Starting training...\n")
    try:
        for step in range(0, total_timesteps, 5_000):
            # Update curriculum
            if curriculum:
                curriculum.update_difficulty(step, total_timesteps)
                min_d, max_d = env.target_distance_range
                print(f"[Curriculum] Step {step:,}: Target range {min_d:.2f}-{max_d:.2f}m")

            model.learn(
                total_timesteps=5_000,
                callback=[checkpoint_callback, eval_callback],
                log_interval=20,
                progress_bar=True,
            )

        print("\n✓ Training completed successfully!")

        # Save final model
        final_path = save_path / "final_model"
        model.save(str(final_path))
        print(f"✓ Final model saved to {final_path}")

        # Load best model
        best_path = save_path / "best_model" / "best_model.zip"
        if best_path.exists():
            print(f"\n✓ Loading best model from checkpoint...")
            best_model = SAC.load(str(best_path))
            eval_best = make_env(seed=99)
            obs, _ = eval_best.reset()
            print("  Best model loaded successfully")
            eval_best.close()

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
    print("\n" + "=" * 80)
    print("IMPROVED 2D THROWING TRAINING")
    print("=" * 80 + "\n")

    model, save_path = train_sac_throwing(
        total_timesteps=250_000,
        learning_rate=5e-5,
        use_curriculum=True,
    )

    print(f"\n✓ Training complete! Models saved in: {save_path}")
    print("\nNext step:")
    print("  python analyze_throwing_results.py")
