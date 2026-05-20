from pathlib import Path

import numpy as np
from config import get_config
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


def make_env(env_config, seed=None, max_distance=None):
    """Create and wrap environment."""
    initial_range = env_config.get("target_distance_range", (0.5, 1.5))
    if max_distance:
        initial_range = (0.5, max_distance)

    env = Arm2DThrowingEnv(
        max_steps=env_config["max_steps"],
        target_distance_range=initial_range,
        gravity=env_config["gravity"],
        air_resistance=env_config["air_resistance"],
        projectile_radius=env_config["projectile_radius"],
        dt=env_config["dt"],
    )
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def train_sac_throwing(device="auto", save_dir="models_sac_throwing"):
    """
    Train SAC agent on 2D throwing task using configuration parameters.

    Args:
        device: "auto" (detect), "cuda", or "cpu"
        save_dir: Directory to save models
    """
    # Load configuration
    config = get_config(device=device)
    env_cfg = config["env"]
    curr_cfg = config["curriculum"]
    training_cfg = config["training"]
    sac_cfg = config["sac"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("THROWING TASK TRAINING")
    print("=" * 80)
    print(f"Device: {sac_cfg['device']}")
    print(f"Total timesteps: {training_cfg['total_timesteps']:,}")
    print(f"Learning rate: {sac_cfg['learning_rate']}")
    print(f"Batch size: {sac_cfg['batch_size']}")
    print(f"Network: {sac_cfg['net_arch']}")
    print(f"Curriculum learning: {curr_cfg['use_curriculum']}")
    print()

    # Create environments
    env = make_env(env_cfg, seed=42, max_distance=curr_cfg["initial_range"][1])
    eval_env = make_env(env_cfg, seed=43, max_distance=curr_cfg["initial_range"][1])

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        learning_starts=sac_cfg["learning_starts"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        train_freq=sac_cfg["train_freq"],
        gradient_steps=sac_cfg["gradient_steps"],
        ent_coef=sac_cfg["ent_coef"],
        target_entropy=sac_cfg["target_entropy"],
        policy_kwargs={
            "net_arch": sac_cfg["net_arch"],
            "activation_fn": lambda: __import__("torch.nn", fromlist=["ReLU"]).ReLU(),
        },
        verbose=1,
        device=sac_cfg["device"],
        seed=sac_cfg["seed"],
    )

    print(f"Model Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    print()

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=training_cfg["checkpoint_freq"],
        save_path=save_path / "checkpoints",
        name_prefix="sac_throwing",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path / "best_model",
        log_path=save_path / "eval_logs",
        eval_freq=training_cfg["eval_freq"],
        n_eval_episodes=training_cfg["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # Curriculum learning
    curriculum = None
    if curr_cfg["use_curriculum"]:
        curriculum = CurriculumCallback(
            env, initial_range=curr_cfg["initial_range"], final_range=curr_cfg["final_range"]
        )

    # Training
    print(f"Starting training for {training_cfg['total_timesteps']:,} timesteps...\n")
    try:
        # Train with curriculum learning via callback
        if curriculum:
            # Need to update curriculum during training manually
            # Train in chunks with curriculum updates
            for step in range(0, training_cfg["total_timesteps"], 1_000):
                curriculum.update_difficulty(step, training_cfg["total_timesteps"])
                min_d, max_d = env.target_distance_range
                print(f"[Curriculum] Step {step:>7,}: Target range {min_d:.2f}-{max_d:.2f}m")

                model.learn(
                    total_timesteps=1_000,
                    callback=[checkpoint_callback, eval_callback],
                    log_interval=20,
                    progress_bar=True,
                )
        else:
            # No curriculum: single training call (recommended - cleaner eval logging)
            model.learn(
                total_timesteps=training_cfg["total_timesteps"],
                callback=[checkpoint_callback, eval_callback],
                log_interval=100,
                progress_bar=True,
            )

        print("\n✓ Training completed successfully!")

        # Save final model
        final_path = save_path / "final_model"
        model.save(str(final_path))
        print(f"✓ Final model saved to {final_path}")

        # Check best model
        best_path = save_path / "best_model" / "best_model.zip"
        if best_path.exists():
            print(f"✓ Best model available at {best_path}")

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
    print("2D THROWING TASK - SAC TRAINING")
    print("=" * 80 + "\n")

    model, save_path = train_sac_throwing(device="auto")

    print(f"\n✓ Training complete! Models saved in: {save_path}")
    print("\nNext step:")
    print("  cd simple_2d_arm && python analyze_throwing_results.py")
