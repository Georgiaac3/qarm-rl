from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from core.qarm.sim import QARMSim
from reinforcement_learning.gym_environment import QArmGymEnv
from utils.logger import logger


def make_qarm_env():
    """
    Create and wrap QArm environment.
    """
    qarm = QARMSim()

    # Create environment
    env = QArmGymEnv(
        qarm_interface=qarm,
        max_steps=1000,
        target_position=None,  # Random targets
        action_scale=0.1,
        dt=0.002,
    )

    # Wrap with Monitor for logging
    env = Monitor(env)

    return env


def train_sac(
    total_timesteps: int = 100_000,
    learning_rate: float = 3e-4,
    save_dir: str = "models/sac",
):
    """
    Train SAC agent on QArm environment.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Training SAC on QArm with Stable Baselines3")
    logger.info("=" * 80)

    # Create environments
    env = make_qarm_env()
    eval_env = make_qarm_env()

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        policy_kwargs={
            "net_arch": [128],  # Single hidden layer of 128 units
        },
        verbose=1,
        tensorboard_log=save_path / "tensorboard",
    )

    logger.info("Model created successfully")
    logger.info(f"Policy architecture: {model.policy}")

    # Create callbacks
    # 1. Checkpoint callback - saves model every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=save_path / "checkpoints",
        name_prefix="sac_qarm",
    )

    # 2. Evaluation callback - evaluates and saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path / "best_model",
        log_path=save_path / "eval_logs",
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    logger.info("Starting training...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
        logger.info("Training completed successfully!")

        # Save final model
        final_model_path = save_path / "final_model"
        model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        # Save model on keyboard interrupt - could implement continue training later function
        interrupt_model_path = save_path / "interrupted_model"
        model.save(interrupt_model_path)
        logger.info(f"Model saved to {interrupt_model_path}")

    finally:
        env.close()
        eval_env.close()
        logger.info("Environments closed")

    return model


if __name__ == "__main__":
    train_sac(
        total_timesteps=100_000,
        learning_rate=3e-4,
        save_dir="models/sac",
    )
