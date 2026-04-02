from typing import Any, Dict

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim
from reinforcement_learning.gym_environment import QArmGymEnv
from utils.logger import logger


def _make_eval_env(use_real_robot: bool) -> Monitor:
    """Create environment for evaluation (real robot or simulation)."""
    qarm = QARMReal() if use_real_robot else QARMSim()
    env = QArmGymEnv(qarm_interface=qarm, max_steps=1000, action_scale=0.1, dt=0.002)
    return Monitor(env)


def evaluate_model(
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    log_frequency: int = 100,
    use_real_robot: bool = True,
) -> Dict[str, Any]:
    """
    Test a trained SAC model on the real robot or simulation.

    Typical usage:
        - Training done on simulation (QARMSim)
        - Evaluation done on real robot (QARMReal) to validate the policy

    Args:
        model_path: Path to saved model (.zip)
        n_episodes: Number of test episodes
        deterministic: If True, use mean action (no exploration)
        log_frequency: Log distance every N steps
        use_real_robot: If True, use QARMReal. If False, use QARMSim.
    """
    logger.info(f"Loading model from {model_path}")
    logger.info(f"Environment: {'Real robot' if use_real_robot else 'Simulation'}")

    # Create environment
    env = _make_eval_env(use_real_robot)

    # Load model
    model = SAC.load(model_path)
    logger.info(f"Model loaded successfully from {model_path}")

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    final_distances = []

    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    logger.info("=" * 80)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        logger.info(f"Episode {episode + 1}/{n_episodes}")
        logger.info(f"Target position: {info['target_position']}")

        while not done:
            # Get action
            action, _ = model.predict(obs, deterministic=deterministic)

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Log progress every log_frequency steps
            if episode_length % log_frequency == 0:
                logger.info(
                    f"  Step {episode_length}: Distance = {info['distance']:.4f}m, "
                    f"Reward = {reward:.4f}"
                )

        # Episode finished
        final_distance = info["distance"]
        success = terminated  # True if goal was reached

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        final_distances.append(final_distance)
        if success:
            success_count += 1

        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Episode length: {episode_length}")
        logger.info(f"  Final distance: {final_distance:.4f}m")
        logger.info(f"  Success: {success}")
        logger.info("-" * 80)

    # statistics
    # TODO - Change into pydantic model
    results = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "mean_distance": np.mean(final_distances),
        "std_distance": np.std(final_distances),
        "success_rate": success_count / n_episodes,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "final_distances": final_distances,
    }

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Episodes: {n_episodes}")
    logger.info(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"Mean episode length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    logger.info(
        f"Mean final distance: {results['mean_distance']:.4f}m ± {results['std_distance']:.4f}m"
    )
    logger.info(f"Success rate: {results['success_rate']*100:.1f}%")
    logger.info("=" * 80)

    env.close()
    return results


if __name__ == "__main__":
    evaluate_model(
        model_path="models/sac/best_model/best_model.zip",
        n_episodes=10,
        deterministic=True,
        use_real_robot=True,  # Switch to False to test in simulation
    )
