import sys
from pathlib import Path

import numpy as np

from core.config import MODE, settings
from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim
from reinforcement_learning.environment import QArmEnvironment
from reinforcement_learning.sac_utils import save_model, test_qarm_sac, train_qarm_sac
from utils.logger import logger


def main():
    logger.info("=" * 70)
    logger.info("SAC TRAINING FOR QARM")
    logger.info("=" * 70)

    # Initialize ROS if simulation mode
    if settings.mode == MODE.SIM:
        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
            logger.info("ROS initialized for simulation")
        except ImportError:
            logger.error("rclpy not available - use real robot mode")
            sys.exit(1)

    # Create environment
    logger.info("Creating QArm environment...")
    qarm = QARMSim() if settings.mode == MODE.SIM else QARMReal()
    env = QArmEnvironment(
        qarm_interface=qarm,
        max_steps=200,
        target_position=np.array([0.4, 0.0, 0.3]),
    )

    # Train
    logger.info("Starting SAC training...")
    policy, q1, q2 = train_qarm_sac(env, num_episodes=500, print_interval=10)

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    save_model(policy, q1, q2, filepath="models/sac_qarm.pth")

    # Test
    logger.info("Testing trained agent...")
    test_qarm_sac(env, policy, num_episodes=3)

    # Cleanup
    env.close()
    if settings.mode == MODE.SIM:
        rclpy.shutdown()
        logger.info("ROS shutdown")

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
