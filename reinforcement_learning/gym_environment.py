import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.qarm.interface import QARMInterface


class QArmGymEnv(gym.Env):
    """
    Custom Gymnasium environment for QArm robotic arm control.

    Observation Space: Box(7,)
        - 4 joint angles (radians)
        - 3 target coordinates (x, y, z in meters)

    Action Space: Box(4,)
        - 4 velocity corrections for each joint (bounded in [-1, 1])

    Reward:
        - Negative distance to target: r = -||end_effector - target||
        - Bonus of +10 when within 5cm of target
        - Episode ends when within 2cm or max_steps reached
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        qarm_interface: QARMInterface,
        max_steps: int = 1000,
        target_position: Optional[np.ndarray] = None,
        action_scale: float = 0.1,
        dt: float = 0.002,
    ):
        """
        Initialize QArm Gym environment.

        Args:
            qarm_interface: Interface to QArm
            max_steps: Maximum steps per episode
            target_position: Fixed target position [x, y, z] or None for random
            action_scale: Scaling factor for actions
            dt: Time step between actions (seconds)
        """
        super().__init__()

        self.qarm = qarm_interface
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.dt = dt
        self.fixed_target = target_position

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # State: [angle1, angle2, angle3, angle4, target_x, target_y, target_z]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Initialize state variables
        self.target_position = None
        self.current_step = 0

    def _sample_target(self) -> np.ndarray:
        """Generate random target position within workspace bounds."""
        x = np.random.uniform(0.3, 0.6)
        y = np.random.uniform(-0.3, 0.3)
        z = np.random.uniform(0.1, 0.5)
        return np.array([x, y, z], dtype=np.float32)

    def _get_end_effector_position(self) -> np.ndarray:
        """
        Get end effector position using forward kinematics.

        TODO: Implement actual forward kinematics
        Options:
            1. Manual calculation with DH parameters
            2. PyBullet URDF
            3. MuJoCo XML
            4. ROS TF

        Returns:
            position: [x, y, z] in meters
        """
        # PLACEHOLDER: Replace with actual forward kinematics
        angles = self.qarm.read_angles()
        # Simplified approximation (NOT ACCURATE)
        x = 0.4 + 0.2 * np.sin(angles[0]) * np.cos(angles[1])
        y = 0.2 * np.sin(angles[0]) * np.sin(angles[1])
        z = 0.3 + 0.2 * np.cos(angles[1])
        return np.array([x, y, z], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Get current observation (state)."""
        angles = np.array(self.qarm.read_angles(), dtype=np.float32)
        obs = np.concatenate([angles, self.target_position])
        return obs

    def _compute_reward(self) -> float:
        """Compute reward based on distance to target."""
        end_effector_pos = self._get_end_effector_position()
        distance = float(np.linalg.norm(end_effector_pos - self.target_position))

        reward = -distance

        # Bonus for reaching target
        if distance < 0.05:  # Within 5 cm
            reward += 10.0

        return reward

    def _is_terminated(self) -> bool:
        """Check if episode is terminated (goal reached)."""
        end_effector_pos = self._get_end_effector_position()
        distance = float(np.linalg.norm(end_effector_pos - self.target_position))
        return distance < 0.02  # Within 2 cm

    def _is_truncated(self) -> bool:
        """Check if episode is truncated (max steps reached)."""
        return self.current_step >= self.max_steps

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Set or sample target position
        if self.fixed_target is not None:
            self.target_position = self.fixed_target.astype(np.float32)
        else:
            self.target_position = self._sample_target()

        self.current_step = 0
        time.sleep(0.1)  # Let robot stabilize

        observation = self._get_obs()
        assert self.target_position is not None
        info = {
            "target_position": self.target_position.copy(),
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment.

        Args:
            action: Action vector (4D)

        Returns:
            observation: Next state
            reward: Reward value
            terminated: Whether episode is terminated (goal reached)
            truncated: Whether episode is truncated (max steps)
            info: Additional information
        """
        # Scale and apply action
        scaled_action = action * self.action_scale
        self.qarm.send_speeds(scaled_action.tolist(), grip=0)
        time.sleep(self.dt)

        # Get new state
        self.current_step += 1
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        # Additional info
        info = {
            "step": self.current_step,
            "end_effector_pos": self._get_end_effector_position(),
            "target_pos": self.target_position,
            "distance": np.linalg.norm(self._get_end_effector_position() - self.target_position),
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        self.qarm.close()
