import time
from typing import Optional, Tuple

import numpy as np

from core.qarm.interface import QARMInterface


class QArmEnvironment:
    def __init__(
        self,
        qarm_interface: QARMInterface,
        max_steps: int = 1000,
        target_position: Optional[np.ndarray] = None,
        action_scale: float = 0.1,
        dt: float = 0.002,
    ):
        self.qarm = qarm_interface
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.dt = dt

        self.state_dim = 7  # 4 joint angles + 3 target coordinates
        self.action_dim = 4  # 4 velocity corrections

        if target_position is None:
            # Generate random target position within workspace bounds
            self.target_position = self._sample_target()
        else:
            self.target_position = np.array(target_position)

        self.current_step = 0
        self.episode_return = 0.0

    def _sample_target(self) -> np.ndarray:
        """
        Generate a random target position within the robot's workspace.

        Returns:
            target: 3D position [x, y, z]
        """
        x = np.random.uniform(0.3, 0.6)  # X reach (meters)
        y = np.random.uniform(-0.3, 0.3)  # Y reach
        z = np.random.uniform(0.1, 0.5)  # Z height

        return np.array([x, y, z])

    def _get_end_effector_position(self) -> np.ndarray:
        """
        Compute the Cartesian position of the end effector using forward kinematics.

        This function converts joint angles to 3D position [x, y, z].

        Implementation options:
        1. Denavit-Hartenberg (DH) parameters: Manual calculation
        2. PyBullet: physics simulation with built-in FK
        3. MuJoCo: high-performance physics engine with FK
        4. ROS TF: transformation library (if using ROS)

        PyBullet example:
            import pybullet as p
            link_state = p.getLinkState(robot_id, end_effector_link_id)
            position = link_state[0]  # world position

        MuJoCo example:
            import mujoco
            mujoco.mj_fwdPosition(model, data)
            position = data.site_xpos[end_effector_site_id]

        Returns:
            position: 3D position of the end effector
        """
        # TODO: Implement actual forward kinematics for QArm
        # Currently returns a placeholder position
        # Replace with DH transformations, PyBullet, or MuJoCo

        angles = self.qarm.read_angles()

        # PLACEHOLDER: Simplified forward kinematics
        # Replace this with your robot's actual kinematics!
        x = 0.5 * np.cos(angles[0]) * (np.sin(angles[1]) + np.sin(angles[2]))
        y = 0.5 * np.sin(angles[0]) * (np.sin(angles[1]) + np.sin(angles[2]))
        z = 0.3 + 0.2 * np.cos(angles[1]) + 0.2 * np.cos(angles[2])

        return np.array([x, y, z])

    def _get_state(self) -> np.ndarray:
        """
        Get the current environment state.

        State = [robot joint angles, target position]

        Returns:
            state: State vector (numpy array)
        """
        angles = np.array(self.qarm.read_angles())
        state = np.concatenate([angles, self.target_position])

        return state.astype(np.float32)

    def _compute_reward(self) -> float:
        """
        Compute reward based on distance to target.

        Reward: r = -||p_end_effector - p_target||

        A smaller distance gives a less negative reward
        (closer to 0), which is better.

        Returns:
            reward: Reward value
        """
        # Get current end effector position
        end_effector_pos = self._get_end_effector_position()

        # Calculate Euclidean distance to target
        distance = float(np.linalg.norm(end_effector_pos - self.target_position))

        # Reward = negative distance
        reward = -distance

        # Bonus if very close to target
        if distance < 0.05:  # 5 cm
            reward += 10.0  # Bonus for reaching target

        return reward

    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state
        """
        self.current_step = 0
        self.episode_return = 0.0

        # Optional: generate a new target for each episode
        # Uncomment if you want varied targets
        # self.target_position = self._sample_target()

        # Reset robot to starting position
        # For Gazebo, you could use a ROS reset_world service
        # Currently assumes robot is already in initial position

        # Wait for robot to stabilize
        time.sleep(0.1)

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute an action in the environment.
        """

        scaled_action = action * self.action_scale

        # Send command to robot
        # grip=0 since we don't handle gripper for now
        self.qarm.send_speeds(scaled_action.tolist(), grip=0)

        # Wait for time step
        time.sleep(self.dt)

        # Observe new state
        next_state = self._get_state()

        # Calculate reward
        reward = self._compute_reward()

        # Update counters
        self.current_step += 1
        self.episode_return += reward

        # Check if episode is done
        done = self._is_done()

        # Additional information for logging
        info = {
            "step": self.current_step,
            "episode_return": self.episode_return,
            "end_effector_pos": self._get_end_effector_position(),
            "target_pos": self.target_position,
        }

        return next_state, reward, done, info

    def _is_done(self) -> bool:
        """
        Determine if the episode is finished.

        Termination conditions:
        1. Maximum number of steps reached
        2. Target reached (distance < threshold)
        3. (Optional) Robot out of bounds or joint limits exceeded

        Returns:
            True if episode should terminate, False otherwise
        """
        # Termination by step count
        if self.current_step >= self.max_steps:
            return True

        # Termination by success (target reached)
        distance = np.linalg.norm(self._get_end_effector_position() - self.target_position)
        if distance < 0.02:  # 2 cm precision
            return True

        # Can add more termination conditions here
        # e.g., if robot goes out of bounds, joint limits violated, etc.

        return False

    def close(self):
        self.qarm.close()
