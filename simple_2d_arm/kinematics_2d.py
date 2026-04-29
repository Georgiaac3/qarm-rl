"""
2D Arm Kinematics Module
Simple 2-DOF planar manipulator: Shoulder (q1) + Elbow (q2)
"""

from typing import Tuple

import numpy as np


class Arm2D:
    """2D planar arm with 2 DOF."""

    def __init__(self, l1: float = 0.5, l2: float = 0.5):
        """
        Initialize 2D arm.

        Args:
            l1: Length of first segment (m)
            l2: Length of second segment (m)
        """
        self.l1 = l1  # Length of first segment
        self.l2 = l2  # Length of second segment
        self.workspace_radius = l1 + l2

        # Joint limits (radians)
        self.q_min = np.array([-np.pi, -np.pi])
        self.q_max = np.array([np.pi, np.pi])

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector position from joint angles.

        Args:
            q: Joint angles [q1, q2] in radians
               q1 = shoulder angle
               q2 = elbow angle

        Returns:
            pos: End-effector position [x, y] in meters
        """
        q1, q2 = q[0], q[1]

        # Position of elbow joint
        x1 = self.l1 * np.cos(q1)
        y1 = self.l1 * np.sin(q1)

        # Position of end-effector (elbow + forearm)
        # Note: q2 is relative angle, so absolute angle is q1 + q2
        x = x1 + self.l2 * np.cos(q1 + q2)
        y = y1 + self.l2 * np.sin(q1 + q2)

        return np.array([x, y], dtype=np.float32)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Jacobian matrix for velocity mapping.

        Args:
            q: Joint angles [q1, q2]

        Returns:
            J: 2x2 Jacobian matrix (dx/dq)
        """
        q1, q2 = q[0], q[1]

        # Derivative of end-effector position w.r.t. joint angles
        J = np.array(
            [
                [-self.l1 * np.sin(q1) - self.l2 * np.sin(q1 + q2), -self.l2 * np.sin(q1 + q2)],
                [self.l1 * np.cos(q1) + self.l2 * np.cos(q1 + q2), self.l2 * np.cos(q1 + q2)],
            ],
            dtype=np.float32,
        )

        return J

    def inverse_kinematics(
        self, target: np.ndarray, q_init: np.ndarray = None
    ) -> Tuple[np.ndarray, bool]:
        """
        Compute inverse kinematics using analytical solution.

        For a 2D planar arm, analytical solution exists.
        Uses: law of cosines + atan2

        Args:
            target: Target position [x, y]
            q_init: Initial guess (optional, for elbow-up/down ambiguity)

        Returns:
            q: Joint angles [q1, q2]
            success: Boolean indicating if solution exists
        """
        x, y = target[0], target[1]

        # Distance from base to target
        d = np.sqrt(x**2 + y**2)

        # Check if target is reachable
        if d > self.l1 + self.l2 or d < abs(self.l1 - self.l2):
            return np.array([0.0, 0.0], dtype=np.float32), False

        # Law of cosines: compute q2 (elbow angle)
        cos_q2 = (d**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)  # Numerical stability

        # Choose elbow-up configuration (positive q2)
        q2 = np.arccos(cos_q2)

        # Compute q1 (shoulder angle)
        alpha = np.arctan2(y, x)  # Angle to target
        beta = np.arctan2(self.l2 * np.sin(q2), self.l1 + self.l2 * np.cos(q2))
        q1 = alpha - beta

        q = np.array([q1, q2], dtype=np.float32)

        # Ensure within bounds
        q = np.clip(q, self.q_min, self.q_max)

        return q, True

    def velocity_ik(self, x_dot: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute joint velocities from end-effector velocity using Jacobian.

        Args:
            x_dot: End-effector velocity [vx, vy]
            q: Current joint angles [q1, q2]

        Returns:
            q_dot: Joint velocities [q1_dot, q2_dot]
        """
        J = self.jacobian(q)

        # Pseudo-inverse with damping
        lambda_damp = 0.01
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damp**2 * np.eye(2))

        q_dot = J_pinv @ x_dot
        return q_dot

    def is_within_workspace(self, target: np.ndarray) -> bool:
        """Check if target is within workspace."""
        d = np.linalg.norm(target)
        return d <= (self.l1 + self.l2) and d >= abs(self.l1 - self.l2)
