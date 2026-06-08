"""
Base kinematics class. All kinematics classes should inherit from this one.
"""

from abc import ABC, abstractmethod

import numpy as np


class Kinematics(ABC):

    def __init__(self):
        # TODO : in the futur, should take an urdf to derive the dynamics directly
        pass

    @abstractmethod
    def jacobian(self, q) -> np.ndarray:
        """
        Compute the Jacobian matrix of the kinematic chain at the given joint angles.
        q: joint angles
        Returns: J(q) Jacobian matrix
        """

    @abstractmethod
    def djacobian(self, q, qd) -> np.ndarray:
        """
        Compute the time derivative of the Jacobian matrix of the kinematic chain at the given joint angles and velocities.
        q: joint angles
        qd: joint velocities
        Returns: dJ(q, dq) time derivative of the Jacobian matrix
        """

    @abstractmethod
    def forward_kinematics(self, q) -> np.ndarray:
        """
        Compute the forward kinematics of the robot at the given joint angles.
        q: joint angles
        Returns: end-effector position (and orientation ??)
        """
