"""
This module defines the base class for robot dynamics.
Convention :
- q, dq, ddq are the angles, velocities and accelerations used for dynamic calculations. They are derived from the measured angles phi, velocities dphi and accelerations ddphi through the transform_angles method.
- phi, dphi, ddphi are the measured angles, velocities and accelerations of the robot's motors angle after the gear reduction : this is not the drive shaft angles.
"""

from abc import ABC, abstractmethod

from robot_control.utils import Matrix4x4, Matrix4x6, Vector4x1, Vector6x1


class Dynamics(ABC):
    """
    Base class for robot dynamics.
    """

    def __init__(self):
        # TODO : in the futur, should take an urdf to derive the dynamics directly
        pass

    @abstractmethod
    def transform_angles(self, phi: Vector4x1, dphi: Vector4x1, ddphi: Vector4x1) -> tuple:
        """
        Converts the measured angles phi into angles q used for dynamic calculations. This is useful for cases where the robot's angles are not directly the joint angles, such as when using a tendon-driven mechanism. The method should also convert the measured angular velocities dphi and accelerations ddphi into dq and ddq respectively.
        phi: measured angles
        dphi: measured angular velocities
        ddphi: measured angular accelerations
        Returns:
            q, dq, ddq
        """

    @abstractmethod
    def inertia_matrix(self, q: Vector4x1, mL: float = 0) -> Matrix4x4:
        """
        Computes the inertia matrix M(q) of the robot at the given joint configuration q.
        q: joint angles for dynamic calculations
        mL: mass of the load at the end-effector, if applicable (default is 0 for no load)
        Returns:
            M(q): inertia matrix
        """

    @abstractmethod
    def coriolis_matrix(self, q: Vector4x1, mL: float = 0) -> Matrix4x6:
        """
        Computes the Coriolis matrix B(q) of the robot at the given joint configuration q and joint velocities dq. It should not be confused with C(q, dq) which is the Coriolis and centrifugal forces vector.
        Usage : B @ B_signals
        q: joint angles for dynamic calculations
        mL: mass of the load at the end-effector, if applicable (default is 0 for no load)
        Returns:
            B(q): Coriolis matrix
        """

    @abstractmethod
    def coriolis_velocity_signals(self, dq: Vector4x1, mL: float = 0) -> Vector6x1:
        """
        Computes the Coriolis signals [dq_i, dq_j] for all pairs of joints i, j.
        Usage : B @ B_signals
        dq: joint velocities for dynamic calculations
        mL: mass of the load at the end-effector, if applicable (default is 0 for no load)
        Returns:
            B_signals(dq): Coriolis signals
        """

    @abstractmethod
    def centrifugal_matrix(self, q: Vector4x1, mL: float = 0) -> Matrix4x4:
        """
        Computes the centrifugal matrix C(q) of the robot at the given joint configuration q. It should not be confused with B(q) which is the Coriolis matrix.
        Usage : C @ dq_geo_mes**2
        q: joint angles for dynamic calculations
        mL: mass of the load at the end-effector, if applicable (default is 0 for no load)
        Returns:
            C(q): Centrifugal matrix
        """

    @abstractmethod
    def gravity_vector(self, q: Vector4x1, mL: float = 0) -> Vector4x1:
        """
        Computes the gravity vector G(q) of the robot at the given joint configuration q.
        q: joint angles for dynamic calculations
        mL: mass of the load at the end-effector, if applicable (default is 0 for no load)
        Returns:
            G(q): Gravity vector
        """

    @abstractmethod
    def friction_vector(self, dq: Vector4x1) -> Vector4x1:
        """
        Computes the friction vector F(dq) of the robot at the given joint velocities dq.
        dq: joint velocities for dynamic calculations
        Returns:
            F(dq): Friction vector
        """
