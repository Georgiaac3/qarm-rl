"""
"""

from abc import ABC, abstractmethod


class Kinematics(ABC):

    def __init__(self):
        # TODO : in the futur, should take an urdf to derive the dynamics directly
        pass

    @abstractmethod
    def jacobian(self, q):
        """ """

    @abstractmethod
    def djacobian(self, q, qd):
        """ """

    @abstractmethod
    def forward_kinematics(self, q):
        """ """
