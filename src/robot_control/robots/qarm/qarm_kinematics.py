"""
QArm Kinematics module.
"""

import numpy as np

from robot_control import Kinematics
from robot_control.utils import get_trig_values4

from .qarm_convertor import QArmConvertor
from .qarm_data import QArmData


class QArmKinematics(Kinematics, QArmData, QArmConvertor):
    def __init__(self):
        super().__init__()
        # TODO : should take an urdf in the futur to derive everything (in the base class Kinematics)

    def jacobian(self, q):
        """
        Calcule le jacobien (J) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1, s1, c2, s2, c3, s3, c23, s23 = self.convert_trig_values(*get_trig_values4(q))

        J = np.array(
            [
                [
                    -self.l2 * s1 * c2 + self.l3 * s1 * s23,
                    -self.l2 * c1 * s2 - self.l3 * c1 * c23,
                    -self.l3 * c1 * c23,
                    0,
                ],
                [
                    self.l2 * c1 * c2 - self.l3 * c1 * s23,
                    -self.l2 * s1 * s2 - self.l3 * s1 * c23,
                    -self.l3 * s1 * c23,
                    0,
                ],
                [0, -self.l2 * c2 + self.l3 * s23, self.l3 * s23, 0],
            ]
        )

        return J

    def djacobian(self, q, dq):
        """
        Calcule la dérivée du jacobien (dJ) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - dq : vecteur colonne des vitesses articulaires de la dynamique géométrique (4,1)
        - retourne : dérivée du jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1) or dq.shape != (4, 1):
            raise ValueError("q et dq doivent être des vecteurs colonne de dimension (4, 1)")

        dq1 = dq[0, 0]
        dq2 = dq[1, 0]
        dq3 = dq[2, 0]

        c1, s1, c2, s2, c3, s3, c23, s23 = self.convert_trig_values(*get_trig_values4(q))

        dJ = np.array(
            [
                [
                    (-self.l2 * c1 * c2 + self.l3 * c1 * s23) * dq1
                    + (self.l2 * s1 * s2 + self.l3 * s1 * c23) * dq2
                    + (self.l3 * s1 * c23) * dq3,
                    (self.l2 * s1 * s2 + self.l3 * s1 * c23) * dq1
                    + (-self.l2 * c1 * c2 + self.l3 * c1 * s23) * dq2
                    + (self.l3 * c1 * s23) * dq3,
                    self.l3 * s1 * c23 * dq1
                    + (self.l3 * c1 * s23) * dq2
                    + (self.l3 * c1 * s23) * dq3,
                    0,
                ],
                [
                    (-self.l2 * s1 * c2 + self.l3 * s1 * s23) * dq1
                    + (-self.l2 * c1 * s2 - self.l3 * c1 * c23) * dq2
                    + (-self.l3 * c1 * c23) * dq3,
                    (-self.l2 * c1 * s2 - self.l3 * c1 * c23) * dq1
                    + (-self.l2 * s1 * c2 + self.l3 * s1 * s23) * dq2
                    + (self.l3 * s1 * s23) * dq3,
                    -self.l3 * c1 * c23 * dq1
                    + (self.l3 * s1 * s23) * dq2
                    + (self.l3 * s1 * s23) * dq3,
                    0,
                ],
                [
                    0,
                    (self.l2 * s2 + self.l3 * c23) * dq2 + (self.l3 * c23) * dq3,
                    (self.l3 * c23) * dq2 + (self.l3 * c23) * dq3,
                    0,
                ],
            ]
        )

        return dJ

    def forward_kinematics(self, q):
        """
        Calcule la position cartésienne de l'effecteur en fonction des angles articulaires q.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : position cartésienne de l'effecteur (3,1)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """
        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1, s1, c2, s2, c3, s3, c23, s23 = self.convert_trig_values(*get_trig_values4(q))

        x = c1 * (self.l2 * c2 - self.l3 * s23)
        y = s1 * (self.l2 * c2 - self.l3 * s23)
        z = self.l1 - self.l2 * s2 - self.l3 * c23
        return np.array([[x], [y], [z]])
