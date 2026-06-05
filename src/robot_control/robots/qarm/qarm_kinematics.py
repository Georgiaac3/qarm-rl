"""
QArm Kinematics module.
"""

import numpy as np

from robot_control import Kinematics, utils

from .qarm_data import QArmData


class QArmKinematics(Kinematics):
    def __init__(self):
        # TODO : should take an urdf in the futur to derive everything (in the base class Kinematics)
        self.qarm_data = QArmData()

    def get_jacobian(self, q):
        """
        Calcule le jacobien (J) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        J = np.array(
            [
                [
                    -self.qarm_data.l2 * s1 * c2 + self.qarm_data.l3 * s1 * s23,
                    -self.qarm_data.l2 * c1 * s2 - self.qarm_data.l3 * c1 * c23,
                    -self.qarm_data.l3 * c1 * c23,
                    0,
                ],
                [
                    self.qarm_data.l2 * c1 * c2 - self.qarm_data.l3 * c1 * s23,
                    -self.qarm_data.l2 * s1 * s2 - self.qarm_data.l3 * s1 * c23,
                    -self.qarm_data.l3 * s1 * c23,
                    0,
                ],
                [0, -self.qarm_data.l2 * c2 + self.qarm_data.l3 * s23, self.qarm_data.l3 * s23, 0],
            ]
        )

        return J

    def get_djacobian(self, q, dq):
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

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        dJ = np.array(
            [
                [
                    (-self.qarm_data.l2 * c1 * c2 + self.qarm_data.l3 * c1 * s23) * dq1
                    + (self.qarm_data.l2 * s1 * s2 + self.qarm_data.l3 * s1 * c23) * dq2
                    + (self.qarm_data.l3 * s1 * c23) * dq3,
                    (self.qarm_data.l2 * s1 * s2 + self.qarm_data.l3 * s1 * c23) * dq1
                    + (-self.qarm_data.l2 * c1 * c2 + self.qarm_data.l3 * c1 * s23) * dq2
                    + (self.qarm_data.l3 * c1 * s23) * dq3,
                    self.qarm_data.l3 * s1 * c23 * dq1
                    + (self.qarm_data.l3 * c1 * s23) * dq2
                    + (self.qarm_data.l3 * c1 * s23) * dq3,
                    0,
                ],
                [
                    (-self.qarm_data.l2 * s1 * c2 + self.qarm_data.l3 * s1 * s23) * dq1
                    + (-self.qarm_data.l2 * c1 * s2 - self.qarm_data.l3 * c1 * c23) * dq2
                    + (-self.qarm_data.l3 * c1 * c23) * dq3,
                    (-self.qarm_data.l2 * c1 * s2 - self.qarm_data.l3 * c1 * c23) * dq1
                    + (-self.qarm_data.l2 * s1 * c2 + self.qarm_data.l3 * s1 * s23) * dq2
                    + (self.qarm_data.l3 * s1 * s23) * dq3,
                    -self.qarm_data.l3 * c1 * c23 * dq1
                    + (self.qarm_data.l3 * s1 * s23) * dq2
                    + (self.qarm_data.l3 * s1 * s23) * dq3,
                    0,
                ],
                [
                    0,
                    (self.qarm_data.l2 * s2 + self.qarm_data.l3 * c23) * dq2
                    + (self.qarm_data.l3 * c23) * dq3,
                    (self.qarm_data.l3 * c23) * dq2 + (self.qarm_data.l3 * c23) * dq3,
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

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        x = c1 * (self.qarm_data.l2 * c2 - self.qarm_data.l3 * s23)
        y = s1 * (self.qarm_data.l2 * c2 - self.qarm_data.l3 * s23)
        z = self.qarm_data.l1 - self.qarm_data.l2 * s2 - self.qarm_data.l3 * c23
        return np.array([[x], [y], [z]])
