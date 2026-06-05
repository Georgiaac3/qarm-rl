"""
Module de calcul de la dynamique du QArm.
"""

import numpy as np

from robot_control import Dynamics, utils

from .qarm_data import QArmData


class QArmDynamics(Dynamics):
    """
    Classe de calcul de la dynamique du QArm.
    """

    def __init__(self):
        super().__init__()
        # TODO : should take an urdf in the futur to derive everything (in the base class Dynamics)
        self.qarm_data = QArmData()

    def convert_trig_values(self, c1, s1, c2, s2, c3, s3, c23, s23):
        """
        Convertit des valeurs trigonométriques de la convention old vers la convention new.

        Retourne:
            Une liste dans l'ordre (c1_new, s1_new, c2_new, s2_new, c3_new, s3_new, c23_new, s23_new).
        """
        # old -> new:
        # c2_old = s2_new, s2_old = -c2_new
        # c23_old = s23_new, s23_old = -c23_new
        return [c1, s1, s2, -c2, c3, s3, s23, -c23]

    def transform_angles(self, phi, phi_d, phi_dd):
        if phi.shape != (4, 1) or phi_d.shape != (4, 1) or phi_dd.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        q = phi + np.array([[0, self.qarm_data.beta, -self.qarm_data.beta, 0]]).T
        dq = phi_d
        ddq = phi_dd
        return q, dq, ddq

    def get_inertia_matrix(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        M11 = (
            self.qarm_data.I1A
            + self.qarm_data.m2 * (self.qarm_data.l2 - self.qarm_data.lc2) ** 2 * c2**2
            + self.qarm_data.I2A * s2**2
            + self.qarm_data.I2L * c2**2
            + self.qarm_data.m3 * self.qarm_data.l2**2 * c2**2
            + self.qarm_data.m3 * self.qarm_data.lc3**2 * s23**2
            - 2 * self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * c2 * s23
            + self.qarm_data.I3L * s23**2
            + self.qarm_data.I3A * c23**2
            + self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) ** 2 * s23**2
            + self.qarm_data.m4 * self.qarm_data.l2**2 * c2**2
            - 2
            * self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            * self.qarm_data.l2
            * c2
            * s23
            + self.qarm_data.I4L * s23**2
            + self.qarm_data.I4A * c23**2
            + mL * self.qarm_data.l3**2 * s23**2
            + mL * self.qarm_data.l2**2 * c2**2
            - 2 * mL * self.qarm_data.l2 * self.qarm_data.l3 * c2 * s23
        )

        M14 = -self.qarm_data.I4A * c23
        M22 = (
            self.qarm_data.m2 * (self.qarm_data.l2 - self.qarm_data.lc2) ** 2
            + self.qarm_data.I2L
            + self.qarm_data.m3 * self.qarm_data.l2**2
            + self.qarm_data.m3 * self.qarm_data.lc3**2
            - 2 * self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * s3
            + self.qarm_data.I3L
            + self.qarm_data.m4 * (self.qarm_data.l2 + self.qarm_data.l3 - self.qarm_data.lc4) ** 2
            + self.qarm_data.I4L
            + mL * self.qarm_data.l2**2
            + mL * self.qarm_data.l3**2
            - 2 * mL * self.qarm_data.l2 * self.qarm_data.l3 * s3
        )

        M23 = (
            self.qarm_data.m3 * self.qarm_data.lc3**2
            - self.qarm_data.m3 * self.qarm_data.lc3 * self.qarm_data.l2 * s3
            + self.qarm_data.I3L
            + self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4 - self.qarm_data.l2 * s3)
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            + self.qarm_data.I4L
            - mL * (self.qarm_data.l3**2 - self.qarm_data.l2 * self.qarm_data.l3 * s3)
        )

        M33 = (
            self.qarm_data.m3 * self.qarm_data.lc3**2
            + self.qarm_data.I3L
            + self.qarm_data.m4 * (self.qarm_data.lc4 - self.qarm_data.l3) ** 2
            + self.qarm_data.I4L
            + mL * self.qarm_data.l3**2
        )

        M = np.array(
            [
                [M11, 0, 0, M14],
                [0, M22, M23, 0],
                [0, M23, M33, 0],
                [M14, 0, 0, self.qarm_data.I4A],
            ]  # M32 = M23
        )

        return M

    def coriolis_matrix(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        # Calcul des coefficients
        B11 = (
            -2 * self.qarm_data.m2 * (self.qarm_data.l2 - self.qarm_data.lc2) ** 2 * s2 * c2
            + 2 * self.qarm_data.I2A * s2 * c2
            - 2 * self.qarm_data.I2L * s2 * c2
            - 2 * self.qarm_data.m3 * self.qarm_data.l2**2 * s2 * c2
            + 2 * self.qarm_data.m3 * self.qarm_data.lc3**2 * s23 * c23
            + 2 * self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * (s2 * s23 - c2 * c23)
            + 2 * self.qarm_data.I3L * s23 * c23
            - 2 * self.qarm_data.I3A * s23 * c23
            + 2 * self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) ** 2 * s23 * c23
            - 2 * self.qarm_data.m4 * self.qarm_data.l2**2 * s2 * c2
            + 2
            * self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            * self.qarm_data.l2
            * (s2 * s23 - c2 * c23)
            + 2 * self.qarm_data.I4L * s23 * c23
            - 2 * self.qarm_data.I4A * s23 * c23
            + 2 * mL * self.qarm_data.l3**2 * s23 * c23
            - 2 * mL * self.qarm_data.l2**2 * s2 * c2
            + 2 * mL * self.qarm_data.l2 * self.qarm_data.l3 * (s2 * s23 - c2 * c23)
        )

        B12 = (
            2 * self.qarm_data.m3 * self.qarm_data.lc3**2 * s23 * c23
            - 2 * self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * c2 * c23
            + 2 * self.qarm_data.I3L * s23 * c23
            - 2 * self.qarm_data.I3A * s23 * c23
            + 2 * self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) ** 2 * s23 * c23
            - 2
            * self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            * self.qarm_data.l2
            * c2
            * c23
            + 2 * self.qarm_data.I4L * s23 * c23
            - 2 * self.qarm_data.I4A * s23 * c23
            + 2 * mL * self.qarm_data.l3**2 * s23 * c23
            - 2 * mL * self.qarm_data.l2 * self.qarm_data.l3 * c2 * c23
        )

        B15 = self.qarm_data.I4A * s23
        B16 = self.qarm_data.I4A * s23
        B23 = -self.qarm_data.I4A * s23
        B24 = (
            -2 * self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * c3
            - 2 * mL * self.qarm_data.l2 * self.qarm_data.l3 * c3
        )
        B33 = -self.qarm_data.I4A * s23
        B41 = self.qarm_data.I4A * s23
        B42 = self.qarm_data.I4A * s23

        # Assemblage de la matrice B (4 lignes, 6 colonnes)
        B = np.array(
            [
                [B11, B12, 0, 0, B15, B16],
                [0, 0, B23, B24, 0, 0],
                [0, 0, B33, 0, 0, 0],
                [B41, B42, 0, 0, 0, 0],
            ]
        )

        return B

    def coriolis_velocity_signals(self, dq):
        """
        Calcule les signaux de vitesse pour la matrice de Coriolis [dq_i, dq_j].
        dq: vitesses angulaires utilisés pour les calculs dynamiques
        Retourne un array de taille (6, 1) avec les produits de vitesses nécessaires pour la matrice de Coriolis.
        """
        if dq.shape != (4, 1):
            raise ValueError("Les vitesses d'entrée doivent être des np arrays de taille (4, 1).")

        B_signals = np.array(
            [
                dq[0, 0] * dq[1, 0],
                dq[0, 0] * dq[2, 0],
                dq[0, 0] * dq[3, 0],
                dq[1, 0] * dq[2, 0],
                dq[1, 0] * dq[3, 0],
                dq[2, 0] * dq[3, 0],
            ]
        ).reshape((6, 1))
        return B_signals

    def centrifugal_matrix(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        # Calcul des coefficients
        C21 = -(
            -self.qarm_data.m2 * (self.qarm_data.l2 - self.qarm_data.lc2) ** 2 * s2 * c2
            + self.qarm_data.I2A * s2 * c2
            - self.qarm_data.I2L * s2 * c2
            - self.qarm_data.m3 * self.qarm_data.l2**2 * s2 * c2
            + self.qarm_data.m3 * self.qarm_data.lc3**2 * s23 * c23
            + self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * (s2 * s23 - c2 * c23)
            + self.qarm_data.I3L * s23 * c23
            - self.qarm_data.I3A * s23 * c23
            + self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) ** 2 * s23 * c23
            - self.qarm_data.m4 * self.qarm_data.l2**2 * s2 * c2
            + self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            * self.qarm_data.l2
            * (s2 * s23 - c2 * c23)
            + self.qarm_data.I4L * s23 * c23
            - self.qarm_data.I4A * s23 * c23
            + mL * self.qarm_data.l3**2 * s23 * c23
            - mL * self.qarm_data.l2**2 * s2 * c2
            + mL * self.qarm_data.l2 * (self.qarm_data.l3) * (s2 * s23 - c2 * c23)
        )

        C23 = (
            -mL * self.qarm_data.lc3 * self.qarm_data.l2 * c3
            - self.qarm_data.m4 * self.qarm_data.l2 * c3 * (self.qarm_data.l3 - self.qarm_data.lc4)
            + mL * self.qarm_data.l2 * self.qarm_data.l3 * c3
        )

        C31 = -(
            self.qarm_data.m3 * self.qarm_data.lc3**2 * s23 * c23
            - self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * c2 * c23
            + self.qarm_data.I3L * s23 * c23
            - self.qarm_data.I3A * s23 * c23
            + self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) ** 2 * s23 * c23
            - self.qarm_data.m4
            * (self.qarm_data.l3 - self.qarm_data.lc4)
            * self.qarm_data.l2
            * c2
            * c23
            + self.qarm_data.I4L * s23 * c23
            - self.qarm_data.I4A * s23 * c23
            + mL * self.qarm_data.l3**2 * s23 * c23
            - mL * self.qarm_data.l2 * self.qarm_data.l3 * c2 * c23
        )

        C32 = (
            self.qarm_data.m3 * self.qarm_data.l2 * self.qarm_data.lc3 * c3
            + mL * self.qarm_data.l2 * self.qarm_data.l3 * c3
        )

        # Assemblage de la matrice
        C = np.array([[0, 0, 0, 0], [C21, 0, C23, 0], [C31, C32, 0, 0], [0, 0, 0, 0]])
        return C

    def get_gravity_vector(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        # Rappel des raccourcis trigonométriques
        c1, s1, c2, s2, c3, s3, c23, s23 = utils.get_trig_values4(q)

        # Calcul des composantes
        G2 = -self.qarm_data.g * (
            self.qarm_data.m2 * (self.qarm_data.l2 - self.qarm_data.lc2) * c2
            + self.qarm_data.m3 * (self.qarm_data.l2 * c2 - self.qarm_data.lc3 * s23)
            + self.qarm_data.m4
            * (self.qarm_data.l2 * c2 - (self.qarm_data.l3 - self.qarm_data.lc4) * s23)
            + mL * (self.qarm_data.l2 * c2 - self.qarm_data.l3 * s23)
        )

        G3 = self.qarm_data.g * (
            self.qarm_data.m3 * self.qarm_data.lc3 * s23
            + self.qarm_data.m4 * (self.qarm_data.l3 - self.qarm_data.lc4) * s23
            + mL * self.qarm_data.l3 * s23
        )

        # Assemblage du vecteur G (4x1)
        G = np.array([[0, G2, G3, 0]]).T
        return G

    def friction_vector(self, dq):
        friction = self.qarm_data.Bv * dq + self.qarm_data.Bc * np.sign(dq)
        return friction
