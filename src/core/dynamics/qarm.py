import numpy as np

from src.core.dynamics.base_dynamics import Dynamics


class QArmDynamics(Dynamics):

    def __init__(self, QArmData):
        super().__init__()
        # TODO : should take an urdf in the futur to derive everything

        self.g = 9.80665
        self.Bv = np.array([[0.1516, 0.0443, 0.001, 0.0182]]).T  # Coefficients de friction visqueuse (N.m.s/rad)
        # Bv = np.array(
        #    [[0.2119, 0.0457, 0.001, 0.008]]
        # ).T  # Viscous coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
        self.Bc = np.array([[0.2150, 0.43, 0.4799, 0.0307]]).T  # Coefficients de friction de Coulomb (N.m)
        # Bc = np.array(
        #    [[0.0838, 0.6701, 0.6156, 0.0275]]
        # ).T  # Coulomb coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
        self.Valim = 12.0  # Tension d'alimentation du moteur (V)
        self.C_R = 1 / (12 / 4.4)  # Resistance électrique du moteur (Ohm)
        self.R = np.array(
            [
                self.C_R,
                self.C_R,
                self.C_R,
                5.21,
            ]
        ).reshape(
            4, 1
        )  # Résistances électriques équivalentes pour chaque moteur (Ohm)
        self.C_ktGR = 10.6 / 4.4
        self.ktGR = np.array(
            [
                0.5* self.C_ktGR,
                0.8 * self.C_ktGR,
                self.C_ktGR / 3,
                100000000000, #0.005 * 353.5,
            ]
        ).reshape(
            4, 1
        )  # Coefficients de conversion du torque en tension (N.m/A)
        # ktGR = np.array(
        #     [
        #         self.C_ktGR,
        #         1.5 * self.C_ktGR,
        #         self.C_ktGR / 2,
        #         100000000000,
        #     ]
        # ).reshape(
        #     4, 1
        # )  # Coefficients de conversion du torque en tension (N.m/A)
        self.C_kvGR = 12 / (30 * 2 * np.pi / 60)
        self.kvGR = np.array(
            [
                self.C_kvGR,
                self.C_kvGR,
                self.C_kvGR,
                100000000000, #0.007 * 353.5,
            ]
        ).reshape(
            4, 1
        )  # Coefficients de conversion de la vitesse angulaire en tension (V.s/rad)


        # Paramètres du manipulateur (Longueurs en mètres)
        self.L1 = 0.1400
        self.L2 = 0.3500
        self.L3 = 0.0500
        self.L4 = 0.2500
        self.L5 = 0.1500

        # Calcul de beta et longueurs alternées, ce sont les lambda, les longueurs utilisées pour les calculs dynamiques (voir schéma du bras)
        self.beta = np.arctan(self.L3 / self.L2)
        self.l1 = self.L1
        self.l2 = np.sqrt(self.L2**2 + self.L3**2)
        self.l3 = self.L4 + self.L5

        self.Lbras = (
            self.l2 + self.l3
        )  # Longueur totale du bras, utilisée pour les contraintes de vitesse et accélération linéaires
        self.Ltotal = (
            self.l1 + self.Lbras
        )  # Longueur totale du système, utilisée pour les contraintes de vitesse et accélération linéaires

        self.dq_max = np.pi / 2  # Vitesse angulaire maximale (rad/s)

        # Paramètres dynamiques (Inerties et centres de masse)
        # I1A = 1.489e-3
        # I2A, I2L = 1.922e-4, 9.61e-3
        # I3A, I3L = 2.679e-4, 2.069e-3
        # I4A, I4L = 5.528e-4, 1.12e-3

        # innerties but inversed with what is on doc 7 (not the original one from the Quanser code)
        self.I1A = 1.489e-3
        self.I2A, self.I2L = 1.922e-4, 9.61e-3
        self.I3A, self.I3L = 2.069e-3, 2.679e-4
        self.I4A, self.I4L = 1.12e-3, 5.528e-4

        # lc1 = l1/3;
        # lc2 = l2/2;
        # lc3 = L4/2;
        # lc4 = L5/2;

        # Centres de masse (en mètres, depuis l'axe de rotation), ce sont les lambda_c sur le schéma du bras
        self.lc1 = 0.0399
        self.lc2 = 0.1071
        self.lc3 = 0.1561
        self.lc4 = 0.0998

        # Masses (en kg)
        self.m1 = 0.7906
        self.m2 = 0.4591
        self.m3 = 0.269
        self.m4 = 0.257
        # mL = 0  # Masse de la charge utile (peut être ajustée selon la mission)

    def get_trig_values(self, q):
        """
        Calcule les valeurs trigonométriques nécessaires pour les calculs dynamiques à partir des angles q déjà transformés depuis les angles mesurés pour le calcul de la dynamique.
        q: angles utilisés pour les calculs dynamiques
        Retourne c1, s1, c2, s2, c3, s3, c23, s23: les valeurs trigonométriques nécessaires pour les calculs dynamiques
        """
        c1 = np.cos(q[0, 0])
        s1 = np.sin(q[0, 0])
        c2 = np.cos(q[1, 0])
        s2 = np.sin(q[1, 0])
        c3 = np.cos(q[2, 0])
        s3 = np.sin(q[2, 0])
        c23 = np.cos(q[1, 0] + q[2, 0])
        s23 = np.sin(q[1, 0] + q[2, 0])

        return self.convert_trig_values(c1, s1, c2, s2, c3, s3, c23, s23)


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

        q = phi + np.array([[0, self.beta, -self.beta, 0]]).T
        dq = phi_d
        ddq = phi_dd
        return q, dq, ddq

    def get_inertia_matrix(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        c1, s1, c2, s2, c3, s3, c23, s23 = self.get_trig_values(q)

        M11 = (
            self.I1A
            + self.m2 * (self.l2 - self.lc2) ** 2 * c2**2
            + self.I2A * s2**2
            + self.I2L * c2**2
            + self.m3 * self.l2**2 * c2**2
            + self.m3 * self.lc3**2 * s23**2
            - 2 * self.m3 * self.l2 * self.lc3 * c2 * s23
            + self.I3L * s23**2
            + self.I3A * c23**2
            + self.m4 * (self.l3 - self.lc4) ** 2 * s23**2
            + self.m4 * self.l2**2 * c2**2
            - 2 * self.m4 * (self.l3 - self.lc4) * self.l2 * c2 * s23
            + self.I4L * s23**2
            + self.I4A * c23**2
            + self.mL * self.l3**2 * s23**2
            + self.mL * self.l2**2 * c2**2
            - 2 * self.mL * self.l2 * self.l3 * c2 * s23
        )

        M14 = -self.I4A * c23
        M22 = (
            self.m2 * (self.l2 - self.lc2) ** 2
            + self.I2L
            + self.m3 * self.l2**2
            + self.m3 * self.lc3**2
            - 2 * self.m3 * self.l2 * self.lc3 * s3
            + self.I3L
            + self.m4 * (self.l2 + self.l3 - self.lc4) ** 2
            + self.I4L
            + self.mL * self.l2**2
            + self.mL * self.l3**2
            - 2 * self.mL * self.l2 * self.l3 * s3
        )

        M23 = (
            self.m3 * self.lc3**2
            - self.m3 * self.lc3 * self.l2 * s3
            + self.I3L
            + self.m4 * (self.l3 - self.lc4 - self.l2 * s3) * (self.l3 - self.lc4)
            + self.I4L
            - self.mL * (self.l3**2 - self.l2 * self.l3 * s3)
        )

        M33 = self.m3 * self.lc3**2 + self.I3L + self.m4 * (self.lc4 - self.l3) ** 2 + self.I4L + self.mL * self.l3**2

        M = np.array(
            [[M11, 0, 0, M14], [0, M22, M23, 0], [0, M23, M33, 0], [M14, 0, 0, self.I4A]]  # M32 = M23
        )

        return M

    def coriolis_matrix(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        c1, s1, c2, s2, c3, s3, c23, s23 = self.get_trig_values(q)

        # Calcul des coefficients
        B11 = (
            -2 * self.m2 * (self.l2 - self.lc2) ** 2 * s2 * c2
            + 2 * self.I2A * s2 * c2
            - 2 * self.I2L * s2 * c2
            - 2 * self.m3 * self.l2**2 * s2 * c2
            + 2 * self.m3 * self.lc3**2 * s23 * c23
            + 2 * self.m3 * self.l2 * self.lc3 * (s2 * s23 - c2 * c23)
            + 2 * self.I3L * s23 * c23
            - 2 * self.I3A * s23 * c23
            + 2 * self.m4 * (self.l3 - self.lc4) ** 2 * s23 * c23
            - 2 * self.m4 * self.l2**2 * s2 * c2
            + 2 * self.m4 * (self.l3 - self.lc4) * self.l2 * (s2 * s23 - c2 * c23)
            + 2 * self.I4L * s23 * c23
            - 2 * self.I4A * s23 * c23
            + 2 * self.mL * self.l3**2 * s23 * c23
            - 2 * self.mL * self.l2**2 * s2 * c2
            + 2 * self.mL * self.l2 * self.l3 * (s2 * s23 - c2 * c23)
        )

        B12 = (
            2 * self.m3 * self.lc3**2 * s23 * c23
            - 2 * self.m3 * self.l2 * self.lc3 * self.c2 * self.c23
            + 2 * self.I3L * s23 * c23
            - 2 * self.I3A * s23 * c23
            + 2 * self.m4 * (self.l3 - self.lc4) ** 2 * s23 * c23
            - 2 * self.m4 * (self.l3 - self.lc4) * self.l2 * self.c2 * self.c23
            + 2 * self.I4L * s23 * c23
            - 2 * self.I4A * s23 * c23
            + 2 * self.mL * self.l3**2 * s23 * c23
            - 2 * self.mL * self.l2 * self.l3 * self.c2 * self.c23
        )

        B15 = self.I4A * s23
        B16 = self.I4A * s23
        B23 = -self.I4A * s23
        B24 = -2 * self.m3 * self.l2 * self.lc3 * self.c3 - 2 * self.mL * self.l2 * self.l3 * self.c3
        B33 = -self.I4A * s23
        B41 = self.I4A * s23
        B42 = self.I4A * s23

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

        c1, s1, c2, s2, c3, s3, c23, s23 = self.get_trig_values(q)

        # Calcul des coefficients
        C21 = -(
            -self.m2 * (self.l2 - self.lc2) ** 2 * self.s2 * self.c2
            + self.I2A * self.s2 * self.c2
            - self.I2L * self.s2 * self.c2
            - self.m3 * self.l2**2 * self.s2 * self.c2
            + self.m3 * self.lc3**2 * self.s23 * self.c23
            + self.m3 * self.l2 * self.lc3 * (self.s2 * self.s23 - self.c2 * self.c23)
            + self.I3L * self.s23 * self.c23
            - self.I3A * self.s23 * self.c23
            + self.m4 * (self.l3 - self.lc4) ** 2 * self.s23 * self.c23
            - self.m4 * self.l2**2 * self.s2 * self.c2
            + self.m4 * (self.l3 - self.lc4) * self.l2 * (self.s2 * self.s23 - self.c2 * self.c23)
            + self.I4L * self.s23 * self.c23
            - self.I4A * self.s23 * self.c23
            + self.mL * self.l3**2 * self.s23 * self.c23
            - self.mL * self.l2**2 * self.s2 * self.c2
            + self.mL * self.l2 *(self.l3) * (self.s2 * self.s23 - self.c2 *(self.c23)
        )

        C23 = -self.mL * self.lc3 * self.l2 * self.c3 - self.m4 * self.l2 * self.c3 * (self.l3 - self.lc4) + self.mL * self.l2 * self.l3 * self.c3

        C31 = -(
            self.m3 * self.lc3**2 * self.s23 * self.c23
            - self.m3 * self.l2 * self.lc3 * self.c2 * self.c23
            + self.I3L * self.s23 * self.c23
            - self.I3A * self.s23 * self.c23
            + self.m4 * (self.l3 - self.lc4) ** 2 * self.s23 * self.c23
            - self.m4 * (self.l3 - self.lc4) * self.l2 * self.c2 * self.c23
            + self.I4L * self.s23 * self.c23
            - self.I4A * self.s23 * self.c23
            + self.mL * self.l3**2 * self.s23 * self.c23
            - self.mL * self.l2 * self.l3 * self.c2 * self.c23
        )

        C32 = self.m3 * self.l2 * self.lc3 * self.c3 + self.mL * self.l2 * self.l3 * self.c3

        # Assemblage de la matrice
        C = np.array([[0, 0, 0, 0], [C21, 0, C23, 0], [C31, C32, 0, 0], [0, 0, 0, 0]])
        return C

    def get_gravity_vector(self, q, mL=0):
        if q.shape != (4, 1):
            raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

        # Rappel des raccourcis trigonométriques
        c1, s1, c2, s2, c3, s3, c23, s23 = self.get_trig_values(q)

        # Calcul des composantes
        G2 = -self.g * (
            self.m2 * (self.l2 - self.lc2) * self.c2
            + self.m3 * (self.l2 * self.c2 - self.lc3 * self.s23)
            + self.m4 * (self.l2 * self.c2 - (self.l3 - self.lc4) * self.s23)
            + self.mL * (self.l2 * self.c2 - self.l3 * self.s23)
        )

        G3 = self.g * (self.m3 * self.lc3 * self.s23 + self.m4 * (self.l3 - self.lc4) * self.s23 + self.mL * self.l3 * self.s23)

        # Assemblage du vecteur G (4x1)
        G = np.array([[0, G2, G3, 0]]).T
        return G

    def friction_vector(self, dq):
        friction = self.Bv * dq + self.Bc * np.sign(dq)
        return friction
