"""
Module pour calculer les dynamiques du bras robotique QARM.
Traduit en Python depuis un code mis a jours par Leonam Pecly - November 10, 2021
Based on code provided by Quanser
"""

import numpy as np

# Constante de l'accélération gravitationnelle
g = 9.80665  # *0.9967
Bv = np.array([[0.1516, 0.0443, 0.001, 0.0182]]).T  # Coefficients de friction visqueuse (N.m.s/rad)
# Bv = np.array(
#    [[0.2119, 0.0457, 0.001, 0.008]]
# ).T  # Viscous coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
Bc = np.array([[0.2150, 0.43, 0.4799, 0.0307]]).T  # Coefficients de friction de Coulomb (N.m)
# Bc = np.array(
#    [[0.0838, 0.6701, 0.6156, 0.0275]]
# ).T  # Coulomb coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
Valim = 12.0  # Tension d'alimentation du moteur (V)
R = 1 / (12 / 4.4)  # Resistance électrique du moteur (Ohm)
ktGR = 1.5 * 10.6 / 4.4
kvGR = 12 / (30 * 2 * np.pi / 60)

# Paramètres du manipulateur (Longueurs en mètres)
L1 = 0.1400
L2 = 0.3500
L3 = 0.0500
L4 = 0.2500
L5 = 0.1500

# Calcul de beta et longueurs alternées, ce sont les lambda, les longueurs utilisées pour les calculs dynamiques (voir schéma du bras)
beta = np.arctan(L3 / L2)
l1 = L1
l2 = np.sqrt(L2**2 + L3**2)
l3 = L4 + L5

Lbras = (
    l2 + l3
)  # Longueur totale du bras, utilisée pour les contraintes de vitesse et accélération linéaires
Ltotal = (
    l1 + Lbras
)  # Longueur totale du système, utilisée pour les contraintes de vitesse et accélération linéaires

dq_max = np.pi / 2  # Vitesse angulaire maximale (rad/s)

# Paramètres dynamiques (Inerties et centres de masse)
# I1A = 1.489e-3
# I2A, I2L = 1.922e-4, 9.61e-3
# I3A, I3L = 2.679e-4, 2.069e-3
# I4A, I4L = 5.528e-4, 1.12e-3

# innerties but inversed with what is on doc 7 (not the original one from the Quanser code)
I1A = 1.489e-3
I2A, I2L = 1.922e-4, 9.61e-3
I3A, I3L = 2.069e-3, 2.679e-4
I4A, I4L = 1.12e-3, 5.528e-4


# lc1 = l1/3;
# lc2 = l2/2;
# lc3 = L4/2;
# lc4 = L5/2;

# Centres de masse (en mètres, depuis l'axe de rotation), ce sont les lambda_c sur le schéma du bras
lc1 = 0.0399
lc2 = 0.1071
lc3 = 0.1561
lc4 = 0.0998

# Masses (en kg)
m1 = 0.7906
m2 = 0.4591
m3 = 0.269
m4 = 0.257
# mL = 0  # Masse de la charge utile (peut être ajustée selon la mission)


def get_trig_values(q):
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

    return convert_trig_values(c1, s1, c2, s2, c3, s3, c23, s23)


def convert_trig_values(c1, s1, c2, s2, c3, s3, c23, s23):
    """
    Convertit des valeurs trigonométriques de la convention old vers la convention new.

    Retourne:
        Une liste dans l'ordre (c1_new, s1_new, c2_new, s2_new, c3_new, s3_new, c23_new, s23_new).
    """
    # old -> new:
    # c2_old = s2_new, s2_old = -c2_new
    # c23_old = s23_new, s23_old = -c23_new
    return [c1, s1, s2, -c2, c3, s3, s23, -c23]


# def transform_angles(phi, phi_d, phi_dd):
#     """
#     Transforme les angles mesurés phi en angles q utilisés pour les calculs dynamiques.
#     phi: array de taille 4 (angles mesurés)
#     phi_d: array de taille 4 (vitesses angulaires mesurées)
#     phi_dd: array de taille 4 (accélérations angulaires mesurées)
#     Retourne q, dq, ddq: arrays de taille 4
#     """
#     if phi.shape != (4, 1) or phi_d.shape != (4, 1) or phi_dd.shape != (4, 1):
#         raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

#     q = phi - np.array([[0, np.pi / 2 - beta, beta, 0]]).T
#     dq = phi_d
#     ddq = phi_dd
#     return q, dq, ddq


def transform_angles(phi, phi_d, phi_dd):
    """
    Transforme les angles mesurés phi en angles q utilisés pour les calculs dynamiques.
    phi: array de taille 4 (angles mesurés)
    phi_d: array de taille 4 (vitesses angulaires mesurées)
    phi_dd: array de taille 4 (accélérations angulaires mesurées)
    Retourne q, dq, ddq: arrays de taille 4
    """
    if phi.shape != (4, 1) or phi_d.shape != (4, 1) or phi_dd.shape != (4, 1):
        raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

    q = phi + np.array([[0, beta, -beta, 0]]).T
    dq = phi_d
    ddq = phi_dd
    return q, dq, ddq


def get_inertia_matrix(q, mL=0):
    """
    Calcule la matrice d'inertie (M) du bras robotique.
    mL: masse de la charge utile
    q: angles utilisés pour les calculs dynamiques
    """

    # TODO : vérifier que la matrice d'inertie prends en compte l'inertie des moteurs

    if q.shape != (4, 1):
        raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

    c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

    M11 = (
        I1A
        + m2 * (l2 - lc2) ** 2 * c2**2
        + I2A * s2**2
        + I2L * c2**2
        + m3 * l2**2 * c2**2
        + m3 * lc3**2 * s23**2
        - 2 * m3 * l2 * lc3 * c2 * s23
        + I3L * s23**2
        + I3A * c23**2
        + m4 * (l3 - lc4) ** 2 * s23**2
        + m4 * l2**2 * c2**2
        - 2 * m4 * (l3 - lc4) * l2 * c2 * s23
        + I4L * s23**2
        + I4A * c23**2
        + mL * l3**2 * s23**2
        + mL * l2**2 * c2**2
        - 2 * mL * l2 * l3 * c2 * s23
    )

    M14 = -I4A * c23
    M22 = (
        m2 * (l2 - lc2) ** 2
        + I2L
        + m3 * l2**2
        + m3 * lc3**2
        - 2 * m3 * l2 * lc3 * s3
        + I3L
        + m4 * (l2 + l3 - lc4) ** 2
        + I4L
        + mL * l2**2
        + mL * l3**2
        - 2 * mL * l2 * l3 * s3
    )

    M23 = (
        m3 * lc3**2
        - m3 * lc3 * l2 * s3
        + I3L
        + m4 * (l3 - lc4 - l2 * s3) * (l3 - lc4)
        + I4L
        - mL * (l3**2 - l2 * l3 * s3)
    )

    M33 = m3 * lc3**2 + I3L + m4 * (lc4 - l3) ** 2 + I4L + mL * l3**2

    M = np.array(
        [[M11, 0, 0, M14], [0, M22, M23, 0], [0, M23, M33, 0], [M14, 0, 0, I4A]]  # M32 = M23
    )

    return M


def get_centrifugal_matrix(q, mL=0):
    """
    Calcule la matrice centrifuge (C) du bras robotique.
    q: angles utilisés pour les calculs dynamiques
    dq: vitesses angulaires utilisés pour les calculs dynamiques
    """

    if q.shape != (4, 1):
        raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

    c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

    # Calcul des coefficients
    C21 = -(
        -m2 * (l2 - lc2) ** 2 * s2 * c2
        + I2A * s2 * c2
        - I2L * s2 * c2
        - m3 * l2**2 * s2 * c2
        + m3 * lc3**2 * s23 * c23
        + m3 * l2 * lc3 * (s2 * s23 - c2 * c23)
        + I3L * s23 * c23
        - I3A * s23 * c23
        + m4 * (l3 - lc4) ** 2 * s23 * c23
        - m4 * l2**2 * s2 * c2
        + m4 * (l3 - lc4) * l2 * (s2 * s23 - c2 * c23)
        + I4L * s23 * c23
        - I4A * s23 * c23
        + mL * l3**2 * s23 * c23
        - mL * l2**2 * s2 * c2
        + mL * l2 * l3 * (s2 * s23 - c2 * c23)
    )

    C23 = -mL * lc3 * l2 * c3 - m4 * l2 * c3 * (l3 - lc4) + mL * l2 * l3 * c3

    C31 = -(
        m3 * lc3**2 * s23 * c23
        - m3 * l2 * lc3 * c2 * c23
        + I3L * s23 * c23
        - I3A * s23 * c23
        + m4 * (l3 - lc4) ** 2 * s23 * c23
        - m4 * (l3 - lc4) * l2 * c2 * c23
        + I4L * s23 * c23
        - I4A * s23 * c23
        + mL * l3**2 * s23 * c23
        - mL * l2 * l3 * c2 * c23
    )

    C32 = m3 * l2 * lc3 * c3 + mL * l2 * l3 * c3

    # Assemblage de la matrice
    C = np.array([[0, 0, 0, 0], [C21, 0, C23, 0], [C31, C32, 0, 0], [0, 0, 0, 0]])
    return C


def get_coriolis_matrix(q, mL=0):
    """
    Calcule la matrice de Coriolis (B) du bras robotique.
    q: angles utilisés pour les calculs dynamiques
    dq: vitesses angulaires utilisés pour les calculs dynamiques
    """

    if q.shape != (4, 1):
        raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

    c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

    # Calcul des coefficients
    B11 = (
        -2 * m2 * (l2 - lc2) ** 2 * s2 * c2
        + 2 * I2A * s2 * c2
        - 2 * I2L * s2 * c2
        - 2 * m3 * l2**2 * s2 * c2
        + 2 * m3 * lc3**2 * s23 * c23
        + 2 * m3 * l2 * lc3 * (s2 * s23 - c2 * c23)
        + 2 * I3L * s23 * c23
        - 2 * I3A * s23 * c23
        + 2 * m4 * (l3 - lc4) ** 2 * s23 * c23
        - 2 * m4 * l2**2 * s2 * c2
        + 2 * m4 * (l3 - lc4) * l2 * (s2 * s23 - c2 * c23)
        + 2 * I4L * s23 * c23
        - 2 * I4A * s23 * c23
        + 2 * mL * l3**2 * s23 * c23
        - 2 * mL * l2**2 * s2 * c2
        + 2 * mL * l2 * l3 * (s2 * s23 - c2 * c23)
    )

    B12 = (
        2 * m3 * lc3**2 * s23 * c23
        - 2 * m3 * l2 * lc3 * c2 * c23
        + 2 * I3L * s23 * c23
        - 2 * I3A * s23 * c23
        + 2 * m4 * (l3 - lc4) ** 2 * s23 * c23
        - 2 * m4 * (l3 - lc4) * l2 * c2 * c23
        + 2 * I4L * s23 * c23
        - 2 * I4A * s23 * c23
        + 2 * mL * l3**2 * s23 * c23
        - 2 * mL * l2 * l3 * c2 * c23
    )

    B15 = I4A * s23
    B16 = I4A * s23
    B23 = -I4A * s23
    B24 = -2 * m3 * l2 * lc3 * c3 - 2 * mL * l2 * l3 * c3
    B33 = -I4A * s23
    B41 = I4A * s23
    B42 = I4A * s23

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


# def get_gravity_vector(q, mL=0):
#     """
#     Calcule le vecteur de gravité (G) du bras robotique.
#     q: angles utilisés pour les calculs dynamiques
#     mL: masse de la charge utile
#     """
#     if q.shape != (4, 1):
#         raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

#     # Rappel des raccourcis trigonométriques
#     c2 = np.cos(q[1, 0])
#     s23 = np.sin(q[1, 0] + q[2, 0])

#     # Calcul des composantes
#     G2 = -g * (
#         m2 * (l2 - lc2) * c2
#         + m3 * (l2 * c2 - lc3 * s23)
#         + m4 * (l2 * c2 - (l3 - lc4) * s23)
#         + mL * (l2 * c2 - l3 * s23)
#     )
#     #beta = 0
#     G2 = -g * (
#         m2 * (l2 - lc2) * np.sin(q[1, 0] + beta)
#         + m3 * (l2 * np.sin(q[1, 0] + beta) + lc3 * np.cos(q[1, 0]+q[2, 0]))
#         + m4 * (l2 * np.sin(q[1, 0] + beta) + (l3 - lc4) * np.cos(q[1, 0]+q[2, 0]))
#         + mL * (l2 * np.sin(q[1, 0] + beta) + l3 * np.cos(q[1, 0]+q[2, 0]))
#     )

#     G3 = g * (
#         m3 * lc3 * s23
#         + m4 * (l3 - lc4) * s23
#         + mL * l3 * s23
#     )

#     G3 = -g * (
#         m3 * lc3
#         + m4 * (l3 - lc4)
#         + mL * l3
#     ) * np.cos(q[1, 0] + q[2, 0])

#     #print(q[1, 0] + beta)

#     #G3 = (-g * lc3 * m3 - g * (l3 - lc4) * m4)*np.cos(q[2, 0] + beta)

#     # Assemblage du vecteur G (4x1)
#     G = np.array([[0,
#                    G2,
#                    G3, 0]]).T
#     return G

# def get_gravity_vector(q, mL=0):
#     """
#     Calcule le vecteur de gravité (G) du bras robotique.
#     q: angles utilisés pour les calculs dynamiques
#     mL: masse de la charge utile
#     """
#     if q.shape != (4, 1):
#         raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

#     # Rappel des raccourcis trigonométriques
#     s2 = np.sin(q[1, 0])
#     c23 = np.cos(q[1, 0] + q[2, 0])

#     # Calcul des composantes
#     G2 = -g * (
#         m2 * (l2 - lc2) * s2
#         + m3 * (l2 * s2 + lc3 * c23)
#         + m4 * (l2 * s2 + (l3 - lc4) * c23)
#         + mL * (l2 * s2 + l3 * c23)
#     )

#     G3 = -g * (
#         m3 * lc3
#         + m4 * (l3 - lc4)
#         + mL * l3
#     ) * c23

#     # Assemblage du vecteur G (4x1)
#     G = np.array([[0, G2, G3, 0]]).T
#     return G


def get_gravity_vector(q, mL=0):
    """
    Calcule le vecteur de gravité (G) du bras robotique.
    q: angles utilisés pour les calculs dynamiques
    mL: masse de la charge utile
    """
    if q.shape != (4, 1):
        raise ValueError("Les angles d'entrée doivent être des np arrays de taille (4, 1).")

    # Rappel des raccourcis trigonométriques
    c1, s1, c2, s2, c3, s3, c23, s23 = get_trig_values(q)

    # Calcul des composantes
    G2 = -g * (
        m2 * (l2 - lc2) * c2
        + m3 * (l2 * c2 - lc3 * s23)
        + m4 * (l2 * c2 - (l3 - lc4) * s23)
        + mL * (l2 * c2 - l3 * s23)
    )

    G3 = g * (m3 * lc3 * s23 + m4 * (l3 - lc4) * s23 + mL * l3 * s23)

    # Assemblage du vecteur G (4x1)
    G = np.array([[0, G2, G3, 0]]).T
    return G


def get_friction(dq):
    """
    Calcule la friction (tau_f) du bras robotique.
    dq: vitesses angulaires utilisés pour les calculs dynamiques
    Bv: coefficient de friction visqueux
    Bc: coefficient de friction de Coulomb

    Remarque : la striction (friction statique) n'est pas prise en compte ni modélisée ici (à faire plus tard si nécessaire !)
    """
    friction = Bv * dq + Bc * np.sign(dq)
    return friction


def get_coriolis_velocity_signals(dq):
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


def get_pwm(q_geo_mes, dq_geo_mes, ddq_geo_cmd, mL=0):
    """
    Calcule les signaux PWM à envoyer aux moteurs à partir des angles, vitesses et accélérations angulaires géométriques mesurées, et accélérations angulaires géométriques commandées.
    q_geo_mes: angles géométriques mesurés
    dq_geo_mes: vitesses angulaires géométriques mesurées
    ddq_geo_cmd: accélérations angulaires géométriques commandées
    mL: masse de la charge utile
    Retourne un array de taille (4, 1) avec les signaux PWM normalisés entre -1 et 1.
    """
    # 1. Calcul des matrices et vecteurs dynamiques à partir des angles géométriques mesurés
    M = get_inertia_matrix(q_geo_mes, mL)
    C = get_centrifugal_matrix(q_geo_mes, mL)
    G = get_gravity_vector(q_geo_mes, mL)
    B = get_coriolis_matrix(q_geo_mes, mL)

    friction = get_friction(dq_geo_mes)

    B_signals = get_coriolis_velocity_signals(dq_geo_mes)

    # print("vitesses mesurées et accélérations commandées:", dq_geo_mes.ravel(), ddq_geo_cmd.ravel())

    # 2. Calcul du torque total à appliquer
    tau_cmd = np.array([[0.0, 0.0, 0.0, 0.0]]).T
    tau_cmd += M @ ddq_geo_cmd
    tau_cmd += B @ B_signals
    tau_cmd += C @ dq_geo_mes**2
    tau_cmd += G
    tau_cmd += friction

    return tau_cmd

    # print(dq_geo_mes.ravel())

    # 3. Conversion du torque en signal de tension (V) à envoyer au moteur
    Vcmd = (R / ktGR) * tau_cmd  # + kvGR * dq_geo_mes

    # mutliply by 2 the voltage command of the first motor because there are 2 motors in parallel for the first joint
    # Vcmd[0, 0] *= 2

    # 4. Normalisation du signal de tension entre -1 et 1
    # pwm = np.clip(Vcmd / (10 * Valim), -1, 1)
    pwm = np.clip(Vcmd / Valim, -1, 1)

    return pwm
