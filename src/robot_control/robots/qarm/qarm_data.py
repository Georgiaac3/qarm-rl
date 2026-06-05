from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QArmData:
    ###############################
    # Geometric dynamics parameters
    g = 9.80665
    # Longueurs (en mètres)
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

    ############################
    # Motors dynamic parameters
    Bv = np.array(
        [[0.1516, 0.0443, 0.001, 0.0182]]
    ).T  # Coefficients de friction visqueuse (N.m.s/rad)
    # Bv = np.array(
    #    [[0.2119, 0.0457, 0.001, 0.008]]
    # ).T  # Viscous coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
    Bc = np.array([[0.2150, 0.43, 0.4799, 0.0307]]).T  # Coefficients de friction de Coulomb (N.m)
    # Bc = np.array(
    #    [[0.0838, 0.6701, 0.6156, 0.0275]]
    # ).T  # Coulomb coefficients // TODO : voir si on trouve mieux la c'était en commentaire du fichier avec toute la dynamique
    Valim = 12.0  # Tension d'alimentation du moteur (V)
    C_R = 1 / (12 / 4.4)  # Resistance électrique du moteur (Ohm)
    R = np.array(
        [
            C_R,
            C_R,
            C_R,
            5.21,
        ]
    ).reshape(
        4, 1
    )  # Résistances électriques équivalentes pour chaque moteur (Ohm)
    C_ktGR = 10.6 / 4.4
    ktGR = np.array(
        [
            0.5 * C_ktGR,
            0.8 * C_ktGR,
            C_ktGR / 3,
            100000000000,  # 0.005 * 353.5,
        ]
    ).reshape(
        4, 1
    )  # Coefficients de conversion du torque en tension (N.m/A)
    # ktGR = np.array(
    #     [
    #         C_ktGR,
    #         1.5 * C_ktGR,
    #         C_ktGR / 2,
    #         100000000000,
    #     ]
    # ).reshape(
    #     4, 1
    # )  # Coefficients de conversion du torque en tension (N.m/A)
    C_kvGR = 12 / (30 * 2 * np.pi / 60)
    kvGR = np.array(
        [
            C_kvGR,
            C_kvGR,
            C_kvGR,
            100000000000,  # 0.007 * 353.5,
        ]
    ).reshape(
        4, 1
    )  # Coefficients de conversion de la vitesse angulaire en tension (V.s/rad)
