"""
Module de calcul de trajectoire pour un robot à partir de conditions initiales et finales.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def get_quintic_coeffs_and_time(
    pos0: NDArray[np.float64],
    vel0: NDArray[np.float64],
    acc0: NDArray[np.float64],
    posf: NDArray[np.float64],
    velf: NDArray[np.float64],
    acclf: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
    """
    Calcule les coefficients du polynôme de degré 5 pour x, y, z.
    Ils sont solution du système linéaire :
    [[1, 0,  0,     0,       0,        0       ],
     [0, 1,  0,     0,       0,        0       ],
     [0, 0,  2,     0,       0,        0       ],
     [1, tf, tf**2, tf**3,   tf**4,    tf**5   ],
     [0, 1,  2*tf,  3*tf**2, 4*tf**3,  5*tf**4 ],
     [0, 0,  2,     6*tf,    12*tf**2, 20*tf**3]]
            @ [[a0_x, a0_y, a0_z],
              [a1_x, a1_y, a1_z],
              [a2_x, a2_y, a2_z],
              [a3_x, a3_y, a3_z],
              [a4_x, a4_y, a4_z],
              [a5_x, a5_y, a5_z]] = [[pos0_x, pos0_y, pos0_z],
                                      [vel0_x, vel0_y, vel0_z],
                                      [acc0_x, acc0_y, acc0_z],
                                      [posf_x, posf_y, posf_z],
                                      [velf_x, velf_y, velf_z],
                                      [acclf_x, acclf_y, acclf_z]]

    Avec tf = durée minimale pour passer de pos0 à posf tout en respectant les contraintes de vitesse Vmax et accélération Amax.
    Retourne une matrice de forme (6, 3) :
    [[a0_x, a0_y, a0_z],
     [a1_x, a1_y, a1_z],
     [a2_x, a2_y, a2_z],
     [a3_x, a3_y, a3_z],
     [a4_x, a4_y, a4_z],
     [a5_x, a5_y, a5_z]]
    """
    Vmax = Ltotal * omega_max  # Vitesse linéaire maximale (m/s)
    Amax = Ltotal * alpha_max  # Accélération linéaire maximale (m/s²)
    tmin = max(max(posf - pos0) / Vmax, max(posf - pos0) / Amax) + marge # Durée minimale pour respecter les contraintes
    tmax = 10.0  # Durée maximale pour éviter des trajets trop longs

    # Méthode de Brent pour trouver tf dans [tmin, tmax] tel que les conditions finales soient respectées
    def objective(tf):


def get_desired_state(
    t: float,
    coeffs: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calcule l'état désiré au temps t à partir des coefficients du polynôme de degré 5.

    Args:
        t: Temps actuel (s)
        coeffs: Matrice des coefficients de forme (6, 3):
        [[a0_x, a0_y, a0_z],
         [a1_x, a1_y, a1_z],
         [a2_x, a2_y, a2_z],
         [a3_x, a3_y, a3_z],
         [a4_x, a4_y, a4_z],
         [a5_x, a5_y, a5_z]]

    Returns:
        pos_des: Position désirée [x, y, z] (m)
        vel_des: Vitesse désirée [vx, vy, vz] (m/s)
        accl_des: Accélération désirée [ax, ay, az] (m/s²)
    """
    # 1. Vecteurs de base temporelle pour le degré 5
    # On définit les dérivées successives du vecteur [1, t, t^2, t^3, t^4, t^5]
    t_vec = np.array([1, t, t**2, t**3,   t**4,    t**5],    dtype=np.float64)
    v_vec = np.array([0, 1, 2*t, 3*t**2, 4*t**3,  5*t**4],  dtype=np.float64)
    a_vec = np.array([0, 0, 2,   6*t,    12*t**2, 20*t**3], dtype=np.float64)

    # 2. Projection sur les coefficients (Produit matriciel)
    # Chaque opération calcule simultanément les composantes X, Y et Z
    pos_des  = t_vec @ coeffs
    vel_des  = v_vec @ coeffs
    accl_des = a_vec @ coeffs

    return pos_des, vel_des, accl_des
