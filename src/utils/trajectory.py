"""
Module de calcul de trajectoire pour un robot à partir de conditions initiales et finales.
"""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray

# from core.dynamics import Lbras, beta, dq_max, l1, l2, l3
from src.utils.types import Waypoint


def get_quintic_coeffs_and_time(
    waypoint_start: Waypoint,
    waypoint_end: Waypoint,
) -> Tuple[NDArray[np.float64], float]:
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
    Retourne une matrice de forme (6, 3) et un temps tf :
    [[a0_x, a0_y, a0_z],
     [a1_x, a1_y, a1_z],
     [a2_x, a2_y, a2_z],
     [a3_x, a3_y, a3_z],
     [a4_x, a4_y, a4_z],
     [a5_x, a5_y, a5_z]], tf
    """
    # I = masse * rayon**2
    # Vmax = Lbras*dq_max
    # Amax = tau_max / I
    # tmin = (waypoint_end.position - waypoint_start.position) / Vmax

    a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(waypoint_start, waypoint_end)

    tf = 5  # Temporaire

    A = np.concatenate([a0(), a1(), a2(), a3(tf), a4(tf), a5(tf)], axis=1).T

    return A, tf


def get_coeffs_as_function_of_time(
    waypoint_start: Waypoint,
    waypoint_end: Waypoint,
) -> Tuple[
    Callable[[], NDArray[np.float64]],
    Callable[[], NDArray[np.float64]],
    Callable[[], NDArray[np.float64]],
    Callable[[float], NDArray[np.float64]],
    Callable[[float], NDArray[np.float64]],
    Callable[[float], NDArray[np.float64]],
]:
    """
    Calcule les coefficients du polynôme de degré 5 pour x, y, z en fonction du temps tf.
    """
    X0 = waypoint_start.position
    dX0 = waypoint_start.velocity if waypoint_start.velocity is not None else np.zeros(3)
    ddX0 = waypoint_start.acceleration if waypoint_start.acceleration is not None else np.zeros(3)
    Xf = waypoint_end.position
    dXf = waypoint_end.velocity if waypoint_end.velocity is not None else np.zeros(3)
    ddXf = waypoint_end.acceleration if waypoint_end.acceleration is not None else np.zeros(3)

    def a0():
        return X0

    def a1():
        return dX0

    def a2():
        return ddX0 / 2

    def a3(tf):
        return (20 * (Xf - X0) - (12 * dX0 + 8 * dXf) * tf - (3 * ddX0 - ddXf) * tf**2) / (
            2 * tf**3
        )

    def a4(tf):
        return (30 * (X0 - Xf) + (16 * dX0 + 14 * dXf) * tf + (3 * ddX0 - 2 * ddXf) * tf**2) / (
            2 * tf**4
        )

    def a5(tf):
        return (12 * (Xf - X0) - 6 * (dX0 + dXf) * tf + (ddXf - ddX0) * tf**2) / (2 * tf**5)

    return a0, a1, a2, a3, a4, a5


def get_desired_state(t: float, coeffs: NDArray[np.float64]) -> Waypoint:
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
        waypoint: Waypoint contenant la position, la vitesse et l'accélération désirées à l'instant t.
    """
    # 1. Vecteurs de base temporelle pour le degré 5
    # On définit les dérivées successives du vecteur [1, t, t^2, t^3, t^4, t^5]
    t_vec = np.array([1, t, t**2, t**3, t**4, t**5], dtype=np.float64)
    v_vec = np.array([0, 1, 2 * t, 3 * t**2, 4 * t**3, 5 * t**4], dtype=np.float64)
    a_vec = np.array([0, 0, 2, 6 * t, 12 * t**2, 20 * t**3], dtype=np.float64)

    # 2. Projection sur les coefficients (Produit matriciel)
    # Chaque opération calcule simultanément les composantes X, Y et Z
    pos_des = (t_vec @ coeffs).reshape(
        3, 1
    )  # [pos_x, pos_y, pos_z].reshape(3, 1) pour s'assurer que c'est un vecteur colonne
    vel_des = (v_vec @ coeffs).reshape(
        3, 1
    )  # [vel_x, vel_y, vel_z].reshape(3, 1) pour s'assurer que c'est un vecteur colonne
    accl_des = (a_vec @ coeffs).reshape(
        3, 1
    )  # [accl_x, accl_y, accl_z].reshape(3, 1) pour s'assurer que c'est un vecteur colonne

    waypoint = Waypoint(position=pos_des, velocity=vel_des, acceleration=accl_des)

    return waypoint
