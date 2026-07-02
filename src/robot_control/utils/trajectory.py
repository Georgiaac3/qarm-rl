"""
Module de calcul de trajectoire pour un robot à partir de conditions initiales et finales.
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, minimize

# from core.dynamics import Lbras, beta, dq_max, l1, l2, l3
from .types import Matrix6x3, Waypoint

VEL_MAX = 1  # m/s
ACC_MAX = 1  # m/s^2
C = 0.15
epsilon = 0.001
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
Ltotal = l1 + Lbras


def get_quintic_coeffs_and_time(
    waypoint_start: Waypoint, waypoint_end: Waypoint, tf: Optional[float] = None
) -> Union[Tuple[Matrix6x3, float, bool], Tuple[List[Matrix6x3], List[float], bool]]:
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
     [a5_x, a5_y, a5_z]], tf, intermediate_waypoint_created(boolean = False)

     OU, s'il y a besoin d'un point intermédaire pour respecter les contraintes (position, vitesses, accélérations)
    [matrice de coefficients pour le premier segment, matrice de coefficients pour le second segment], [tf1, tf2], intermediate_waypoint_created(boolean = True)
    """
    # I = masse * rayon**2
    # Vmax = Lbras*dq_max
    # Amax = tau_max / I
    # tmin = (waypoint_end.position - waypoint_start.position) / Vmax

    if in_central_cylinder(waypoint_start):
        raise ValueError(
            "PROBLÈME DE WAYPOINT DANS LE CYLINDRE CENTRAL : Le waypoint de départ est dans le cylindre central. CE PROGRAMME NE GÈRE PAS CE CAS : Il faudrait implémenter une trajectoire qui prend en compte l'orientation initiale pour calculer une trajectoire potable."
        )
    if in_central_cylinder(waypoint_end):
        raise ValueError(
            "PROBLÈME DE WAYPOINT DANS LE CYLINDRE CENTRAL : Le waypoint de fin est dans le cylindre central. CE PROGRAMME NE GÈRE PAS CE CAS : Il faudrait implémenter une trajectoire qui prend en compte l'orientation initiale pour calculer une trajectoire potable."
        )

    verif_constraints(waypoint_start)
    verif_constraints(waypoint_end)

    a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(waypoint_start, waypoint_end)

    # si le temps final n'est pas fourni, on optimise sur le temps pour avoir la première extimation de la trajectoire
    if tf is None:

        def objective(x):
            t = x[0]  # Extraire la valeur scalaire de t (x est un array ici)
            return t  # On veut minimiser le temps final

        def force_constraint(x):
            tf = x[0]  # Extraire la valeur scalaire de tf
            coeffs = np.concatenate([a0(), a1(), a2(), a3(tf), a4(tf), a5(tf)], axis=1).T

            t_samples = np.linspace(0, tf, num=20)
            margins = []

            for t in t_samples:
                wp = get_desired_state(t, coeffs)
                vel_norm = np.linalg.norm(wp.velocity.flatten())
                acc_norm = np.linalg.norm(wp.acceleration.flatten())

                # On ajoute les marges (elles doivent toutes être >= 0 pour que la contrainte soit validée)
                margins.append(VEL_MAX - vel_norm)  # Velmax = 0.5
                margins.append(ACC_MAX - acc_norm)  # Accmax = 1.0

            # On renvoie un tableau numpy avec toutes les marges.
            # SLSQP va s'assurer qu'absolument tous ces éléments restent >= 0.
            return np.array(margins)

        cons = {"type": "ineq", "fun": force_constraint}

        # On démarre avec un temps initial suffisamment grand pour que la première évaluation donne des contraintes valides
        print("PREMIERE optimisation du temps final :")
        res = minimize(
            objective, [5.0], method="SLSQP", bounds=[(epsilon, 10.0)], constraints=cons, tol=1e-6
        )

        print(f"Optimisation du temps final: {res.message}")
        if res.success:
            tf = res.x[0]
            print(f"Temps optimal trouvé : {tf:.2f}s")

    if not isinstance(tf, float) or tf <= 0:
        raise ValueError("Le temps final tf doit être un nombre positif.")

    A = np.concatenate([a0(), a1(), a2(), a3(tf), a4(tf), a5(tf)], axis=1).T

    # creation d'un ou deux points intermédaire si nécessaire pour respecter les contraintes de position
    create_intermediate_waypoint = False
    # vérification des contraintes de position (on se peut pas savoir en avance car les vitesses initiales et finales ne sont pas forcément nulles dont les trajectoires pas forcément en ligne droite)
    t_samples = np.linspace(0, tf, num=30)
    for t in t_samples:
        wp = get_desired_state(t, A)
        if in_central_cylinder(wp):
            create_intermediate_waypoint = True
            break
        try:
            verif_constraints(wp)
        except ValueError as e:
            create_intermediate_waypoint = True
            break

    if create_intermediate_waypoint:
        # Ce waypoint intermédiaire est le résultat d'un optimisation du temps sur les deux segments tout en respectant les contraintes (position, vitesses, accélérations etc)
        def objective_with_intermediate(x):
            x_int, y_int, z_int, dx_int, dy_int, dz_int, ddx_int, ddy_int, ddz_int, tf1, tf2 = x

            return tf1 + tf2  # On veut minimiser le temps total

        def all_constraint_with_intermediate(x):
            x_int, y_int, z_int, dx_int, dy_int, dz_int, ddx_int, ddy_int, ddz_int, tf1, tf2 = x

            intermediate_wp = Waypoint(
                position=np.array([x_int, y_int, z_int]).reshape(3, 1),
                velocity=np.array([dx_int, dy_int, dz_int]).reshape(3, 1),
                acceleration=np.array([ddx_int, ddy_int, ddz_int]).reshape(3, 1),
            )

            a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(waypoint_start, intermediate_wp)
            coeffs1 = np.concatenate([a0(), a1(), a2(), a3(tf1), a4(tf1), a5(tf1)], axis=1).T
            a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(intermediate_wp, waypoint_end)
            coeffs2 = np.concatenate([a0(), a1(), a2(), a3(tf2), a4(tf2), a5(tf2)], axis=1).T

            t_samples1 = np.linspace(0, tf1, num=20)
            t_samples2 = np.linspace(0, tf2, num=20)
            margins = []

            for t in t_samples1:
                wp = get_desired_state(t, coeffs1)

                ###########################
                # Forces (via vel and acc)
                vel_norm = np.linalg.norm(wp.velocity.flatten())
                acc_norm = np.linalg.norm(wp.acceleration.flatten())
                margins.append(VEL_MAX - vel_norm)
                margins.append(ACC_MAX - acc_norm)

                ###########
                # Position
                x, y, z = wp.position.flatten()
                # Central Cylinder
                margins.append(
                    np.sqrt(x**2 + y**2) - C
                )  # On doit être en dehors du cylindre central
                # Dans la base parallélipédique
                c1 = abs(x) + abs(y) - l1
                c2 = z - l1
                # TODO
                # Par le plan x < 0 et y = 0
                margins.append(
                    x + (y**2 / 0.05) - 0.01
                )  # 0.05 est l'épaisseur du plan, 0.01 est la marge de sécurité
                # Hors de portée
                margins.append(
                    Lbras**2 - (x**2 + y**2 + (z - l1) ** 2) - epsilon
                )  # On ajoute une marge
                # Sous le sol
                margins.append(z - epsilon)  # On ajoute une marge de 1cm

            for t in t_samples2:
                wp = get_desired_state(t, coeffs2)

                ###########################
                # Forces (via vel and acc)
                vel_norm = np.linalg.norm(wp.velocity.flatten())
                acc_norm = np.linalg.norm(wp.acceleration.flatten())
                margins.append(VEL_MAX - vel_norm)
                margins.append(ACC_MAX - acc_norm)

                ###########
                # Position
                x, y, z = wp.position.flatten()
                # Central Cylinder
                margins.append(
                    np.sqrt(x**2 + y**2) - C
                )  # On doit être en dehors du cylindre central
                # Dans la base parallélipédique
                c1 = abs(x) + abs(y) - l1
                c2 = z - l1
                # TODO
                # Par le plan x < 0 et y = 0
                margins.append(
                    x + (y**2 / 0.05) - 0.01
                )  # 0.05 est l'épaisseur du plan, 0.01 est la marge de sécurité
                # Hors de portée
                margins.append(
                    Lbras**2 - (x**2 + y**2 + (z - l1) ** 2) - epsilon
                )  # On ajoute une marge
                # Sous le sol
                margins.append(z - epsilon)  # On ajoute une marge de 1cm

            return np.array(margins)

        cons = {"type": "ineq", "fun": all_constraint_with_intermediate}

        ini_int_wp, tf1, tf2 = get_valid_initial_guess(waypoint_start, waypoint_end)
        x0 = np.concatenate(
            [
                ini_int_wp.position.flatten(),
                ini_int_wp.velocity.flatten(),
                ini_int_wp.acceleration.flatten(),
                [tf1, tf2],
            ]
        )
        print(
            f"Initial guess for intermediate waypoint: position = {ini_int_wp.position.flatten()}, velocity = {ini_int_wp.velocity.flatten()}, acceleration = {ini_int_wp.acceleration.flatten()}, tf1 = {tf1}, tf2 = {tf2}"
        )
        initial_margins = all_constraint_with_intermediate(x0)
        print(f"Nombre de contraintes violées au départ : {np.sum(initial_margins < 0)}")
        print(f"Valeurs min des marges : {np.min(initial_margins)}")

        print("WAYPOINT INTERMEDIAIRE :")
        res = minimize(
            objective_with_intermediate,
            x0,
            method="SLSQP",
            bounds=[
                (-1, 1),
                (-1, 1),
                (epsilon, 1),
                (-5, 5),
                (-5, 5),
                (-5, 5),
                (-5, 5),
                (-5, 5),
                (-5, 5),
                (epsilon, 10.0),
                (epsilon, 10.0),
            ],
            constraints=cons,
            tol=1e-6,
        )

        print(f"Optimisation du temps final: {res.message}")
        if res.success:
            tf1 = res.x[9]
            tf2 = res.x[10]
            print(
                f"Temps optimal trouvé : tf1 = {tf1:.2f}s, tf2 = {tf2:.2f}s, total = {tf1 + tf2:.2f}s"
            )
            wp_int = Waypoint(
                position=res.x[0:3].reshape(3, 1),
                velocity=res.x[3:6].reshape(3, 1),
                acceleration=res.x[6:9].reshape(3, 1),
            )
            a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(waypoint_start, wp_int)
            A1 = np.concatenate([a0(), a1(), a2(), a3(tf1), a4(tf1), a5(tf1)], axis=1).T
            a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(wp_int, waypoint_end)
            A2 = np.concatenate([a0(), a1(), a2(), a3(tf2), a4(tf2), a5(tf2)], axis=1).T
            return [A1, A2], [tf1, tf2], True
        else:
            raise ValueError(
                f"L'optimisation du temps final avec un waypoint intermédiaire a échoué entre les waypoints {waypoint_start.position.flatten()} et {waypoint_end.position.flatten()}. Veuillez vérifier les contraintes et les conditions initiales/finales. DON'T ASK FOR TO COMPLECATED HELP CREATING YOUR TRAJECTORIES !! (les vitesses élevées, d'autant plus quand elles sont à des points proches des limites, sont pas gérées par exemple)"
            )

    return A, tf, False


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


def verif_constraints(waypoint: Waypoint):
    # Vérification des contraintes de position, vitesse et accélération pour les waypoints initiaux et finaux
    # - position :
    x = waypoint.position[0, 0]
    y = waypoint.position[1, 0]
    z = waypoint.position[2, 0]
    # hors de portée du robot
    if x**2 + y**2 + (z - l1) ** 2 > Lbras**2:
        raise ValueError("Le waypoint de départ est hors de la portée du robot.")
    # sous le sol
    if z < 0:
        raise ValueError("Le waypoint de départ est sous le sol.")
    # dans la base parallélépipédique du robot
    if abs(x) + abs(y) < l1 and z < l1:
        raise ValueError(
            f"Le waypoint ({x}, {y}, {z}) de départ est dans la base parallélépipédique du robot."
        )
    # dans le plan x < 0 et y = 0 (plan de sécurité)
    if abs(y) < 0.01 and x < 0:
        raise ValueError(f"Le waypoint ({x}, {y}, {z}) de départ est dans le plan x < 0 et y = 0.")
    # - vitesse & accélération :
    # tension de commande, qui est reliée à la vitesse et à l'accélération, trop élevéé
    # TODO : trouver le bon critère, la bonne méthode de faire ça bien et rapidement, car on va pas calculer toutes les commandes en avance à chaque fois ici a priori (trop couteux pour ce qui en est fait)


def in_central_cylinder(waypoint: Waypoint):
    x = waypoint.position[0, 0]
    y = waypoint.position[1, 0]

    if x**2 + y**2 < C**2:
        return True
    return False


def get_valid_initial_guess(
    waypoint_start: Waypoint, waypoint_end: Waypoint
) -> Tuple[Waypoint, float, float]:
    """
    Dans le cas ou il faut trouver un waypoint intermédiaire, avant d'optimiser le temps final, on doit trouver un point de départ pour l'optimisation qui respecte les contraintes de position, vitesse et accélération.
    On utilise pour cela differential_evolution de scipy.optimize pour trouver un point de départ valide.
    """

    def constraint_cost_func(x):
        x_int, y_int, z_int, dx_int, dy_int, dz_int, ddx_int, ddy_int, ddz_int, tf1, tf2 = x

        # tf1 = tf2 = 5.0
        intermediate_wp = Waypoint(
            position=np.array([x_int, y_int, z_int]).reshape(3, 1),
            velocity=np.array([dx_int, dy_int, dz_int]).reshape(3, 1),
            acceleration=np.array([ddx_int, ddy_int, ddz_int]).reshape(3, 1),
        )
        a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(waypoint_start, intermediate_wp)
        coeffs1 = np.concatenate([a0(), a1(), a2(), a3(tf1), a4(tf1), a5(tf1)], axis=1).T
        a0, a1, a2, a3, a4, a5 = get_coeffs_as_function_of_time(intermediate_wp, waypoint_end)
        coeffs2 = np.concatenate([a0(), a1(), a2(), a3(tf2), a4(tf2), a5(tf2)], axis=1).T

        t_samples1 = np.linspace(0, tf1, num=20)
        t_samples2 = np.linspace(0, tf2, num=20)
        violation = 0
        for t in t_samples1:
            wp = get_desired_state(t, coeffs1)

            ##########################
            # Vitesse et accélération
            vel_norm = np.linalg.norm(wp.velocity.flatten())
            acc_norm = np.linalg.norm(wp.acceleration.flatten())
            violation += max(0, vel_norm - VEL_MAX + epsilon)
            violation += max(0, acc_norm - ACC_MAX + epsilon)

            ###########
            # Position
            x, y, z = wp.position.flatten()
            # Central Cylinder
            violation += max(
                0, -(np.sqrt(x**2 + y**2) - C - epsilon)
            )  # On doit être en dehors du cylindre central
            # Dans la base parallélipédique
            c1 = abs(x) + abs(y) - l1
            c2 = z - l1
            # TODO
            # Par le plan x < 0 et y = 0
            violation += max(
                0, -(x + (y**2 / 0.05) - 0.01)
            )  # 0.05 est l'épaisseur du plan, 0.01 est la marge de sécurité
            # Hors de portée
            violation += max(
                0, -(Lbras**2 - (x**2 + y**2 + (z - l1) ** 2) - epsilon)
            )  # On ajoute une marge
            # Sous le sol
            violation += max(0, -(z - epsilon))  # On ajoute une marge de 1cm

        for t in t_samples2:
            wp = get_desired_state(t, coeffs2)

            ##########################
            # Vitesse et accélération
            vel_norm = np.linalg.norm(wp.velocity.flatten())
            acc_norm = np.linalg.norm(wp.acceleration.flatten())
            violation += max(0, vel_norm - VEL_MAX + epsilon)
            violation += max(0, acc_norm - ACC_MAX + epsilon)

            ###########
            # Position
            x, y, z = wp.position.flatten()
            # Central Cylinder
            violation += max(
                0, -(np.sqrt(x**2 + y**2) - C - epsilon)
            )  # On doit être en dehors du cylindre central
            # Dans la base parallélipédique
            c1 = abs(x) + abs(y) - l1
            c2 = z - l1
            # TODO
            # Par le plan x < 0 et y = 0
            violation += max(
                0, -(x + (y**2 / 0.05) - 0.01)
            )  # 0.05 est l'épaisseur du plan, 0.01 est la marge de sécurité
            # Hors de portée
            violation += max(
                0, -(Lbras**2 - (x**2 + y**2 + (z - l1) ** 2) - epsilon)
            )  # On ajoute une marge
            # Sous le sol
            violation += max(0, -(z - epsilon))  # On ajoute une marge de 1cm

        return violation

    result = differential_evolution(
        func=constraint_cost_func,  # On ne cherche pas à optimiser une fonction, juste à trouver un point valide
        bounds=[
            (-1, 1),  # x
            (-1, 1),  # y
            (epsilon, 1),  # z
            (-5, 5),  # dx
            (-5, 5),  # dy
            (-5, 5),  # dz
            (-5, 5),  # ddx
            (-5, 5),  # ddy
            (-5, 5),  # ddz
            (epsilon, 10.0),  # tf1
            (epsilon, 10.0),  # tf2
        ],
    )
    x_int, y_int, z_int, dx_int, dy_int, dz_int, ddx_int, ddy_int, ddz_int, tf1, tf2 = result.x
    int_wp = Waypoint(
        position=np.array([x_int, y_int, z_int]).reshape(3, 1),
        velocity=np.array([dx_int, dy_int, dz_int]).reshape(3, 1),
        acceleration=np.array([ddx_int, ddy_int, ddz_int]).reshape(3, 1),
    )

    return int_wp, tf1, tf2
