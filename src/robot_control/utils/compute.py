import numpy as np


def get_trig_values4(q):
    """
    Calcule les valeurs trigonométriques nécessaires pour les calculs dynamiques et cinématiques à partir d'angles q.
    q: vecteur colonne des angles articulaires de la dynamique géométrique (4, 1)
    Retourne c1, s1, c2, s2, c3, s3, c23, s23: les valeurs trigonométriques nécessaires pour les calculs dynamiques et cinématiques.
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
