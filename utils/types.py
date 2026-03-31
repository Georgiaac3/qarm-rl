"""
Module de définition de types de données.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Waypoint:
    """
    Représente un point de passage ou la position finale pour la trajectoire du robot.
    """

    position: NDArray[np.float64]
    velocity: Optional[NDArray[np.float64]] = None  # None = Vitesse nulle par défaut
    acceleration: Optional[NDArray[np.float64]] = None
