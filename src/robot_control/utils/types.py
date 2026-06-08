"""
Module de définition de types de données.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from jaxtyping import Float

Vector3x1 = Float[np.ndarray, "3 1"]  # Column vector 3x1
Vector4x1 = Float[np.ndarray, "4 1"]  # Column vector 4x1
Vector6x1 = Float[np.ndarray, "6 1"]  # Column vector 6x1

Matrix3x3 = Float[np.ndarray, "3 3"]  # 3x3 matrix
Matrix3x4 = Float[np.ndarray, "3 4"]  # 3x4 matrix
Matrix4x4 = Float[np.ndarray, "4 4"]  # 4x4 matrix
Matrix4x6 = Float[np.ndarray, "4 6"]  # 4x6 matrix


@dataclass
class Waypoint:
    """
    Représente un point de passage ou la position finale pour la trajectoire du robot en coordonnées cartésiennes.
    Position, vitesse et accélération sont tous des vecteurs colonnes (3x1).
    """

    position: Vector3x1
    velocity: Optional[Vector3x1] = None
    acceleration: Optional[Vector3x1] = None

    def __post_init__(self):
        # Initialisation des vitesses et accélérations à zéro si elles ne sont pas fournies
        if self.velocity is None:
            self.velocity = np.zeros((3, 1))
        if self.acceleration is None:
            self.acceleration = np.zeros((3, 1))

        # Vérification des types et des dimensions
        self._validate_vector("position", self.position)
        self._validate_vector("velocity", self.velocity)
        self._validate_vector("acceleration", self.acceleration)

    def _validate_vector(self, name: str, value: Vector3x1):
        # Vérification du type de base
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} doit être un numpy.ndarray, pas {type(value)}")

        # Vérification des dimensions (Shape)
        if value.shape != (3, 1):
            raise ValueError(f"{name} doit avoir la forme (3, 1), actuelle : {value.shape}")


class CommandEnum(Enum):
    """
    Represent the command type : pwm or torques
    """

    PWM = 0
    TORQUES = 1


@dataclass
class DoNothing:
    """Classe informant que la mission est de ne rien faire, utilisée pour les missions DoNothingUntilConditionMission."""

    pass
