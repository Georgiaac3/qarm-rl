"""
Module de définition de types de données.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Waypoint:
    """
    Représente un point de passage ou la position finale pour la trajectoire du robot en coordonnées cartésiennes.
    Position, vitesse et accélération sont tous des vecteurs colonnes (3x1).
    """

    position: NDArray[np.float64]
    velocity: Optional[NDArray[np.float64]] = None
    acceleration: Optional[NDArray[np.float64]] = None

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

    def _validate_vector(self, name: str, value: NDArray):
        # Vérification du type de base
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name} doit être un numpy.ndarray, pas {type(value)}")

        # Vérification des dimensions (Shape)
        if value.shape != (3, 1):
            raise ValueError(f"{name} doit avoir la forme (3, 1), actuelle : {value.shape}")


@dataclass
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
