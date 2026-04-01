"""
Module pour définir la classe abstraite de mission pour le bras robotique QARM.
Les missions n'ont pas pour vocation de gérer la boucle de contrôle, mais simplement de fournir des points de consigne (position, vitesse, accélération) à suivre à chaque instant t.
"""

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from utils.types import Waypoint


class Mission(ABC):
    """
    Classe abstraite représentant une mission pour le bras robotique QARM.
    """

    def __init__(self):
        self.start_time = (
            None  # Temps de début de la mission, à initialiser lors de l'empilement de la mission
        )
        self.finished = False  # Indique si la mission est terminée

    def stop(self):
        """Permet de terminer la mission et de passer à la mission suivante."""
        self.finished = True

    def is_finished(self):
        """Indique si la mission est terminée."""
        return self.finished

    @abstractmethod
    def finish_condition(self, t: float, waypoint_mes: Waypoint) -> bool:
        """Condition de fin de la mission, à implémenter selon les besoins spécifiques de chaque mission."""

    @abstractmethod
    def set_ini_waypoint(self, waypoint: Waypoint):
        """Permet de définir le waypoint initial de la mission, nécessaire pour toutes les missions."""

    @abstractmethod
    def get_waypoint_at_t(self, t: float) -> Waypoint:
        """Retourne le point de consigne (position, vitesse, accélération) à l'instant t."""
