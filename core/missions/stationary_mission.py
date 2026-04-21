"""
Module contenant la classe StationaryMission, qui représente une mission où le robot doit rester immobile jusqu'à nouvel ordre.
"""

import numpy as np

from core.missions.base_mission import Mission
from utils.types import Waypoint


class StationaryMission(Mission):
    """
    Mission de stationnarité : le robot doit rester immobile à sa position actuelle.
    Utile pour faire une pause ou attendre une condition avant de lancer une autre mission.
    """

    def __init__(self, ini_waypoint=None):
        super().__init__()
        if ini_waypoint is not None:
            self.check_waypoint_validity(
                ini_waypoint
            )  # Vérifie que le waypoint initial est valide pour une mission de stationnarité
        self.ini_waypoint = ini_waypoint

    def finish_condition(self, t, waypoint_mes):
        """La mission de stationnarité ne se termine jamais d'elle-même, elle doit être stoppée manuellement pour passer à la mission suivante."""
        return False

    def set_ini_waypoint(self, waypoint):
        """Permet de définir le waypoint initial de la mission, nécessaire pour toutes les missions."""
        # self.check_waypoint_validity(waypoint)
        # self.ini_waypoint = waypoint
        new_waypoint = Waypoint(
            position=waypoint.position,
            velocity=np.array([0.0, 0.0, 0.0]).reshape(3, 1),
            acceleration=np.array([0.0, 0.0, 0.0]).reshape(3, 1),
        )
        print("Setting initial waypoint for StationaryMission:", new_waypoint)
        self.ini_waypoint = new_waypoint

    def get_waypoint_at_t(self, t):
        """Retourne un waypoint avec la position actuelle du robot et des vitesses/accélérations nulles."""
        if self.ini_waypoint is None:
            raise ValueError("Waypoint initial non défini pour la mission de stationnarité.")
        return self.ini_waypoint

    def check_waypoint_validity(self, waypoint):
        """Vérifie que le waypoint est valide pour une mission de stationnarité (vitesses et accélérations nulles)."""
        if waypoint is None:
            raise ValueError("Waypoint ne peut pas être None pour une mission de stationnarité.")
        if waypoint.velocity is not None and not np.allclose(waypoint.velocity, 0, atol=0.05):
            print("Waypoint velocity:", waypoint.velocity)
            raise ValueError("Pour une mission de stationnarité, la vitesse doit être nulle.")
        if waypoint.acceleration is not None and not np.allclose(
            waypoint.acceleration, 0, atol=0.05
        ):
            raise ValueError("Pour une mission de stationnarité, l'accélération doit être nulle.")
