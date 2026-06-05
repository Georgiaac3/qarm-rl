import numpy as np
from numpy.typing import NDArray

from src.core.config import settings
from src.core.missions.base_mission import Mission
from src.utils.types import Waypoint


class CircleMission(Mission):
    """Mission de suivi d'un cercle dans un plan défini par un vecteur normal."""

    def __init__(
        self,
        center: NDArray[np.float64],
        radius: float,
        plane: NDArray[np.float64],
        nb_of_circles: int,
        time_per_circle: float,
    ):
        """
        center: centre du cercle (3D)
        radius: rayon du cercle
        plane: vecteur normal au plan du cercle (3D)
        nb_of_circles: nombre de cercles à parcourir
        time_per_circle: temps en secondes pour parcourir un cercle complet
        """
        super().__init__()
        self.center = center.reshape(3, 1)
        self.radius = radius
        self.plane = plane.reshape(3, 1)
        self.nb_of_circles = nb_of_circles
        self.time_per_circle = time_per_circle

        # Calcul des waypoints du cercle
        self.waypoints = self._calculate_circle_waypoints()

        self.set_ini_waypoint(self.waypoints[0])

    def _calculate_circle_waypoints(self) -> NDArray[np.float64]:
        """Calcule les waypoints du cercle en fonction du centre, du rayon et du plan."""
        # Normalisation du vecteur normal
        normal = (self.plane / np.linalg.norm(self.plane)).reshape(
            3,
        )
        center = self.center.reshape(
            3,
        )

        # Choix d'un vecteur de référence dans le plan
        if np.allclose(normal, np.array([0.0, 0.0, 1.0])):
            ref_vector = np.array([1.0, 0.0, 0.0])
        else:
            print("Normal:", normal.shape)
            ref_vector = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            print(ref_vector)
            ref_vector /= np.linalg.norm(ref_vector)

        # Calcul des waypoints
        waypoints = []
        for _ in range(self.nb_of_circles):
            num_points = int(self.time_per_circle / settings.timestep)
            for t in np.linspace(0, self.time_per_circle, num=num_points):
                angle = (2 * np.pi * t) / self.time_per_circle
                vector_on_circle = self.radius * (
                    np.cos(angle) * ref_vector + np.sin(angle) * np.cross(normal, ref_vector)
                )
                waypoint_on_circle = Waypoint(
                    position=(center + vector_on_circle).reshape(3, 1),
                    velocity=np.cross(vector_on_circle, ref_vector).reshape(3, 1)
                    / self.time_per_circle,
                    acceleration=np.cross(
                        np.cross(vector_on_circle, ref_vector), ref_vector
                    ).reshape(3, 1)
                    / self.time_per_circle**2,
                )
                waypoints.append(waypoint_on_circle)
        return np.array(waypoints)

    def finish_condition(self, t, waypoint_mes):
        """La mission est terminée après avoir parcouru le nombre de cercles spécifié."""
        return t >= self.nb_of_circles * self.time_per_circle

    def set_ini_waypoint(self, waypoint):
        """Le waypoint initial est le premier point du cercle."""
        self.ini_waypoint = waypoint

    def get_waypoint_at_t(self, t):
        """Retourne le waypoint désiré à l'instant t."""
        return self.waypoints[int(t / settings.timestep) % len(self.waypoints)]
