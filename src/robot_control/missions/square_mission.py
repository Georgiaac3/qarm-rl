import numpy as np

from robot_control.utils import Waypoint

from .trajectory_mission import MultiTrajectoryMission


class SquareMission(MultiTrajectoryMission):

    def __init__(
        self,
        center: np.ndarray,
        side_length: float,
        plane: np.ndarray,
        nb_of_squares: int,
        time_per_side: float,
    ):
        self.center = center
        self.side_length = side_length
        self.plane = plane
        self.nb_of_squares = nb_of_squares
        self.time_per_side = time_per_side

        super().__init__(waypoints=self._calculate_square_waypoints())
        self.ini_waypoint = Waypoint(
            position=center.reshape(3, 1), velocity=None, acceleration=None
        )

    def _calculate_square_waypoints(self):
        """Calcule les waypoints du carré en fonction du centre, de la longueur des côtés et du plan."""
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
            ref_vector = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            ref_vector /= np.linalg.norm(ref_vector)

        # Calcul des waypoints
        half_side = self.side_length / 2
        corners = [
            center + half_side * ref_vector + half_side * np.cross(normal, ref_vector),
            center - half_side * ref_vector + half_side * np.cross(normal, ref_vector),
            center - half_side * ref_vector - half_side * np.cross(normal, ref_vector),
            center + half_side * ref_vector - half_side * np.cross(normal, ref_vector),
        ]

        waypoints = []
        for _ in range(self.nb_of_squares):
            for i in range(4):
                waypoint = Waypoint(
                    position=corners[i].reshape(3, 1), velocity=None, acceleration=None
                )
                waypoints.append(waypoint)

        return waypoints
