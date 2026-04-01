"""
Module de mission de suivi de trajectoire multi-segment.
"""

from typing import List, Union

from core.missions.base_mission import Mission
from utils.trajectory import get_desired_state, get_quintic_coeffs_and_time
from utils.types import Waypoint


class MultiTrajectoryMission(Mission):
    """
    Mission de suivi de trajectoire multi-segment.
    Chaque segment est défini par un Waypoint avec position, durée, et éventuellement vitesse/accélération.
    """

    def __init__(
        self, waypoints: Union[Waypoint, List[Waypoint]], ini_waypoint: Waypoint | None = None
    ):
        """
        ini_waypoint: waypoint initial (3D). Les vitesses et accélérations initiales et finales (respectivement du premier et dernier waypoint) doivent être nulles par mesure de sécurité.
        waypoints: liste de Waypoint définissant les segments de la trajectoire
        """
        super().__init__()

        self.ini_waypoint = (
            ini_waypoint  # Peut être défini plus tard via set_ini_waypoint() si besoin
        )
        self.waypoints = waypoints if isinstance(waypoints, list) else [waypoints]

        self.segments: list[dict] = (
            []
        )  # Liste des segments de la trajectoire avec leurs coefficients et durées
        self.total_duration = 0.0

    def finish_condition(self, t, waypoint_mes):
        """
        La mission se termine lorsque le temps t dépasse la durée totale de la trajectoire.
        Note: on pourrait aussi ajouter une condition de proximité à la position finale pour plus de robustesse, mais cela peut être délicat à régler en pratique.
        """
        return t >= self.total_duration

    def set_ini_waypoint(self, waypoint):
        """Permet de définir le waypoint initial de la mission, nécessaire pour toutes les missions."""
        self.ini_waypoint = waypoint

    def get_waypoint_at_t(self, t):
        """
        Retourne la position, vitesse, et accélération désirées à l'instant t.
        """
        if len(self.segments) == 0:
            raise ValueError(
                "Segments de trajectoire non calculés. Veuillez appeler compute_trajectory() avant d'obtenir les waypoints."
            )

        # 1. Trouver le bon segment en fonction de t
        for seg in self.segments:
            if seg["start_t"] <= t < seg["end_t"]:
                # t_segment est le temps local au polynôme (entre 0 et segment_duration)
                t_segment = t - seg["start_t"]
                return get_desired_state(t_segment, seg["coeffs"])

        # 2. Si t dépasse la fin, on reste sur le dernier point
        last_coeffs = self.segments[-1]["coeffs"]
        last_duration = self.segments[-1]["duration"]
        return get_desired_state(last_duration, last_coeffs)

    def compute_trajectory(self):
        """
        Calcul des coefficients de tous les segments de la trajectoire à partir des waypoints.
        Chaque segment est un polynôme quintique calculé à partir des conditions initiales et finales (position, vitesse, accélération).
        """

        if self.ini_waypoint is None:
            raise ValueError(
                "Waypoint initial non défini. Veuillez appeler set_ini_waypoint() avant de calculer la trajectoire."
            )

        start_time = 0.0

        for i, wp in enumerate(self.waypoints):
            # Calcul des coeffs pour ce segment précis
            coeffs, segment_duration = get_quintic_coeffs_and_time(
                waypoint_start=self.ini_waypoint if i == 0 else self.waypoints[i - 1],
                waypoint_end=wp,
            )

            # On stocke le segment avec son intervalle de temps
            self.segments.append(
                {
                    "start_t": start_time,
                    "end_t": start_time + segment_duration,
                    "duration": segment_duration,
                    "coeffs": coeffs,
                }
            )

            # Mise à jour pour le segment suivant
            start_time += segment_duration

        self.total_duration = start_time
