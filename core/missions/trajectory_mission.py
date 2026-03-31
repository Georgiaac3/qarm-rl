"""
Module de mission de suivi de trajectoire multi-segment.
"""
from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from utils.types import Waypoint
from utils.trajectory import get_quintic_coeffs_and_time, get_desired_state
from core.missions.base_mission import Mission

class MultiTrajectoryMission(Mission):
    """
    Mission de suivi de trajectoire multi-segment.
    Chaque segment est défini par un Waypoint avec position, durée, et éventuellement vitesse/accélération.
    """
    def __init__(self, pos_start: NDArray, waypoints: Union[Waypoint, List[Waypoint]]):
        """
        pos_start: position initiale (3D). Les vitesses et accélérations initiales et finales (respectivement du premier et dernier waypoint) doivent être nulles par mesure de sécurité.
        waypoints: liste de Waypoint définissant les segments de la trajectoire
        """

        self.segments = []
        self.total_duration = 0.0

        current_pos = pos_start
        current_speed = np.zeros(3)
        current_acc = np.zeros(3)
        start_time = 0.0

        # Convertir waypoints en liste si ce n'est pas déjà le cas, i.e. si un seul Waypoint est fourni
        if not isinstance(waypoints, list):
            waypoints = [waypoints]

        for wp in waypoints:
            # Définition des conditions finales du segment
            target_speed = wp.velocity if wp.velocity is not None else np.zeros(3)
            target_acc = wp.acceleration if wp.acceleration is not None else np.zeros(3)

            # Calcul des coeffs pour ce segment précis
            coeffs, segment_duration = get_quintic_coeffs_and_time(
                current_pos, current_speed, current_acc,
                wp.position, target_speed, target_acc,
            )

            # On stocke le segment avec son intervalle de temps
            self.segments.append({
                'start_t': start_time,
                'end_t': start_time + segment_duration,
                'coeffs': coeffs
            })

            # Mise à jour pour le segment suivant
            start_time += segment_duration
            current_pos = wp.position
            current_speed = target_speed
            current_acc = target_acc

        self.total_duration = start_time

    def get_target(self, t):
        """
        Retourne la position, vitesse, et accélération désirées à l'instant t."""
        # 1. Trouver le bon segment en fonction de t
        for seg in self.segments:
            if seg['start_t'] <= t < seg['end_t']:
                # t_segment est le temps local au polynôme (entre 0 et duration)
                t_segment = t - seg['start_t']
                return get_desired_state(t_segment, seg['coeffs'])

        # 2. Si t dépasse la fin, on reste sur le dernier point
        last_coeffs = self.segments[-1]['coeffs']
        last_duration = self.segments[-1]['end_t'] - self.segments[-1]['start_t']
        return get_desired_state(last_duration, last_coeffs)

    def is_finished(self, t):
        """
        Condition d'arrêt : la mission est terminée lorsque t dépasse la durée totale de tous les segments.
        """
        return t >= self.total_duration
