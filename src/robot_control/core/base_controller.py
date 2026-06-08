"""
Contrôleur commun pour simulation ou robot réel.
"""

import time
from abc import ABC, abstractmethod
from collections import deque

# Typing imports
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Custom imports
from robot_control import Dynamics, Kinematics
from robot_control.missions import StationaryMission
from robot_control.utils import Waypoint, robot_says, robot_says_phase


def connect_decorator(func):
    """Decorator to ensure that the connection is established before executing the decorated method."""

    def wrapper(*args, **kwargs):
        robot_says_phase("Tentative de connexion")
        func(*args, **kwargs)

    return wrapper


class Controller(Dynamics, Kinematics, ABC):
    """Contrôleur commun pour simulation ou robot réel."""

    def __init__(self):
        ###############################
        # Communication with the robot
        self.last_packet: Optional[list[float]] = None  # Last packet received from the robot

        ############
        # Missions
        self.missions = deque()

    #################################################################
    # All the abstract methods are for communication with the robot
    @abstractmethod
    def _update_packet(self):
        """
        This method receive data from the robot, and update self.last_packet with it, whether it is with udp, tcp, or simulation dedicated methods.
        It should set self.last_packet to None if no data is received, or if the data is not in the expected format.
        """

    @abstractmethod
    def _send_command(self, cmd):
        """Sends the command, PWM or torque + gripper for example"""

    @abstractmethod
    def _close(self):
        """Close the communications"""

    @connect_decorator
    @abstractmethod
    def connect(self):
        """Establishes the initial connection with the robot. Real or simulation robot might need different routines to establish the connection, hence the abstract method."""

    def _read_camera(self):
        """Read the camera of the arm"""
        # TODO : It will use last_packet so it might be implemented in this parent class.
        pass

    ###################
    # Missions helpers
    def init_stationnary(self):
        """
        Stacks a stationary mission so that the robot remains stationary at its current position.
        """
        waiting_mission = StationaryMission()
        self.missions.append(waiting_mission)
        robot_says("Stationary mission added to the queue.")

    #########
    # Logics
    @abstractmethod
    def compute_command(
        self,
        t: float,
    ) -> Optional[np.ndarray]:
        """
        Gives back the command to sent to the robot based on the current mission and the robot's measured status.
        It should use update_and_get_mission to update the missions queue and get the current mission to execute, then compute the command to execute this mission based on the measured state of the robot.
        """

    def update_and_get_mission(self, t, X_mes, dX_mes):
        """
        Perfoms the logic of updating the missions queue and returns the current mission to execute.
        Work with cartesian coordinates as a state for know.
        TODO : this logic should take only a mesured state as input, not specific cartesian coordinates. However the missions work with cartesian coordinates for now.
        """
        # Si aucune mission, rester stationnaire -> cela ajoute une mission de stationnarité à la queue
        if len(self.missions) == 0:
            self.init_stationnary()

        # Récupération de la prochaine mission (la plus ancienne ajoutée)
        current_mission = self.missions[0]

        if current_mission.start_time is not None:
            should_finish = current_mission.finish_condition(
                t - current_mission.start_time, Waypoint(position=X_mes, velocity=dX_mes)
            )
            if should_finish:
                self.missions[0].say_goodbye()  # Message de fin de mission
                self.missions.popleft()  # Retirer la mission terminée de la queue
                if len(self.missions) == 0:
                    self.init_stationnary()  # Si plus de mission, rester stationnaire
                current_mission = self.missions[0]  # Passer à la mission suivante

        if current_mission.start_time is None:
            current_mission.start_time = t
            current_mission.set_ini_waypoint(Waypoint(position=X_mes, velocity=dX_mes))
            current_mission.say_hello()

        return current_mission

    def go(self):
        """Boucle de contrôle principale du robot. Lit les données des capteurs, met à jour les missions en cours et envoie les commandes au robot à une fréquence définie."""

        start_time = time.perf_counter()
        next_tick = start_time + self.timestep

        robot_says_phase("Main control loop of the robot")

        while True:
            self._update_packet()

            phi = self._read_joint_angles()
            dphi = self._read_joint_speeds()

            if phi is not None and dphi is not None:
                cmd = self.compute_command(time.perf_counter() - start_time, phi, dphi)
                self._send_command(cmd)

            # affichage ou non -><-

            while time.perf_counter() < next_tick:
                pass
