"""
Contrôleur commun pour simulation ou robot réel.
"""

from abc import ABC, abstractmethod
from collections import deque

# Typing imports
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from src.core.dynamics.base_dynamics import Dynamics

# Custom imports
from src.core.kinematics.base_kinematics import Kinematics
from src.core.missions.stationary_mission import StationaryMission
from src.utils.logger import robot_says, robot_says_phase
from src.utils.types import CommandEnum


def connect_decorator(func):
    """Decorator to ensure that the connection is established before executing the decorated method."""

    def wrapper(*args, **kwargs):
        robot_says_phase("Tentative de connexion")
        func(*args, **kwargs)

    return wrapper


class PIDController(ABC):
    """Contrôleur commun pour simulation ou robot réel."""

    def __init__(
        self,
        dynamics: Dynamics,
        kinematics: Kinematics,
        command_type: CommandEnum,
        Kp,
        Kd,
        Ki,
    ):
        self.dynamics = dynamics
        self.kinematics = kinematics
        self.command_type = command_type

        ###############################
        # Communication with the robot
        self.last_packet: Optional[list[float]] = None  # Last packet received from the robot

        ############
        # Missions
        self.missions = deque()

        #######
        # PID
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integral_error = np.zeros(
            (3, 1)
        )  # Terme intégral initialisé à zéro pour le contrôle en position
        self.lambda_damping = 0.05  # Facteur de damping pour l'inversion du Jacobien

        ########################################
        # Affichage dans l'interface graphique
        self.last_X_mes: Optional[NDArray[np.float64]] = None
        self.last_X_des: Optional[NDArray[np.float64]] = None
        self.last_pwm: Optional[list] = None

    #################################################################
    # All the abstract methods are for communication withe the robot
    @abstractmethod
    def _update_packet(self):
        """
        This method receive data from the robot, and update self.last_packet with it, whether it is with udp, tcp, or simulation dedicated methods.
        Type of last_packet: (0.0,) * 8 -> 4 first coordonnates for the angles, 4 last coordonnates for speeds
        """

    @abstractmethod
    def _send_command(self, cmd):
        """Sends the command, PWM or torque + gripper"""

    @abstractmethod
    def _close(self):
        """Close the communications"""

    @connect_decorator
    @abstractmethod
    def connect(self):
        """Establishes the initial connection with the robot. Real or simulation robot might need different routines to establish the connection, hence the abstract method."""

    #################
    # Communication
    def _read_joint_angles(self):
        """Returns the measured phi angles of the robot's motors, or None if no data is available."""
        if self.last_packet:
            return self.last_packet[:4]
        return None

    def _read_joint_speeds(self):
        """Returns the measured speeds of the robot's motors, or None if no data is available."""
        if self.last_packet:
            return self.last_packet[4:8]
        return None

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
    def run(self):
        """Boucle de contrôle principale du robot. Lit les données des capteurs, met à jour les missions en cours et envoie les commandes au robot à une fréquence définie."""
        while True:
            self._update_packet()
