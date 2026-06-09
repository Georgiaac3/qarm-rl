import queue
import time
from abc import ABC
from typing import Optional

import numpy as np

from robot_control.core import Controller
from robot_control.utils import CommandEnum, Matrix3x3, Vector5x1, robot_says_phase

from .qarm_dynamics import QArmDynamics
from .qarm_kinematics import QArmKinematics


class BaseQArmController(Controller, QArmDynamics, QArmKinematics, ABC):
    """
    Base class for QArm controllers, providing common methods and attributes for both real and simulated controllers.
    The Sim and Real Controllers are different in there communication methods.
    This is a PID controller.
    """

    def __init__(
        self,
        timestep: float,
        command_type: CommandEnum,
        Kp: Matrix3x3,
        Kd: Matrix3x3,
        Ki: Matrix3x3,
        display: bool = False,
        display_data_queue=None,
    ):
        super().__init__()

        self.timestep = timestep
        self.command_type = command_type  # PWM or TORQUES

        ######################
        # Kinetics parameters
        self.I3 = np.eye(3)  # Matrice d'identité pour l'inversion du Jacobien
        self.lambda_damping = 0.05  # Facteur de damping pour l'inversion du Jacobien

        #######
        # PID
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.integral_error = np.zeros(
            (3, 1)
        )  # Terme intégral initialisé à zéro pour le contrôle en position

        ############################
        # Display setup
        self.display = display
        if display:
            self.display_data_queue = display_data_queue

            self.last_phi_mes = np.zeros((4, 1))
            self.last_dphi_mes = np.zeros((4, 1))
            self.last_X_mes = np.zeros((3, 1))
            self.last_X_des = np.zeros((3, 1))
            self.last_dX_mes = np.zeros((3, 1))
            self.last_dX_des = np.zeros((3, 1))

    def compute_command(
        self,
        t: float,
    ) -> Optional[Vector5x1]:
        """
        Met à jour la commande envoyée au robot en fonction de la mission courante et de l'état mesuré du robot.
        - t : temps actuel en secondes
        - Retourne : None ou un vecteur de commande de dimension 5 (4 pour les moteurs + 1 pour le gripper)
        - la logique de contrôle est la suivante :
            1. Si aucune mission, rester stationnaire -> cela ajoute une mission de stationnarité à la queue
            2. Récupération de la prochaine mission (la plus ancienne ajoutée)
            3. Si la mission a une condition de fin et qu'elle est remplie, la retirer de la queue et passer à la mission suivante (ou rester stationnaire si plus de mission)
            4. Si la mission n'a pas de waypoint initial, l'initialiser avec la position actuelle du robot
            5. Calcul du PWM a envoyer au robot pour suivre la trajectoire définie par la mission à l'instant t :
                - Obtenir le waypoint de consigne à l'instant t
                - Calculer la commande en accélération cartésienne avec un PD en position et vitesse
                - Convertir la commande en accélération cartésienne en commande en accélération articulaire avec l'inversion du Jacobien (Damped Least Squares)
                - Convertir la commande en accélération articulaire en commande de couple (PWM) avec la dynamique du robot (fonction get_pwm) et en tenant compte de la charge utile de la mission
                - Envoyer les commandes de vitesse (PWM) au robot avec self.send_speeds
        """
        #
        # TODO : should take into account the noise
        #

        ##########################################
        # Getting the measured state of the robot
        if self.last_packet is None:
            return None  # Pas de données mesurées disponibles, ne rien faire

        phi_mes = self.last_packet[0:4]
        dphi_mes = self.last_packet[4:8]

        if phi_mes is None or dphi_mes is None:
            return None  # Pas de données mesurées disponibles, ne rien faire

        if len(phi_mes) != 4 or len(dphi_mes) != 4:
            raise ValueError(
                f"phi_mes et dphi_mes n'ont pas la bonne dimension. phi_mes: {phi_mes}, dphi_mes: {dphi_mes}"
            )

        phi_mes = np.array(phi_mes).reshape(4, 1)
        dphi_mes = np.array(dphi_mes).reshape(4, 1)

        ###########################################################################################################################
        # Getting the geometric angles and speeds from the motor angles and speeds and the cartersian position of the end-effector
        q_mes, dq_mes, _ = self.transform_angles(phi_mes, dphi_mes, np.zeros_like(phi_mes))

        J = self.jacobian(q_mes)
        X_mes = self.forward_kinematics(q_mes)
        dX_mes = J @ dq_mes

        ###################
        # Logique missions
        current_mission = self.update_and_get_mission(t, X_mes, dX_mes)
        if current_mission.start_time is None:
            raise ValueError(
                f"Start time of the mission {current_mission} is None, but it should have been initialized in update_and_get_mission()."
            )

        ########################
        # Computing the command
        # Get the target waypoint at time t
        waypoint_desired = current_mission.get_waypoint_at_t(t - current_mission.start_time)

        X_des = waypoint_desired.position
        dX_des = waypoint_desired.velocity
        ddX_des = waypoint_desired.acceleration

        # PID
        ddX_cmd = (
            ddX_des
            + self.Kp @ (X_des - X_mes)
            + self.Kd @ (dX_des - dX_mes)
            + self.Ki @ self.integral_error
        )
        self.integral_error += (X_des - X_mes) * self.timestep

        # Damped Least Squares for the inversion of the Jacobian
        dJ = self.djacobian(q_mes, dq_mes)
        J_dag = J.T @ np.linalg.pinv(
            J @ J.T + (self.lambda_damping**2) * self.I3
        )  # Pseudo-inverse with damping
        ddq_cmd = J_dag @ (ddX_cmd - dJ @ dq_mes)  # Commande en accélération articulaire

        # Payload for the current mission
        mL = current_mission.load if current_mission.load is not None else 0.0

        # Computing of dynamic matrices and vectors based on measured geometric angles and payload
        M = self.inertia_matrix(q_mes, mL)
        C = self.centrifugal_matrix(q_mes, mL)
        G = self.gravity_vector(q_mes, mL)
        B = self.coriolis_matrix(q_mes, mL)

        friction = self.friction_vector(dq_mes)

        B_signals = self.coriolis_velocity_signals(dq_mes)

        # Total torque to be applied
        tau_cmd = M @ ddq_cmd
        tau_cmd += B @ B_signals
        tau_cmd += C @ dq_mes**2
        tau_cmd += G
        tau_cmd += friction

        GR = 0.0  # TODO : 272.5 # Attention ce n'est pas le même GR pour le joint 4
        Jmotor = np.array([[0.28, 0.28 * 2, 0.28 / 2, 0]]).T
        tau_k = 0.05
        mu_v = 0.5

        inertie = Jmotor * GR * ddq_cmd
        frottements = mu_v * GR * dq_mes + tau_k * np.sign(dq_mes)
        backFEM = self.kvGR * dq_mes

        # Conversion of torque into a voltage signal (V) to be sent to the motor
        # !! The GR (Gear Ratio) come from the fact that the equation here take into account the motor shaft
        # The fact that the 2nd joitn have two motors should also be taken into account (1/2 in tau_cmd), however it doesn't seem to work
        Vcmd = (self.R / self.ktGR) * (inertie * GR + frottements * GR + tau_cmd) + backFEM

        # TODO : Take into account tau_s for the deadand (motors don't work under 0.2 V...)
        # doesn't seems very helpful
        # tau_s = 0.2
        # relevant = tau_s/10

        # Vcmd = np.where(
        #    (np.abs(Vcmd) < tau_s) & (np.abs(Vcmd) > relevant),
        #    np.sign(Vcmd) * tau_s,
        #    Vcmd
        # )

        Vcmd[3, 0] = (
            0.0  # No command on the gripper for now, TODO : manage the gripper command in the missions and here
        )
        tau_cmd[3, 0] = (
            0.0  # No command on the gripper for now, TODO : manage the gripper command in the missions and here
        )

        # Normalization of the voltage signal between -1 and 1
        pwm = np.clip(Vcmd / self.Valim, -1, 1)

        if np.any(Vcmd / self.Valim > 1) or np.any(Vcmd / self.Valim < -1):
            print(
                "Warning: Commande de tension dépassant les limites d'alimentation. Vcmd/Valim:",
                Vcmd.ravel() / self.Valim,
            )

        ##############################################
        # Management of the variables for the display
        if self.display:
            self.last_phi_mes = phi_mes
            self.last_dphi_mes = dphi_mes
            self.last_X_des = X_des
            self.last_X_mes = X_mes
            self.last_dX_mes = dX_mes
            self.last_dX_des = dX_des

        ###############################################
        # Return the raw command to be sent to the robot
        gripper_command = np.array(
            [[0.0]]
        )  # TODO : manage the gripper command in the missions and here
        if self.command_type == CommandEnum.TORQUES:
            return np.vstack((tau_cmd, gripper_command))
        return np.vstack((pwm, gripper_command))

    def go(self):
        """
        Boucle de contrôle principale du robot. Lit les données des capteurs, met à jour les missions en cours et envoie les commandes au robot à une fréquence définie.
        """

        start_time = time.perf_counter()
        next_tick = start_time + self.timestep

        robot_says_phase("Main control loop of the robot")

        while True:
            self._update_packet()

            cmd = self.compute_command(time.perf_counter() - start_time)
            if cmd is not None:
                self._send_command(cmd)

            if self.display:
                self._display_variables(time.perf_counter() - start_time)

            while time.perf_counter() < next_tick:
                pass  # Seems to be the best way to have a precise timestep, sleeping is not precise enough

            next_tick += self.timestep

    def _display_variables(self, t):
        """Affiche les variables de contrôle dans la console et les envoie à une interface graphique via une queue."""
        if self.display_data_queue is not None:
            if (
                self.last_X_mes is not None
                and self.last_X_des is not None
                and self.last_dX_mes is not None
                and self.last_dX_des is not None
            ):
                X_mes = self.last_X_mes.ravel().tolist()
                X_des = self.last_X_des.ravel().tolist()
                dX_mes = self.last_dX_mes.ravel().tolist()
                dX_des = self.last_dX_des.ravel().tolist()
                try:
                    self.display_data_queue.put(
                        {
                            "t_s": t,
                            "Angles Articulations mesurés (rad)": self.last_phi_mes.ravel().tolist(),
                            "Vitesses mesurées (rad/s)": self.last_dphi_mes.ravel().tolist(),
                            "X_mes": X_mes,
                            "X_des": X_des,
                            "dX_mes": dX_mes,
                            "dX_des": dX_des,
                            "x": [X_des[0], X_mes[0]],
                            "y": [X_des[1], X_mes[1]],
                            "z": [X_des[2], X_mes[2]],
                            "dx": [dX_des[0], dX_mes[0]],
                            "dy": [dX_des[1], dX_mes[1]],
                            "dz": [dX_des[2], dX_mes[2]],
                            "TCP_Trajectoire": self.last_X_mes.ravel().tolist(),
                            "Wanted_TCP_Trajectoire": self.last_X_des.ravel().tolist(),
                        },
                        block=False,
                    )
                except queue.Full:
                    pass
