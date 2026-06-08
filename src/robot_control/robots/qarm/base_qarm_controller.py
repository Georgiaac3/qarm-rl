from abc import ABC

import numpy as np

from robot_control.core import Controller

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
        Kp: np.ndarray,
        Kd: np.ndarray,
        Ki: np.ndarray,
    ):
        super().__init__()

        self.timestep = timestep

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

    def compute_command(
        self,
        t: float,
    ) -> np.ndarray:
        """
        Met à jour la commande envoyée au robot en fonction de la mission courante et de l'état mesuré du robot.
        - t : temps actuel en secondes
        - phi_mes : vecteur colonne des angles articulaires mesurés du moteur (4,1)
        - dphi_mes : vecteur colonne des vitesses articulaires mesurées du moteur (4,1)
        - retourne : None, mais envoie les commandes de vitesse (PWM) au robot via self.send_speeds()
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

        # 4. Dynmique du bras et envoi des commandes pwwm
        mL = (
            current_mission.load if current_mission.load is not None else 0.0
        )  # Charge utile, à intégrer dans la dynamique

        # 1. Calcul des matrices et vecteurs dynamiques à partir des angles géométriques mesurés
        M = self.inertia_matrix(q_geo_mes, mL)
        C = self.centrifugal_matrix(q_geo_mes, mL)
        G = self.gravity_vector(q_geo_mes, mL)
        B = self.coriolis_matrix(q_geo_mes, mL)

        friction = self.friction_vector(dq_geo_mes)

        B_signals = self.coriolis_velocity_signals(dq_geo_mes)

        # 2. Calcul du torque total à appliquer
        tau_cmd = M @ ddq_geo_cmd
        tau_cmd += B @ B_signals
        tau_cmd += C @ dq_geo_mes**2
        tau_cmd += G
        tau_cmd += friction

        if self.command_type == CommandEnum.TORQUES:
            return tau_cmd

        # 3. Conversion du torque en signal de tension (V) à envoyer au moteur
        Vcmd = (self.R / self.ktGR) * tau_cmd + self.kvGR * dq_geo_mes

        Vcmd[3] = 0.0

        # 4. Normalisation du signal de tension entre -1 et 1
        pwm = np.clip(Vcmd / self.Valim, -1, 1)

        if np.any(Vcmd / self.Valim > 1) or np.any(Vcmd / self.Valim < -1):
            print(
                "Warning: Commande de tension dépassant les limites d'alimentation. Vcmd/Valim:",
                Vcmd.ravel() / self.Valim,
            )

        cmd = cmd.ravel().tolist() + [0.0]

        ##############################################
        # Management of the variables for the display
        self.last_X_mes = X_mes  # Stocker la dernière position mesurée pour l'affichage dans l'interface graphique
        self.last_X_des = X_des  # Stocker la dernière position de consigne pour l'affichage dans l'interface graphique

        ###############################################
        # Return the raw command to be sent to the robot
        return cmd
        # self._send_command(cmd)
