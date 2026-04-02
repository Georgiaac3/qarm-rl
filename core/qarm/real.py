import logging
import socket
import struct
import time
from queue import Queue

import cv2
import numpy as np

# import pyrealsense2 as rs
from numpy.typing import NDArray

from core.config import settings
from core.dynamics import get_pwm, l1, l2, l3, transform_angles
from core.missions.stationary_mission import StationaryMission
from core.qarm.interface import QARMInterface
from utils.types import Waypoint


class QARMReal(QARMInterface):
    """Implémentation pour le bras robotique réel avec communication UDP et caméra RealSense."""

    def __init__(self):
        # Connexion UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", settings.udp_port_recv))
        self.sock.setblocking(False)

        self.last_packet = None  # (0.0,) * 8 -> 4 first coordonnates for the angles, 4 last coordonnates for speeds

        # Caméra avec RealSense
        # self.pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(
        #    rs.stream.color,
        #    settings.camera_width,
        #    settings.camera_height,
        #    rs.format.bgr8,
        #    settings.camera_fps,
        # )
        # config.enable_stream(
        #    rs.stream.depth,
        #    settings.camera_width,
        #    settings.camera_height,
        #    rs.format.z16,
        #    settings.camera_fps,
        # )
        # self.pipeline.start(config)

        # Missiions à exécuter
        self.missions = Queue()

        # PID gains et damping pour l'inversion du Jacobien
        self.I3 = np.eye(3)  # Matrice identité 3x3 pré-allouée pour le calcul du Jacobien
        self.Kp = np.diag(
            [0, 0, 0]
        )  # Gains proportionnels pour le contrôle en position, matrice diagonale pour un contrôle indépendant sur chaque axe (3x3)
        self.Kd = np.diag(
            [0, 0, 0]
        )  # Gains dérivatifs pour le contrôle en vitesse, matrice diagonale pour un contrôle indépendant sur chaque axe (3x3)
        self.lambda_damping = 0.05  # Facteur de damping pour l'inversion du Jacobien

    # -------------------- Lecture angles --------------------
    def read_angles(self):
        """Retourne les angles phi mesurés des moteurs du robot, ou None si aucune donnée n'est disponible."""
        if self.last_packet:
            # print("Angles lus:", self.last_packet[:4])
            return self.last_packet[:4]
        return None

    # -------------------- Lecture vitesses --------------------
    def read_speeds(self):
        """Retourne les vitesses angulaires dphi mesurées des moteurs du robot, ou None si aucune donnée n'est disponible."""
        if self.last_packet:
            return self.last_packet[4:]
        return None

    # -------------------- Lecture packets --------------------
    def update_packet(self):
        """
        Lit les données UDP entrantes et met à jour le dernier packet reçu.
        Le packet attendu est de 64 bytes, contenant 8 doubles (4 pour les angles, 4 pour les vitesses).
        """

        try:
            data, _ = self.sock.recvfrom(1024)
            if len(data) == 64:
                self.last_packet = struct.unpack("8d", data)
            else:
                logging.info("Received data of unexpected size: %d bytes", len(data))
                self.last_packet = None
        except BlockingIOError:
            logging.info("No data received")
        except ConnectionResetError:
            logging.info("Connection reset by peer")

    # -------------------- Envoi commandes --------------------
    def send_speeds(self, v: list, grip: float) -> None:
        try:
            message_bytes = struct.pack("ddddd", v[0], v[1], v[2], v[3], grip)
            self.sock.sendto(message_bytes, (settings.udp_ip, settings.udp_port_send))
        except (OSError, struct.error) as e:
            logging.error("Erreur UDP envoi: %s", e)

    # -------------------- Lecture caméra --------------------
    def read_camera(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    # -------------------- Connexion --------------------
    def connect(self):
        """
        Attend la connexion du robot en envoyant périodiquement des commandes de vitesse nulle jusqu'à ce que des angles soient reçus.
        """
        print(
            "\n"
            "################################\n"
            "#    Tentative de connexion    #\n"
            "################################"
        )
        connexion = False
        while not connexion:
            self.send_speeds([0.0, -0.1, -0.1, 0.0], 0)
            self.update_packet()
            angles = self.read_angles()
            if angles is not None:
                print("Connexion établie, angles initiaux:", angles)
                connexion = True
            else:
                print("En attente de connexion...")
                time.sleep(1)

    # -------------------- Fermeture --------------------
    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.sock.close()

    # -------------------- Mission initiale --------------------
    def init_stationnary(self):
        """
        Empile une mission de stationnarité pour que le robot reste immobile à sa position actuelle.
        """
        waiting_mission = StationaryMission()
        self.missions.put(waiting_mission)
        while waiting_mission.ini_waypoint is None:
            print("En attente de la position actuelle du robot pour renseigner ini_waypoint...")
            self.update_packet()
            angles_phi = self.read_angles()
            if angles_phi is not None:
                q_mes, _, _ = transform_angles(
                    np.array(angles_phi).reshape(4, 1), np.zeros((4, 1)), np.zeros((4, 1))
                )
                X_mes = self.forward_kinematics(q_mes)
                waiting_mission.ini_waypoint = Waypoint(position=X_mes)
            time.sleep(0.01)
        print("Mission de stationnarité initialisée avec la position actuelle du robot.")

    ####
    # -------------------- Contrôle en position --------------------
    ####

    def get_jacobian(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule le jacobien (J) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1 = np.cos(q[0, 0])
        s1 = np.sin(q[0, 0])
        c2 = np.cos(q[1, 0])
        s2 = np.sin(q[1, 0])
        c23 = np.cos(q[1, 0] + q[2, 0])
        s23 = np.sin(q[1, 0] + q[2, 0])
        J = np.array(
            [
                [-l2 * s1 * c2 + l3 * s1 * s23, -l2 * c1 * s2 - l3 * c1 * c23, -l3 * c1 * c23, 0],
                [l2 * c1 * c2 - l3 * c1 * s23, -l2 * s1 * s2 - l3 * s1 * c23, -l3 * s1 * c23, 0],
                [0, -l2 * c2 + l3 * s23, l3 * s23, 0],
            ]
        )

        return J

    def get_djacobian(self, q: NDArray[np.float64], dq: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule la dérivée du jacobien (dJ) de la cinématique directe du robot.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - dq : vecteur colonne des vitesses articulaires de la dynamique géométrique (4,1)
        - retourne : dérivée du jacobien (3,4)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """

        if q.shape != (4, 1) or dq.shape != (4, 1):
            raise ValueError("q et dq doivent être des vecteurs colonne de dimension (4, 1)")

        dq1 = dq[0, 0]
        dq2 = dq[1, 0]
        dq3 = dq[2, 0]

        c1 = np.cos(q[0, 0])
        s1 = np.sin(q[0, 0])
        c2 = np.cos(q[1, 0])
        s2 = np.sin(q[1, 0])
        c23 = np.cos(q[1, 0] + q[2, 0])
        s23 = np.sin(q[1, 0] + q[2, 0])
        dJ = np.array(
            [
                [
                    (-l2 * c1 * c2 + l3 * c1 * s23) * dq1
                    + (l2 * s1 * s2 + l3 * s1 * c23) * dq2
                    + (l3 * s1 * c23) * dq3,
                    (l2 * s1 * s2 + l3 * s1 * c23) * dq1
                    + (-l2 * c1 * c2 + l3 * c1 * s23) * dq2
                    + (l3 * c1 * s23) * dq3,
                    l3 * s1 * c23 * dq1 + (l3 * c1 * s23) * dq2 + (l3 * c1 * s23) * dq3,
                    0,
                ],
                [
                    (-l2 * s1 * c2 + l3 * s1 * s23) * dq1
                    + (-l2 * c1 * s2 - l3 * c1 * c23) * dq2
                    + (-l3 * c1 * c23) * dq3,
                    (-l2 * c1 * s2 - l3 * c1 * c23) * dq1
                    + (-l2 * s1 * c2 + l3 * s1 * s23) * dq2
                    + (l3 * s1 * s23) * dq3,
                    -l3 * c1 * c23 * dq1 + (l3 * s1 * s23) * dq2 + (l3 * s1 * s23) * dq3,
                    0,
                ],
                [
                    0,
                    (l2 * s2 + l3 * c23) * dq2 + (l3 * c23) * dq3,
                    (l3 * c23) * dq2 + (l3 * c23) * dq3,
                    0,
                ],
            ]
        )

        return dJ

    def forward_kinematics(self, q: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calcule la position cartésienne de l'effecteur en fonction des angles articulaires q.
        - q : vecteur colonne des angles articulaires de la dynamique géométrique (4,1)
        - retourne : position cartésienne de l'effecteur (3,1)
        - les paramètres géométriques du robot sont définis dans core/config.py
        """
        if q.shape != (4, 1):
            raise ValueError("q doit être un vecteur colonne de dimension (4, 1)")

        c1 = np.cos(q[0, 0])
        s1 = np.sin(q[0, 0])
        c2 = np.cos(q[1, 0])
        s2 = np.sin(q[1, 0])
        c23 = np.cos(q[1, 0] + q[2, 0])
        s23 = np.sin(q[1, 0] + q[2, 0])
        x = l2 * c1 * c2 - l3 * c1 * s23
        y = l2 * s1 * c2 - l3 * s1 * s23
        z = l1 - l2 * s2 - l3 * c23
        return np.array([[x], [y], [z]])

    def update(
        self,
        t: float,
        phi_mes: NDArray[np.float64],
        dphi_mes: NDArray[np.float64],
    ) -> None:
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

        if phi_mes.shape != (4, 1) or dphi_mes.shape != (4, 1):
            raise ValueError(
                "phi_mes et dphi_mes doivent être des vecteurs colonne de dimension (4, 1)"
            )

        # Position et vitesses articulaires et cartésiennes mesurées
        q_mes, dq_mes, _ = transform_angles(phi_mes, dphi_mes, np.zeros_like(phi_mes))

        J = self.get_jacobian(q_mes)
        X_mes = self.forward_kinematics(q_mes)
        dX_mes = J @ dq_mes

        # Si aucune mission, rester stationnaire -> cela ajoute une mission de stationnarité à la queue
        if self.missions.empty():
            self.init_stationnary()

        # Récupération de la prochaine mission (la plus ancienne ajoutée)
        current_mission = self.missions.queue[0]

        if current_mission.start_time is not None:
            should_finish = current_mission.finish_condition(
                t - current_mission.start_time, Waypoint(position=X_mes, velocity=dX_mes)
            )
            if should_finish:
                self.missions.get()  # Retirer la mission terminée de la queue
                if not self.missions.empty():
                    current_mission = self.missions.queue[0]  # Passer à la mission suivante
                else:
                    self.init_stationnary()  # Si plus de mission, rester stationnaire

        if current_mission.start_time is None:
            current_mission.start_time = t
            current_mission.set_ini_waypoint(Waypoint(position=X_mes, velocity=dX_mes))

        ############################
        # Traitement de la mission #
        ############################

        ## Calcul du PWM a envoyer au robot pour suivre la trajectoire définie par la mission à l'instant t
        # 1. Obtenir le waypoint de consigne à l'instant t
        waypoint_desired = current_mission.get_waypoint_at_t(t - current_mission.start_time)
        X_des = waypoint_desired.position
        dX_des = waypoint_desired.velocity
        ddX_des = waypoint_desired.acceleration

        # 3. Commande et inversion
        ddX_cmd = ddX_des + self.Kp @ (X_des - X_mes) + self.Kd @ (dX_des - dX_mes)

        # Damped Least Squares pour l'inversion du Jacobien
        dJ = self.get_djacobian(q_mes, dq_mes)
        J_dag = J.T @ np.linalg.pinv(
            J @ J.T + (self.lambda_damping**2) * self.I3
        )  # Pseudo-inverse avec damping
        ddq_cmd = J_dag @ (ddX_cmd - dJ @ dq_mes)  # Commande en accélération articulaire

        # 4. Dynmique du bras et envoi des commandes pwwm
        mL = (
            current_mission.load if current_mission.load is not None else 0.0
        )  # Charge utile, à intégrer dans la dynamique
        tau_cmd = get_pwm(
            q_mes, dq_mes, ddq_cmd, mL
        )  # Convertir les accélérations commandées en commandes de couple (PWM)

        self.send_speeds(
            tau_cmd.flatten().tolist(), 0
        )  # Envoi des commandes de vitesse (PWM) au robot

        # TODO: Affichage de la caméra, gestion des erreurs, etc.
