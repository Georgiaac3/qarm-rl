import logging
import socket
import struct
import time
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from numpy.typing import NDArray

from core.config import settings
from core.qarm.interface import QARMInterface


class QARMReal(QARMInterface):
    """Implémentation pour le bras robotique réel avec communication UDP et caméra RealSense."""

    def __init__(self):
        # Connexion UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", settings.udp_port_recv))
        self.sock.setblocking(False)

        self.last_packet = None  # (0.0,) * 8 -> 4 first coordonnates for the angles, 4 last coordonnates for speeds

        # Caméra avec RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            settings.camera_width,
            settings.camera_height,
            rs.format.bgr8,
            settings.camera_fps,
        )
        config.enable_stream(
            rs.stream.depth,
            settings.camera_width,
            settings.camera_height,
            rs.format.z16,
            settings.camera_fps,
        )
        self.pipeline.start(config)

        # PID position command
        # 1. Paramètres Moteurs (SI)
        self.R = 2.5  # Ohms
        self.kt = 0.05  # N.m/A
        self.kv = 0.05  # V.s/rad
        self.V_alim = 24.0  # Volts

        # 2. Gains du Contrôleur
        # Kp pour x, y, z
        self.Kp = np.diag([150.0, 150.0, 150.0])
        self.Kd = 2 * np.sqrt(self.Kp)
        self.lmbda = 0.01  # Amortissement de la Jacobienne (Damped Least Squares)

    # -------------------- Lecture angles --------------------
    def read_angles(self):
        if self.last_packet:
            print("Angles lus:", self.last_packet[:4])
            return self.last_packet[:4]
        return None

    # -------------------- Lecture vitesses --------------------
    def read_speeds(self):
        if self.last_packet:
            return self.last_packet[4:]
        return None

    # -------------------- Lecture packets --------------------
    def update_packet(self):
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
    def send_speeds(self, v, grip):
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

    ####
    # -------------------- Contrôle en position --------------------
    ####

    def get_jacobian(self, q):
        """Jacobienne simplifiée (à remplacer par vos paramètres DH)."""
        # Exemple de dimension 3x4 pour un QArm
        return np.random.rand(3, 4)

    def inverse_dynamics(self, q, dq, ddq):
        """Calcule Tau = M*ddq + C*dq + G."""
        # Ici, insérez vos matrices M, C, G calculées précédemment
        M = np.eye(4) * 0.1
        G = np.array([0, 0.5, 0.2, 0.1])
        return M @ ddq + G  # Simplifié pour l'exemple

    def update(
        self,
        t: float,
        coeffs: NDArray[np.float64],
        q_mes: NDArray[np.float64],
        dq_mes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Boucle de calcul principale (Contrôle en espace opérationnel).
        Calcule le cycle complet : Trajectoire -> Cinématique -> Dynamique -> PWM.
        """

        # A. Consigne issue du polynôme (Desired state)
        pos_des, vel_des, accl_des = self.get_trajectory(t, coeffs)

        # B. État actuel via Modèle Géométrique et Cinématique (Measured state)
        pos_mes = self.forward_kinematics(q_mes)
        J = self.get_jacobian(q_mes)
        vel_mes = J @ dq_mes

        # C. Loi de commande cartésienne (Correction PD + Feedforward)
        # On calcule l'accélération de commande 'a_cmd' pour corriger l'erreur
        accl_cmd = accl_des + self.Kp @ (pos_des - pos_mes) + self.Kd @ (vel_des - vel_mes)

        # D. Inversion différentielle (Pseudo-inverse amortie)
        # On transforme l'accélération cartésienne en accélération articulaire
        # Formule : J_inv = J^T * (J*J^T + lambda^2*I)^-1
        J_inv = J.T @ np.linalg.inv(J @ J.T + self.lmbda**2 * np.eye(3))
        ddq_des = J_inv @ accl_cmd

        # E. Modèle Dynamique Inverse -> Calcul du couple (tau)
        # tau = M(q)@ddq + C(q,dq)@dq + G(q) + F(dq)
        tau = self.inverse_dynamics(q_mes, dq_mes, ddq_des)

        # F. Modèle Électrique Moteur -> Tension -> PWM
        # V = (R/kt)*tau + kv*dq (Compensation de la FEM et résistance)
        v_motor = (self.R / self.kt) * tau + self.kv * dq_mes

        # Conversion en pourcentage de la tension d'alimentation avec saturation
        pwm = np.clip((v_motor / self.V_alim) * 100, -100, 100)

        return pwm


# --- EXEMPLE D'UTILISATION ---
ctrl = QArmController()
x_init = np.array([0.2, 0.0, 0.1])
v_init = np.array([0.0, 0.0, 0.0])
a_init = np.array([0.0, 0.0, 0.0])
x_final = np.array([0.4, 0.1, 0.3])
v_final = np.array([0.0, 0.0, 0.0])
a_final = np.array([0.0, 0.0, 0.0])
c = ctrl.compute_quintic_coeffs(x_init, v_init, a_init, x_final, v_final, a_final, tf=5.0)

# Dans votre boucle temps réel :
# q, dq = robot.read_encoders()
# pwm = ctrl.update(current_time, c, q, dq)
# robot.send_pwm(pwm)
