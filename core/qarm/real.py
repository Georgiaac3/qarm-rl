import logging
import socket
import struct
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from core.config import settings
from core.qarm.interface import QARMInterface


class QARMReal(QARMInterface):
    """Implémentation pour le bras robotique réel avec communication UDP et caméra RealSense."""

    def __init__(self):
        # Connexion UDP
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", settings.udp_port_recv))
        self.sock.setblocking(False)

        self.last_packet = None #(0.0,) * 8 -> 4 first coordonnates for the angles, 4 last coordonnates for speeds

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
        self.last_order_time = None
        self.last_real_speeds = np.zeros(4)
        self.error_integral = 0.

        self.limit_speeds_pourcentage = 0.7 # out of 1 for PWM

        # TODO : fine-tune these parameters... with RL ? or with a simple grid search ? or just intuition ?
        # les gains contiennent l'inverse du coefficient directeur entre pwm et speed
        # valeurs de gain de Maria : Kp = 12.7, Kd = 204.6 Ki = 0.012,
        self.Kp = np.array([1, 1, 1, 1])*30/100   # proportionnel
        self.Kd = np.array([1, 1, 1, 1])*40/100   # dérivé
        self.Ki = np.array([1, 1, 1, 1])*0/1000   # intégral
        self.Kcomp = 0.


    # -------------------- Lecture angles --------------------
    def read_angles(self):
        if self.last_packet:
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

    # -------------------- Fermeture --------------------
    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.sock.close()

    def go_to_position_PID(self, target_angles, current_angles, current_speeds):
        """
        the following must be np.array
        target_angles: [angle_base, angle_shoulder, angle_elbow, angle_wrist]
        current_angles:  [angle_base, angle_shoulder, angle_elbow, angle_wrist]
        current_speeds: [speed_base, speed_shoulder, speed_elbow, speed_wrist]
        """
        now = time.time()
        if self.last_order_time is None:
            self.last_order_time = now
            return np.zeros_like(current_angles)
       
        dt = now - self.last_order_time # custom delta time, not necessarily corresponding to the timestep
        self.last_order_time = now

        # 1. Erreur de position
        erreur = target_angles - current_angles
       
        # 2. Terme P (Proportionnel)
        P = self.Kp * erreur
       
        # 3. Terme I (Intégral)
        self.error_integral += erreur * dt
        I = self.Ki * self.error_integral
       
        # 4. Terme D (Dérivé)
        D = -self.Kd * current_speeds
       
        # 5. Compensation dynamique (Couplage)
        # On regarde comment la vitesse de l'épaule change (accélération)
        acceleration = (current_speeds - self.last_real_speeds) / dt
        geometrical_factor = -np.sin(current_angles[2])
        # On applique un gain de compensation croisé :
        # l'accélération de l'épaule [1] influence la commande du coude [2]
        compensation = np.zeros_like(P)
        compensation[2] = self.Kcomp * geometrical_factor * acceleration[1]
       
        # 6. Somme et Normalisation
        vitesse_brute = P + I + D + compensation
        order = np.clip(vitesse_brute, -self.limit_speeds_pourcentage, self.limit_speeds_pourcentage)
       
        # Sauvegarde pour le prochain cycle
        self.last_real_speeds = np.copy(current_speeds)
       
        self.send_speeds(order.tolist(), 1)
