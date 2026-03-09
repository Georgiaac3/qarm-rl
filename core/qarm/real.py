import logging
import socket
import struct

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

    # -------------------- Lecture angles --------------------
    def read_angles(self):
        try:
            data, _ = self.sock.recvfrom(1024)
            if len(data) == 32:
                return struct.unpack("dddd", data)
            logging.warning("Received data of unexpected size: %d bytes", len(data))
        except BlockingIOError:
            print("No data received")
        except ConnectionResetError:
            print("Connection reset by peer")
        return "Default answer"

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
