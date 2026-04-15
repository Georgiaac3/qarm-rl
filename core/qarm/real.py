import logging
import socket
import struct
import time

import cv2
import numpy as np

from core.config import settings
from core.qarm.interface import QARMInterface
from utils.realsense_camera import RealsenseCamera


class QARMReal(QARMInterface):
    """Real QARM robot arm implementation with UDP communication and RealSense camera integration."""

    def __init__(self):
        # UDP connection
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", settings.udp_port_recv))
        self.sock.setblocking(False)

        self.last_packet = None  # 8 doubles: first 4 for angles, last 4 for speeds

        # Initialize RealSense camera with YOLO object detection
        self.camera = RealsenseCamera(
            model_path=settings.yolo_model,
            confidence_threshold=0.85,
            camera_width=settings.camera_width,
            camera_height=settings.camera_height,
            fps=settings.camera_fps,
        )

        # PID position controller state
        self.filtered_speeds = None
        self.last_filtered_accel = None
        self.last_order_time = None
        self.last_real_speeds = None
        self.error_integral = np.zeros(4)

        # Speed limit (fraction of max PWM)
        self.limit_speeds_pourcentage = 0.7

        # TODO: Fine-tune PID gains with RL, grid search, or manual calibration
        # PID gain matrices (proportional, derivative, integral)
        self.Kp = np.array([0, 0, 0, 0.6*10/100])
        self.Kd = np.array([0, 0, 0, 0.6*10.37/8])
        self.Ki = np.array([0, 0, 0, 2*0.6*10/100/10.37])
        self.Kcomp = 0.


    # ============ State Reading ============
    def read_angles(self) -> np.ndarray:
        """Read current joint angles from last received packet.
        
        Returns:
            Array of 4 joint angles or None if no packet received
        """
        if self.last_packet:
            return self.last_packet[:4]
        return None

    def read_speeds(self) -> np.ndarray:
        """Read current joint speeds from last received packet.
        
        Returns:
            Array of 4 joint speeds or None if no packet received
        """
        if self.last_packet:
            return self.last_packet[4:]
        return None

    def update_packet(self) -> None:
        """Receive and parse UDP packet from robot controller."""
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
            logging.warning("Connection reset by peer")
            
    # ============ Command Sending ============
    def send_speeds(self, v: np.ndarray, grip: float) -> None:
        """Send joint speed commands and gripper state via UDP.
        
        Args:
            v: Array of 4 joint speed commands
            grip: Gripper state (0-1)
        """
        try:
            message_bytes = struct.pack("ddddd", v[0], v[1], v[2], v[3], grip)
            self.sock.sendto(message_bytes, (settings.udp_ip, settings.udp_port_send))
        except (OSError, struct.error) as e:
            logging.error("UDP send error: %s", e)

    # ============ Camera Reading ============
    def read_camera(self):
        """
        Retrieve color and depth frames from RealSense camera.

        Returns:
            Tuple of (color_image, depth_image) or (None, None) if unavailable
        """
        frames = self.camera.get_frames()
        if frames is None:
            return None, None
        color_frame, depth_frame = frames
        # color_frame is already a numpy array from the threaded camera
        color_image = color_frame if isinstance(color_frame, np.ndarray) else np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data()) if hasattr(depth_frame, 'get_data') else depth_frame
        return color_image, depth_image

    def detect_object(self):
        """
        Detect object in camera frame using YOLO.

        Returns:
            Tuple of (3D coordinates, confidence, pixel coordinates) or None
        """
        frames = self.camera.get_frames()
        if frames is None:
            return None
        color_frame, depth_frame = frames
        detection = self.camera.detect_object(color_frame, depth_frame)
        return detection
    # ============ Connection ============
    def connect(self) -> None:
        """Establish connection with robot controller by waiting for first angle packet and start camera thread."""
        connected = False
        while not connected:
            self.send_speeds([0.0, -0.1, -0.1, 0.0], 0)
            self.update_packet()
            angles = self.read_angles()
            if angles is not None:
                logging.info(f"Connection established. Initial angles: {angles}")
                # Start camera thread automatically
                self.camera.start()
                logging.info("Camera thread started")
                connected = True
            else:
                logging.info("Waiting for connection...")
                time.sleep(1)

    # ============ Cleanup ============
    def close(self):
        """Properly shutdown camera, sockets and other resources."""
        self.camera.cleanup()
        cv2.destroyAllWindows()
        self.sock.close()

    # -------------------- Contrôle en position --------------------
    # def go_to_position_PID(self, target_angles, current_angles, current_speeds):
    #     """
    #     the following must be np.array
    #     target_angles: [angle_base, angle_shoulder, angle_elbow, angle_wrist]
    #     current_angles:  [angle_base, angle_shoulder, angle_elbow, angle_wrist]
    #     current_speeds: [speed_base, speed_shoulder, speed_elbow, speed_wrist]
    #     """
    #     now = time.time()
    #     if self.last_order_time is None:
    #         self.last_order_time = now
    #         return np.zeros_like(current_angles)
       
    #     dt = now - self.last_order_time # custom delta time, not necessarily corresponding to the timestep
    #     self.last_order_time = now

    #     # 1. Erreur de position
    #     erreur = target_angles - current_angles
       
    #     # 2. Terme P (Proportionnel)
    #     P = self.Kp * erreur
       
    #     # 3. Terme I (Intégral)
    #     self.error_integral += erreur * dt
    #     I = self.Ki * self.error_integral
       
    #     # 4. Terme D (Dérivé)
    #     D = -self.Kd * current_speeds
       
    #     # 5. Compensation dynamique (Couplage)
    #     # On regarde comment la vitesse de l'épaule change (accélération)
    #     acceleration = (current_speeds - self.last_real_speeds) / dt
    #     geometrical_factor = -np.sin(current_angles[2])
    #     # On applique un gain de compensation croisé :
    #     # l'accélération de l'épaule [1] influence la commande du coude [2]
    #     compensation = np.zeros_like(P)
    #     compensation[2] = self.Kcomp * geometrical_factor * acceleration[1]
       
    #     # 6. Somme et Normalisation
    #     vitesse_brute = P + I + D + compensation
    #     order = np.clip(vitesse_brute, -self.limit_speeds_pourcentage, self.limit_speeds_pourcentage)
       
    #     # Sauvegarde pour le prochain cycle
    #     self.last_real_speeds = np.copy(current_speeds)

    #     return order

    def go_to_position_PID(self, target_angles, current_angles, current_speeds):
        now = time.time()
        if self.last_order_time is None:
            self.last_order_time = now
            self.filtered_speeds = np.copy(current_speeds)
            self.last_real_speeds = np.copy(current_speeds)
            self.last_filtered_accel = np.zeros_like(current_speeds)
            return np.zeros_like(current_angles)

        dt = now - self.last_order_time
        self.last_order_time = now

        # Low-pass filter on speeds (alpha=0.2: responsive but slightly noisy)
        alpha_v = 0.2
        self.filtered_speeds = (alpha_v * current_speeds) + (1 - alpha_v) * self.filtered_speeds

        # 1. Position error
        position_error = target_angles - current_angles

        # 2. Proportional term
        P = self.Kp * position_error

        # 3. Integral term (currently mostly zeroed in tuning)
        self.error_integral += position_error * dt
        I = self.Ki * self.error_integral

        # 4. Derivative term using filtered speeds
        D = -self.Kd * self.filtered_speeds

        # 5. Cross-coupling compensation for geometric coupling
        # Compute filtered acceleration from speed changes
        raw_acceleration = (self.filtered_speeds - self.last_real_speeds) / dt
        alpha_a = 0.1  # Stronger filter for acceleration
        filtered_accel = (alpha_a * raw_acceleration) + (1 - alpha_a) * self.last_filtered_accel
        self.last_filtered_accel = np.copy(filtered_accel)

        # Geometric factor: negative sine of elbow angle
        # (Represents coupling between shoulder and elbow)
        geometrical_factor = -np.sin(current_angles[2])
        
        compensation = np.zeros_like(P)
        compensation[2] = self.Kcomp * geometrical_factor * filtered_accel[1]

        # 6. Sum all terms and saturate to speed limit
        raw_command = P + I + D + compensation
        command = np.clip(raw_command, -self.limit_speeds_pourcentage, self.limit_speeds_pourcentage)

        self.last_real_speeds = np.copy(self.filtered_speeds)

        return command
