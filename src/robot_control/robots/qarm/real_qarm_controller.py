import socket
import struct
import time

import numpy as np

from robot_control.utils import CommandEnum, Matrix3x3, Vector5x1, robot_says

from .base_qarm_controller import BaseQArmController


class RealQArmController(BaseQArmController):
    def __init__(
        self,
        timestep: float,
        Kp: Matrix3x3,
        Kd: Matrix3x3,
        Ki: Matrix3x3,
        display: bool = False,
        display_data_queue=None,
        upd_ip: str = "127.0.0.1",
        udp_port_send: int = 5005,
        udp_port_recv: int = 5006,
        command_type: CommandEnum = CommandEnum.PWM,
    ):
        super().__init__(timestep, command_type, Kp, Kd, Ki, display, display_data_queue)

        #######################################
        # UDP communication (Simulink for now)
        self.udp_ip: str = upd_ip
        self.udp_port_send: int = udp_port_send
        self.udp_port_recv: int = udp_port_recv
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", self.udp_port_recv))
        self.sock.setblocking(False)

    def _update_packet(self):
        """
        Lit les données UDP entrantes et met à jour le dernier packet reçu.
        Le packet attendu est de 64 bytes, contenant 8 doubles (4 pour les angles, 4 pour les vitesses).
        """

        last_received_data: bytes = b""
        packets_cleared = 0

        while True:
            try:
                # We actually receive (remove) the data here
                data, _ = self.sock.recvfrom(1024)
                last_received_data = data
                packets_cleared += 1
            except BlockingIOError:
                # The buffer is finally empty
                break

        try:
            # data, _ = self.sock.recvfrom(1024)
            data = last_received_data
            if len(data) == 64:
                self.last_packet = struct.unpack("8d", data)
            else:
                # robot_says(f"Received data of unexpected size: {len(data)} bytes")
                self.last_packet = None
        except BlockingIOError:
            robot_says("No data received")
        except ConnectionResetError:
            robot_says("Connection reset by peer")

    def _send_command(self, cmd: Vector5x1) -> None:
        try:
            message_bytes = struct.pack(
                "ddddd", cmd[0, 0], cmd[1, 0], cmd[2, 0], cmd[3, 0], cmd[4, 0]
            )  # Pack the command as 5 doubles (4 for motors, 1 for gripper)
            self.sock.sendto(message_bytes, (self.udp_ip, self.udp_port_send))
        except (OSError, struct.error) as e:
            robot_says(f"Erreur UDP envoi: {e}")

    def _close(self):
        self.sock.close()
        # TODO : for the camera

    def connect(self):
        """
        Establishes the initial connection with the robot by sending zero velocity commands until valid angle data is received, indicating that communication is successful.
        """

        connexion = False
        while not connexion:
            self._send_command(
                np.array([[0.0, -0.1, -0.1, 0.0, 0.0]]).T
            )  # Envoi de commandes de vitesse nulle pour initier la communication
            self._update_packet()
            angles = self.last_packet[:4] if self.last_packet is not None else None
            if angles is not None:
                robot_says("Connexion established, initial angles:" + str(angles))
                connexion = True
            else:
                robot_says("Waiting for connection...")
                time.sleep(1)
