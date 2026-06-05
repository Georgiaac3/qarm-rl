import socket
import struct
import time

from robot_control import PIDController
from robot_control.utils import robot_says
from robot_control.utils.types import CommandEnum

from .qarm_dynamics import QArmDynamics
from .qarm_kinematics import QArmKinematics


class QArmReal(PIDController):
    def __init__(
        self,
        dynamics: QArmDynamics,
        kinematics: QArmKinematics,
        command_type: CommandEnum,
        Kp,
        Kd,
        Ki,
    ):
        super().__init__(dynamics, kinematics, command_type, Kp, Kd, Ki)

        ##############################
        # UDP communication (Simulink)
        self.udp_ip: str = "127.0.0.1"
        self.udp_port_send: int = 5005
        self.udp_port_recv: int = 5006
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
                robot_says(f"Received data of unexpected size: {len(data)} bytes")
                self.last_packet = None
        except BlockingIOError:
            robot_says("No data received")
        except ConnectionResetError:
            robot_says("Connection reset by peer")

    def _send_command(self, cmd: list) -> None:
        try:
            message_bytes = struct.pack(
                "ddddd", cmd[0], cmd[1], cmd[2], cmd[3], cmd[4]
            )  # 4 vitesses + 1 commande de préhension
            self.sock.sendto(message_bytes, (self.udp_ip, self.udp_port_send))
            self.last_pwm = cmd  # Stockage de la dernière commande PWM envoyée pour l'affichage dans l'interface graphique
        except (OSError, struct.error) as e:
            robot_says(f"Erreur UDP envoi: {e}")

    def _close(self):
        self.sock.close()
        # TODO

    def connect(self):
        """
        Establishes the initial connection with the robot by sending zero velocity commands until valid angle data is received, indicating that communication is successful.
        """

        connexion = False
        while not connexion:
            self._send_command(
                [0.0, -0.1, -0.1, 0.0, 0.0]
            )  # Envoi de commandes de vitesse nulle pour initier la communication
            self._update_packet()
            angles = self._read_joint_angles()
            if angles is not None:
                robot_says("Connexion established, initial angles:" + str(angles))
                connexion = True
            else:
                robot_says("Waiting for connection...")
                time.sleep(1)
