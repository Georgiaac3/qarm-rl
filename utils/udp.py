import socket
import struct
from typing import Tuple

from core.config import settings
from core.logger import logger


JointAngles = Tuple[float, float, float, float]  # (base, shoulder, elbow, wrist)


def setup_udp_socket() -> socket.socket:
    """
    Configure et initialise le socket UDP pour la communication avec Simulink.
    
    Returns:
        Socket UDP configuré en mode non-bloquant
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((settings.local_bind_ip, settings.udp_port))
    sock.setblocking(False)
    return sock


def receive_joint_angles(sock: socket.socket) -> JointAngles:
    """
    Reçoit les angles articulaires depuis Simulink via UDP.
    
    Args:
        sock: Socket UDP configuré
        
    Returns:
        Tuple contenant les 4 angles articulaires (base, shoulder, elbow, wrist).
        Retourne (0, 0, 0, 0) si aucune donnée n'est reçue.
    """
    try:
        data, addr = sock.recvfrom(1024)
        if len(data) == settings.expected_data_size:
            angles = struct.unpack('dddd', data)
            logger.debug(f"Angles reçus: {angles}")
            return angles
        else:
            logger.warning(
                f"Taille de données incorrecte: {len(data)} octets "
                f"(attendu: {settings.expected_data_size})"
            )
            return (0.0, 0.0, 0.0, 0.0)
            
    except BlockingIOError:
        # Pas de données disponibles
        return (0.0, 0.0, 0.0, 0.0)
        
    except ConnectionResetError:
        logger.warning("Connexion UDP réinitialisée par l'hôte distant")
        return (0.0, 0.0, 0.0, 0.0)


def send_control_commands(
    sock: socket.socket, 
    velocities: list[float], 
    grip_state: int
) -> None:
    """
    Envoie les commandes de vitesse et l'état de la pince vers Simulink.
    
    Args:
        sock: Socket UDP configuré
        velocities: Liste des 4 vitesses articulaires [v_base, v_shoulder, v_elbow, v_wrist]
        grip_state: État de la pince (1=fermée, 0=ouverte)
    """
    try:
        message_bytes = struct.pack(
            'ddddd',
            float(velocities[0]),
            float(velocities[1]),
            float(velocities[2]),
            float(velocities[3]),
            float(grip_state)
        )
        sock.sendto(message_bytes, (settings.udp_ip, settings.udp_port))
        logger.debug(f"Commandes envoyées: v={velocities}, grip={grip_state}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi UDP: {e}")
