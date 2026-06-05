"""
Ce module contient la classe de configuration de l'application, qui utilise Pydantic pour définir les paramètres de configuration et les valeurs par défaut. Il inclut également une énumération pour sélectionner le mode de fonctionnement du bras robotique (réel ou simulation).
"""

from enum import Enum
from typing import Dict, List

from pydantic_settings import BaseSettings


class MODE(Enum):
    """Enum pour sélectionner le mode de fonctionnement du bras robotique (réel ou simulation)."""

    REAL = "real"
    SIM = "sim"


class Settings(BaseSettings):
    """Configuration de l'application de contrôle du bras robotique QARM."""

    mode: MODE = MODE.REAL

    # ========================================================================
    # CONFIGURATION RÉSEAU UDP
    # ========================================================================

    # Adresse et port de destination (Simulink)
    udp_ip: str = "127.0.0.1"
    udp_port_send: int = 5005
    udp_port_recv: int = 5006

    # Adresse locale (Python)
    local_bind_ip: str = "0.0.0.0"

    # ========================================================================
    # PARAMÈTRES TEMPORELS
    # ========================================================================

    timestep: float = (
        0.005  # Période d'échantillonnage (secondes) (moitié plus cours que ce qu'on recoit en simulation (0.01s), TODO : vérifier pour simulink dans QUARC > Model Settings > Solver details > Fixed-step size)
    )
    simulation_duration: float = 15.0  # Durée totale de la simulation (secondes)

    # ========================================================================
    # PARAMÈTRES DE CAMÉRA
    # ========================================================================
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ========================================================================
    # VARIABLES À AFFICHER
    # ========================================================================
    graphs_2d: Dict[str, List[str]] = {
        "Angles Articulations mesurés (rad)": [
            "Joint_1_mes",
            "Joint_2_mes",
            "Joint_3_mes",
            "Joint_4_mes",
        ],
        # "Vitesses mesurées (rad/s)": ["Speed_1_mes", "Speed_2_mes", "Speed_3_mes", "Speed_4_mes"],
        "PWM envoyés": ["PWM_1_cmd", "PWM_2_cmd", "PWM_3_cmd", "PWM_4_cmd", "Grip_cmd"],
    }
    graphs_3d: List[str] = ["TCP_Trajectoire", "Wanted_TCP_Trajectoire"]

    # ========================================================================
    # AUTRES PARAMÈTRES
    # ========================================================================
    expected_data_size: int = 40  # 5 valeurs * 8 octets par double
    log_file: str = "robot_log.csv"  # Fichier de log des données


settings = Settings()
