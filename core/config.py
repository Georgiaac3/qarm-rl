from enum import Enum

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

    timestep: float = 0.002  # Période d'échantillonnage (secondes)
    simulation_duration: float = 15.0  # Durée totale de la simulation (secondes)

    # ========================================================================
    # PARAMÈTRES DE CAMÉRA
    # ========================================================================
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    # ========================================================================
    # TIMING DES PHASES DE MOUVEMENT
    # ========================================================================

    phase_arm_end: float = 2.0  # Fin de la phase d'armement (secondes)
    phase_accel_end: float = 3.5  # Fin de la phase d'accélération (secondes)
    phase_release_end: float = 3.6  # Fin de la phase de relâchement (secondes)
    phase_return_end: float = 5.0  # Fin du retour en position neutre (secondes)

    # ========================================================================
    # POSITIONS CIBLES ANGULAIRES
    # ========================================================================

    # Position d'armement (bras vers l'arrière) en radians
    shoulder_back: float = -0.4
    elbow_back: float = 0.3

    # Position de lancer (bras vers l'avant) en radians
    shoulder_forward: float = 0.6
    elbow_forward: float = -0.2

    # ========================================================================
    # AUTRES PARAMÈTRES
    # ========================================================================
    expected_data_size = 32  # 4 angles * 8 octets par double
    log_file: str = "robot_log.csv"  # Fichier de log des données

    # ========================================================================
    # SAC HYPERPARAMETERS
    # ========================================================================
    lr_pi: float = 0.0005  # Policy learning rate (Actor)
    lr_q: float = 0.001  # Q-network learning rate (Critic)
    init_alpha: float = 0.01  # Initial alpha value (temperature)
    gamma: float = 0.98  # Discount factor
    batch_size: int = 32  # Mini-batch size
    buffer_limit: int = 50000  # Maximum replay buffer size
    tau = 0.01  # Soft update coefficient for target networks
    target_entropy = -4.0  # Target entropy (-dim for 4D action space)
    lr_alpha = 0.001  # Learning rate for alpha


settings = Settings()
