from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration de l'application de contrôle du bras robotique QARM."""

    # ========================================================================
    # CONFIGURATION RÉSEAU UDP
    # ========================================================================

    # Adresse et port de destination (Simulink)
    udp_ip: str = "127.0.0.1"
    udp_port: int = 5005

    # Adresse locale (Python)
    local_bind_ip: str = "0.0.0.0"

    # ========================================================================
    # PARAMÈTRES TEMPORELS
    # ========================================================================

    timestep: float = 0.002  # Période d'échantillonnage (secondes)
    simulation_duration: float = 15.0  # Durée totale de la simulation (secondes)

    # ========================================================================
    # PARAMÈTRES DE CONTRÔLE
    # ========================================================================

    v_max: float = 0.5  # Vitesse maximale normalisée (0 à 1)
    kp_gain: float = 2.0  # Gain du contrôleur proportionnel

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


settings = Settings()
