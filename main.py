"""
Module principal de contrôle du bras robotique QARM via UDP.

Convention des axes (point de vue du robot):
    - Base: + = rotation vers la gauche
    - Épaule: + = mouvement vers le bas/avant
    - Coude: + = mouvement vers l'avant
    - Poignet: + = rotation sens horaire
    - Pince: 1 = fermée, 0 = ouverte
"""

import time
from typing import Tuple

from core.config import settings
from core.logger import logger
from utils.udp import JointAngles, setup_udp_socket, receive_joint_angles, send_control_commands


def calculate_joint_velocities(
    current_time: float, 
    angles: JointAngles
) -> Tuple[list[float], int]:
    """
    Calcule les vitesses articulaires en fonction du temps et des angles actuels.
    
    Cette fonction implémente une séquence de lancer en plusieurs phases :
    1. Armement : le bras recule
    2. Accélération : mouvement rapide vers l'avant
    3. Relâchement : ouverture de la pince
    4. Retour : retour en position neutre
    
    Args:
        current_time: Temps actuel dans la simulation (secondes)
        angles: Tuple contenant les angles actuels (base, shoulder, elbow, wrist)
        
    Returns:
        Tuple contenant:
            - Liste des 4 vitesses articulaires normalisées
            - État de la pince (1=fermée, 0=ouverte)
    """
    base, shoulder, elbow, wrist = angles
    
    # Détermination de la phase et des cibles
    if current_time < settings.phase_arm_end:
        # Phase 1 : Armement - bras vers l'arrière
        target_shoulder = settings.shoulder_back
        target_elbow = settings.elbow_back
        grip_state = 1  # Pince fermée
        
    elif current_time < settings.phase_accel_end:
        # Phase 2 : Accélération - interpolation linéaire vers l'avant
        phase_progress = (current_time - settings.phase_arm_end) / (
            settings.phase_accel_end - settings.phase_arm_end
        )
        target_shoulder = settings.shoulder_back + phase_progress * (
            settings.shoulder_forward - settings.shoulder_back
        )
        target_elbow = settings.elbow_back + phase_progress * (
            settings.elbow_forward - settings.elbow_back
        )
        grip_state = 1  # Pince fermée
        
    elif current_time < settings.phase_release_end:
        # Phase 3 : Relâchement rapide de l'objet
        target_shoulder = settings.shoulder_forward
        target_elbow = settings.elbow_forward
        grip_state = 0  # Pince ouverte
        
    elif current_time < settings.phase_return_end:
        # Phase 4 : Retour en position neutre
        target_shoulder = 0.0
        target_elbow = 0.0
        grip_state = 0  # Pince ouverte
        
    else:
        # Phase finale : maintien en position neutre
        target_shoulder = 0.0
        target_elbow = 0.0
        grip_state = 0
    
    # Contrôle proportionnel avec saturation
    v_shoulder = _saturate_velocity(settings.kp_gain * (target_shoulder - shoulder))
    v_elbow = _saturate_velocity(settings.kp_gain * (target_elbow - elbow))
    
    # Base et poignet restent fixes pour cette séquence
    v_base = 0.0
    v_wrist = 0.0
    
    return [v_base, v_shoulder, v_elbow, v_wrist], grip_state


def _saturate_velocity(velocity: float) -> float:
    """
    Limite une vitesse aux limites définies [-V_MAX, +V_MAX].
    """
    return max(min(velocity, settings.v_max), -settings.v_max)


# ============================================================================
# BOUCLE PRINCIPALE
# ============================================================================

def main() -> None:
    """
    Boucle d'orchestration
    """
    logger.info(
        f"Durée de simulation: {settings.simulation_duration}s, "
        f"Timestep: {settings.timestep}s"
    )
    
    # Initialisation
    sock = setup_udp_socket()
    current_time = 0.0
    
    try:
        while current_time < settings.simulation_duration:
            # Réception des angles actuels
            angles = receive_joint_angles(sock)
            
            # Calcul des commandes de contrôle
            velocities, grip_state = calculate_joint_velocities(current_time, angles)
            
            # Envoi des commandes
            send_control_commands(sock, velocities, grip_state)
            
            # Avancement temporel
            current_time += settings.timestep
            time.sleep(settings.timestep)
            
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        
    finally:
        sock.close()
        logger.info("Contrôleur arrêté, socket fermé")


if __name__ == "__main__":
    main()
