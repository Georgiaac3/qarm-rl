import time
from venv import logger
import numpy as np

from core.config import MODE, settings
from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim


def get_qarm_interface():
    """Factory pour obtenir l'interface du bras robotique selon le mode sélectionné."""
    if settings.mode == MODE.SIM:
        return QARMSim()
    return QARMReal()


def run_robot(robot: QARMReal):
    """ne sert que pour le QARMReal, Gazebo gère déjà la boucle pour QARMSim, on implémente donc le dt de 0.002 s ici"""

    # Envoyer une commande initiale pour s'assurer que la communication est établiee
    connexion = False
    while not connexion:
        robot.send_speeds([0.0, -0.1, -0.1, 0.0], 0)
        robot.update_packet()
        angles = robot.read_angles()
        if angles is not None:
            print("Connexion établie, angles initiaux:", angles)
            connexion = True
        else:
            print("En attente de connexion...")
            time.sleep(0.5)

    #target_angles = np.array([-3*np.pi/4, 0.1, -np.pi/3, np.pi/2])
    target_angles = np.array([0, 0, -np.pi/2, np.pi/3])

    while True:
        robot.update_packet()
        angles = robot.read_angles()
        speeds = robot.read_speeds()

        if angles is None or speeds is None:
            time.sleep(settings.dt)
            continue

        current_angles = np.array(angles)
        current_speeds = np.array(speeds)

        robot.go_to_position_PID(target_angles, current_angles, current_speeds)

        # Attendre 2 ms
        time.sleep(settings.timestep)

if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)