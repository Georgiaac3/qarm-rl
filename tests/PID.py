import time
from venv import logger
import numpy as np

from core.config import MODE, settings
from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim
from utils.logger import RobotLogger


def get_qarm_interface():
    """Factory pour obtenir l'interface du bras robotique selon le mode sélectionné."""
    if settings.mode == MODE.SIM:
        return QARMSim()
    return QARMReal()


def run_robot(robot: QARMReal):
    """ne sert que pour le QARMReal, Gazebo gère déjà la boucle pour QARMSim, on implémente donc le dt de 0.002 s ici"""

    robot.connect()

    logger = RobotLogger()
    start_time = time.time()

    #target_angles = np.array([-3*np.pi/4, 0.1, -np.pi/3, np.pi/2])
    target_angles = np.array([0, 0, 0, np.pi/2])

    while start_time + 1.5 > time.time(): # on arrête après 20 secondes pour ne pas faire tourner indéfiniment
        robot.update_packet()
        angles = robot.read_angles()
        speeds = robot.read_speeds()

        if angles is None or speeds is None:
            time.sleep(settings.timestep)
            continue

        current_angles = np.array(angles)
        current_speeds = np.array(speeds)

        logger.log(time.time() - start_time, current_angles, target_angles)

        order = robot.go_to_position_PID(target_angles, current_angles, current_speeds)

        robot.send_speeds([0., -0.1, -0.1, order[3]], 0)
        
        # Attendre 2 ms
        time.sleep(settings.timestep)
    
    robot.close()
    logger.plot()

if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)