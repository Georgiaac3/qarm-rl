import time

import numpy as np

from core.config import MODE, settings
from core.missions.stationary_mission import StationaryMission
from core.missions.trajectory_mission import MultiTrajectoryMission
from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim
from utils.types import Waypoint


def get_qarm_interface():
    """Factory pour obtenir l'interface du bras robotique selon le mode sélectionné."""
    if settings.mode == MODE.SIM:
        return QARMSim()
    return QARMReal()


def run_robot(robot: QARMReal):
    """ne sert que pour le QARMReal, Gazebo gère déjà la boucle pour QARMSim, on implémente donc le dt de 0.002 s ici"""

    # Connecting to the robot
    robot.connect()
    robot.init_stationnary()  # Donne une mission de stationnarité au robot en attendant les commandes de trajectoire

    # Creating missions
    ## Mission 1 : Se déplacer à une position donnée
    waypoint1 = Waypoint(
        position=np.array([0.3, 0.0, 0.2]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    )
    robot.missions.put(MultiTrajectoryMission(waypoint1))

    next_tick = time.perf_counter() + settings.timestep

    while True:
        robot.update_packet()
        angles_phi = robot.read_angles()
        speeds_dphi = robot.read_speeds()

        if angles_phi is not None and speeds_dphi is not None:
            angles_phi = np.array(angles_phi).reshape(4, 1)
            speeds_dphi = np.array(speeds_dphi).reshape(4, 1)

            robot.update(time.perf_counter(), angles_phi, speeds_dphi)

        while time.perf_counter() < next_tick:
            time.sleep(0.001)  # Sleep pour éviter de boucler trop vite
        next_tick += settings.timestep


if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)
