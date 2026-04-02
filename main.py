import time
import os
import psutil

import numpy as np

from core.config import MODE, settings
from core.missions.stationary_mission import StationaryMission
from core.missions.trajectory_mission import MultiTrajectoryMission
from core.qarm.real import QARMReal
from core.qarm.sim import QARMSim
from utils.types import Waypoint

process = psutil.Process(os.getpid())

def get_qarm_interface():
    """Factory pour obtenir l'interface du bras robotique selon le mode sélectionné."""
    if settings.mode == MODE.SIM:
        return QARMSim()
    return QARMReal()


def run_robot(robot: QARMReal):
    """ne sert que pour le QARMReal, Gazebo gère déjà la boucle pour QARMSim, on implémente donc le dt de 0.002 s ici"""

    print("\n" \
    "################################\n"
    "#  Démarrage du robot en mode  #\n"
    f"#          {settings.mode}           #\n"
    "################################")

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

    print("\n" \
    "################################\n"
    "#  Boucle de contrôle du robot  #\n"
    "################################")
    #phi_arr = np.zeros((4, 1))
    #dphi_arr = np.zeros((4, 1))
    marge = 0.0002  # Marge de sécurité pour éviter les problèmes de timing
    n = 0
    b = 0
    while True:
        if n >= 1000:  # Affiche le taux de boucle toutes les 1000 itérations
            mem = process.memory_info().rss / 1024 / 1024  # RAM en Mo
            print(f"Fréquence erreur: {b/n:.6f} Hz - {b} erreurs sur {n} itérations - RAM utilisée : {mem:.2f} Mo")
            b = 0
            n = 0

        n += 1
        robot.update_packet()
        phi = robot.read_angles()
        dphi = robot.read_speeds()

        if phi is not None and dphi is not None:
            #phi_arr[:, 0] = phi
            #dphi_arr[:, 0] = dphi
            phi = np.array(phi).reshape(4, 1)
            dphi = np.array(dphi).reshape(4, 1)

            t_start_update = time.perf_counter()
            robot.update(time.perf_counter(), phi, dphi)
            t_end_update = time.perf_counter()
            t_update = t_end_update - t_start_update
            if t_update > settings.timestep:
                print(f"⚠️  Alerte timing : update a pris {t_update:.5f} s, dépassant la période de {settings.timestep:.5f} s")

        if time.perf_counter() >= next_tick:
            b += 1

        while time.perf_counter() < next_tick:
            pass
            #time.sleep(0.0001)
            #time.sleep(next_tick - time.perf_counter())  # Sleep pour éviter de boucler trop vite
        next_tick += settings.timestep


if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)
