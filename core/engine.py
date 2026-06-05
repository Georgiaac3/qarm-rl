"""
Ce module contient la boucle de contrôle principale du robot, qui lit les données des capteurs, met à jour les missions en cours et envoie les commandes au robot à une fréquence définie.
"""

import os
import queue
import time
from typing import Union

import numpy as np
import psutil

from core.config import MODE, settings
from core.missions.circle_mission import CircleMission
from core.missions.square_mission import SquareMission
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


def run_robot(data_queue=None, stop_event=None):
    """
    Boucle de contrôle principale du robot. Lit les données des capteurs, met à jour les missions en cours et envoie les commandes au robot à une fréquence définie.
    Args:
        data_queue (multiprocessing.Queue, optional): Queue pour envoyer les données à une interface graphique ou à un logger. Defaults to None.
        stop_event (multiprocessing.Event, optional): Event pour signaler l'arrêt de la boucle de contrôle. Defaults to None.
    """

    print(
        "\n"
        "################################\n"
        "#  Démarrage du robot en mode  #\n"
        f"#          {settings.mode}           #\n"
        "################################"
    )

    # Initializing the robot interface
    robot: Union[QARMReal, QARMSim] = get_qarm_interface()

    # Connecting to the robot via udp
    robot.connect()
    #robot.init_stationnary()  # Donne une mission de stationnarité au robot en attendant les commandes de trajectoire

    point = Waypoint(position=np.array([0.6, -0.075, 0]).reshape(3,1), velocity=None, acceleration=None)
    mission_go_to = MultiTrajectoryMission(point)
    mission_go_to.set_ini_waypoint(Waypoint(position=np.array([0., 0., 0.6]).reshape(3,1), velocity=None, acceleration=None))
    mission_go_to.compute_trajectory()
    #robot.missions.append(mission_go_to)


    mission_circle = CircleMission(
        center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
        radius=0.2,
        plane=np.array([1.0, 0.0, 0.0]).reshape(3, 1),
        nb_of_circles=5,
        time_per_circle=10.0,
    )

    mission_square = SquareMission(
        center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
        side_length=0.4,
        plane=np.array([1.0, 0.0, 1.0]).reshape(3, 1),
        nb_of_squares=5,
        time_per_side=5.0,
    )

    robot.missions.append(mission_circle)
    # mission_square.set_ini_waypoint(mission_circle.waypoints[-1])  # Chaînage des missions pour une transition fluide
    mission_square.compute_trajectory()  # Calcul de la trajectoire pour la mission carré après avoir défini le waypoint initial
    robot.missions.append(mission_square)

    # Creating missions
    # Mission 1 [Armement du bras]: Se déplacer à une position donnée
    waypoint_armement = Waypoint(
        position=np.array([-0.1, 0.0, 0.5]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    )
    mission_armement = MultiTrajectoryMission(waypoint_armement)
    mission_armement.set_ini_waypoint(
        Waypoint(
            position=np.array([[4.87569632e-01], [3.86286014e-05], [3.81366771e-01]]),
            velocity=None,
            acceleration=None,
        )
    )
    mission_armement.compute_trajectory()
    robot.missions.append(mission_armement)
    # Mission 2 [Début du lancé]: Se déplacer à une position donnée
    waypoint_lance = Waypoint(
        position=np.array([0, 0, 0.8]).reshape(3, 1),
        velocity=np.array([0.3, 0.0, 0.3]).reshape(3, 1),
        acceleration=None,
    )
    mission_lance = MultiTrajectoryMission(waypoint_lance)
    mission_lance.set_ini_waypoint(waypoint_armement)
    mission_lance.compute_trajectory()
    robot.missions.append(mission_lance)
    # Mission 3 [Amortir le lancé]: Se déplacer à une position donnée
    waypoint_amortissement = Waypoint(
        position=np.array([0.7, 0.0, 0.3]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    )
    mission_amortissement = MultiTrajectoryMission(waypoint_amortissement)
    mission_amortissement.set_ini_waypoint(waypoint_lance)
    mission_amortissement.compute_trajectory()
    robot.missions.append(mission_amortissement)

    next_tick = time.perf_counter() + settings.timestep

    print(
        "\n"
        "################################\n"
        "#  Boucle de contrôle du robot  #\n"
        "################################"
    )
    # n = 0
    # b = 0
    while True:
        # if n >= 1000:  # Affiche le taux de boucle toutes les 1000 itérations
        #    mem = process.memory_info().rss / 1024 / 1024  # RAM en Mo
        #    print(
        #        f"Fréquence erreur: {b/n:.6f} Hz - {b} erreurs sur {n} itérations - RAM utilisée : {mem:.2f} Mo"
        #    )
        #    b = 0
        #    n = 0

        # n += 1

        # --- LOGIQUE DE CONTRÔLE DU ROBOT ---
        robot.update_packet()
        phi = robot.read_angles()
        dphi = robot.read_speeds()

        if phi is not None and dphi is not None:
            phi = np.array(phi).reshape(4, 1)
            dphi = np.array(dphi).reshape(4, 1)

            # t_start_update = time.perf_counter()
            robot.update(time.perf_counter(), phi, dphi)
            # t_end_update = time.perf_counter()
            # t_update = t_end_update - t_start_update
            # if t_update > settings.timestep:
            #    print(
            #        f"⚠️  Alerte timing : update a pris {t_update:.5f} s, dépassant la période de {settings.timestep:.5f} s"
            #    )

        # if time.perf_counter() >= next_tick:
        #    b += 1

        # --- ENVOI DES DONNÉES POUR VISUALISATION/LOGGING ---
        if data_queue is not None:
            if phi is not None and dphi is not None and robot.last_X_mes is not None:
                packet = {
                    "Angles Articulations mesurés (rad)": phi.ravel().tolist(),
                    "Vitesses mesurées (rad/s)": dphi.ravel().tolist(),
                    "PWM envoyés": robot.last_pwm,
                    "TCP_Trajectoire": robot.last_X_mes.ravel().tolist(),
                    "Wanted_TCP_Trajectoire": robot.last_X_des.ravel().tolist(),
                }

                try:
                    data_queue.put(packet, block=False)
                except queue.Full:
                    pass  # Queue pleine, on ignore pour rester en temps réel

        # --- SYNCHRONISATION TEMPORELLE ---
        while time.perf_counter() < next_tick:
            pass
            # time.sleep(0.0001)
            # time.sleep(next_tick - time.perf_counter())  # Sleep pour éviter de boucler trop vite
        next_tick += settings.timestep
