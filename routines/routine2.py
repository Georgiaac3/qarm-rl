import numpy as np

from robot_control.missions import MultiTrajectoryMission
from robot_control.robots.qarm import RealQArmController
from robot_control.utils import CommandEnum, Waypoint, robot_says_phase

# from routines.base_routine import Routine


class Routine2:  # (Routine):
    def __init__(self, display_data_queue=None, stop_event=None):
        self.qarm_controller = RealQArmController(
            timestep=0.005,
            Kp=200 * np.eye(3),
            Kd=90 * np.eye(3),
            Ki=0.0 * np.eye(3),
            display=True,
            display_data_queue=display_data_queue,
            command_type=CommandEnum.TORQUES,
        )

    def run(self):
        robot_says_phase("Routine 2 : Aller à un point, puis faire un lancé.")

        self.qarm_controller.connect()

        #################################
        # Creating the missions sequence

        # Mission 1 [Armement du bras]: Se déplacer à une position donnée
        waypoint_armement = Waypoint(
            position=np.array([0.4, 0.0, 0.2]).reshape(3, 1),
            velocity=None,
            acceleration=None,
        )
        mission_armement = MultiTrajectoryMission(waypoint_armement)
        mission_armement.set_ini_waypoint(
            Waypoint(
                position=np.array([0, 0.0, 0.40]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            )
        )
        mission_armement.compute_trajectory()
        self.qarm_controller.missions.append(mission_armement)
        # Mission 2 [Début du lancé]: Se déplacer à une position donnée
        waypoint_lance = Waypoint(
            position=np.array([0.15, 0, 0.70]).reshape(3, 1),
            velocity=np.array([-1.8, 0.0, 0.7]).reshape(3, 1),
            acceleration=None,
        )
        mission_lance = MultiTrajectoryMission(waypoint_lance)
        mission_lance.set_ini_waypoint(waypoint_armement)
        mission_lance.compute_trajectory()
        self.qarm_controller.missions.append(mission_lance)
        # Mission 3 [Amortir le lancé]: Se déplacer à une position donnée
        waypoint_amortissement = Waypoint(
            position=np.array([-0.40, 0.0, 0.75]).reshape(3, 1),
            velocity=None,
            acceleration=None,
        )
        mission_amortissement = MultiTrajectoryMission(waypoint_amortissement)
        mission_amortissement.set_ini_waypoint(waypoint_lance)
        mission_amortissement.compute_trajectory()
        self.qarm_controller.missions.append(mission_amortissement)

        ##############
        # Go spurs go
        self.qarm_controller.go()


if __name__ == "__main__":
    routine = Routine2()
    routine.run()
