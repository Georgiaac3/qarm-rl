import numpy as np

from robot_control.missions import MultiTrajectoryMission
from robot_control.robots.qarm import RealQArmController
from robot_control.utils import CommandEnum, Waypoint, robot_says_phase

# from routines.base_routine import Routine


class Routine3:  # (Routine):
    def __init__(self, display_data_queue=None, stop_event=None):
        self.qarm_controller = RealQArmController(
            timestep=0.01,
            Kp=478.9267873576293 * np.diag([1, 1, 1]),
            Kd=124.81491235394921 * np.diag([1, 1, 1]),
            Ki=124.79561626896212 * np.diag([1, 1, 1]),
            display=True,
            display_data_queue=display_data_queue,
            command_type=CommandEnum.TORQUES,
        )

    def run(self):
        robot_says_phase("Routine 3 : Suivre une trajectoire complexe")

        self.qarm_controller.connect()

        #################################
        # Creating the missions sequence

        ini_waypoint = Waypoint(
            position=np.array([0.2, 0.0, 0.5]).reshape(3, 1),
            velocity=None,
            acceleration=None,
        )

        waypoints = [
            Waypoint(
                position=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([0.3, 0.3, 0.5]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([0.3, 0.3, 0.1]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([-0.3, -0.3, 0.6]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([0, -0.5, 0.1]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([-0.5, -0.2, 0.1]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([-0.3, 0.3, 0.4]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
            Waypoint(
                position=np.array([0.4, 0.0, 0.2]).reshape(3, 1),
                velocity=None,
                acceleration=None,
            ),
        ]

        multi_trajectory_mission = MultiTrajectoryMission(
            waypoints=waypoints,
            ini_waypoint=ini_waypoint,
        )

        self.qarm_controller.add_mission(multi_trajectory_mission)

        ##############
        # Go spurs go
        self.qarm_controller.go()


if __name__ == "__main__":
    routine = Routine3()
    routine.run()
