import numpy as np

from robot_control.missions import CircleMission, SquareMission
from robot_control.robots.qarm import RealQArmController
from robot_control.utils import CommandEnum, robot_says_phase

# from routines.base_routine import Routine


class Routine1:  # (Routine):
    def __init__(self, display_data_queue=None, stop_event=None):
        self.qarm_controller = RealQArmController(
            timestep=0.01,
            Kp=720.8212651581156 * np.diag([1, 1, 1]),
            Kd=161.97199510690137 * np.diag([1, 1, 1]),
            Ki=200.62867598716642 * np.diag([1, 1, 1]),
            display=True,
            display_data_queue=display_data_queue,
            command_type=CommandEnum.TORQUES,
        )

    def run(self):
        robot_says_phase("Routine 1 : Aller à un point, faire des cercles et des carrés")

        self.qarm_controller.connect()

        #################################
        # Creating the missions sequence

        # 5 squares #######

        mission_square = SquareMission(
            center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
            side_length=0.4,
            plane=np.array([1.0, 0.0, 1.0]).reshape(3, 1),
            nb_of_squares=50,
            time_per_side=5.0,
        )
        mission_square.compute_trajectory()
        self.qarm_controller.missions.append(mission_square)

        # 5 circles #######

        mission_circle = CircleMission(
            center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
            radius=0.2,
            plane=np.array([1.0, 0.0, 0.0]).reshape(3, 1),
            nb_of_circles=5,
            time_per_circle=10.0,
            timestep=self.qarm_controller.timestep,
        )
        self.qarm_controller.missions.append(mission_circle)

        ##############
        # Go spurs go
        self.qarm_controller.go()


if __name__ == "__main__":
    routine = Routine1()
    routine.run()
