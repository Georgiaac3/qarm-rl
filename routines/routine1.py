import numpy as np

from robot_control.missions import CircleMission, SquareMission
from robot_control.utils import robot_says_phase

from .base_routine import Routine


class Routine1(Routine):

    def __init__(self):
        self.qarm_controller = QArmControllerReal()

    def run(self):
        robot_says_phase("Routine 1 : Aller à un point, faire des cercles et des carrés")

        self.qarm_controller.connect()

        #################################
        # Creating the missions sequence

        # 5 circles #######

        mission_circle = CircleMission(
            center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
            radius=0.2,
            plane=np.array([1.0, 0.0, 0.0]).reshape(3, 1),
            nb_of_circles=5,
            time_per_circle=10.0,
        )
        self.qarm_controller.missions.append(mission_circle)

        # 5 squares #######

        mission_square = SquareMission(
            center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
            side_length=0.4,
            plane=np.array([1.0, 0.0, 1.0]).reshape(3, 1),
            nb_of_squares=5,
            time_per_side=5.0,
        )
        mission_square.compute_trajectory()
        self.qarm_controller.missions.append(mission_square)

        ##############
        # Go spurs go
        self.qarm_controller.go()
