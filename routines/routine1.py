from routines.base_routine import Routine
from src.utils.logger import robot_says_phase


class Routine1(Routine):
    def run(self):
        robot_says_phase("Routine 1 : Aller à un point, faire des cercles et des carrés")

        robot_controller = QArmController()
