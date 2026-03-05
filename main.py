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
    while True:
        print(robot.read_angles())
        robot.send_speeds([0.0, -0.1, -0.1, 0.0], 1)


if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)
