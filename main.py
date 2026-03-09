import time
from venv import logger
import numpy as np

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

    # Envoyer une commande initiale pour s'assurer que la communication est établiee
    connexion = False
    while not connexion:
        robot.send_speeds([0.0, -0.1, -0.1, 0.0], 0)
        angles = robot.read_angles()
        if angles is not None:
            print("Connexion établie, angles initiaux:", angles)
            connexion = True
        else:
            print("En attente de connexion...")
            time.sleep(0.5)

    positions_cibles = np.array([-3*np.pi/4, 0.1, -np.pi/3, np.pi/2])


    # --- Gains PID différents pour chaque moteur ---
    # TODO : fine-tune these parameters... with RL ? or with a simple grid search ? or just intuition ?
    Kp = np.array([1, 1, 1, 1])   # proportionnel
    Ki = np.array([0, 10, 0, 0])   # intégral
    Kd = np.array([0, 0, 0, 0])  # dérivé

    integrale_erreur = np.zeros(4)

    vitesse_max = 0.2
    while True:
        angles = robot.read_angles()
        print(angles)
        if angles is None:
            time.sleep(settings.dt)
            continue
        else:
            print("Angles reçus:", angles)
        positions_actuelles = np.array(angles[:4])
        vitesses_actuelles = np.zeros(4)

        # Calculer l'erreur
        erreur = positions_cibles - positions_actuelles

        # Mise à jour du terme intégral
        integrale_erreur += erreur * settings.dt

        # Calcul de la commande PID
        vitesses_commandes = Kp * erreur + Ki * integrale_erreur - Kd * vitesses_actuelles

        # Limiter les vitesses pour chaque moteur
        vitesses_commandes = np.clip(vitesses_commandes, -vitesse_max, vitesse_max)

        # Envoyer la commande au robot
        robot.send_speeds(vitesses_commandes.tolist(), 1)

        # Attendre 2 ms
        time.sleep(settings.dt)

if __name__ == "__main__":
    qarm = get_qarm_interface()
    run_robot(qarm)
