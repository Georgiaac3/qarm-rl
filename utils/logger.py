import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="robot.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

class RobotLogger:
    def __init__(self):
        self.time_history = []
        self.angles_history = []  # Stockera des vecteurs de 4
        self.targets_history = [] # Stockera les cibles pour comparaison

    def log(self, t, angles, targets):
        self.time_history.append(t)
        self.angles_history.append(np.copy(angles))
        self.targets_history.append(np.copy(targets))

    def plot(self):
        # Conversion en tableaux numpy pour faciliter le slicing
        data = np.array(self.angles_history)
        targets = np.array(self.targets_history)
        t = np.array(self.time_history)
        
        # On crée 4 sous-graphiques (un par moteur)
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        names = ["Base", "Épaule", "Coude", "Pince"]
        
        for i in range(4):
            axs[i].plot(t, data[:, i], label=f"Angle {names[i]} réel", color='blue')
            axs[i].plot(t, targets[:, i], '--', label="Consigne", color='red', alpha=0.7)
            axs[i].set_ylabel("Angle (rad)")
            axs[i].legend(loc="upper right")
            axs[i].grid(True, alpha=0.3)
        
        axs[3].set_xlabel("Temps (s)")
        plt.tight_layout()
        plt.show()

# --- EXEMPLE D'INTÉGRATION DANS TA BOUCLE ---
# logger = RobotLogger()
# start_time = time.time()
# ... dans la boucle ...
# logger.log(time.time() - start_time, current_angles, target_angles)
# ... après la boucle ...
# logger.plot()