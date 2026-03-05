from abc import ABC, abstractmethod


class QARMInterface(ABC):
    """Interface commune pour simulation ou robot réel."""

    @abstractmethod
    def send_speeds(self, v, grip):
        """Envoyer commandes vitesse + pince"""

    @abstractmethod
    def read_angles(self):
        """Lire les angles actuels du bras"""

    @abstractmethod
    def read_camera(self):
        """Lire la caméra du bras"""

    @abstractmethod
    def close(self):
        """Fermer les communications"""
