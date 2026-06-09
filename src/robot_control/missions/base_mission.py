"""
Module pour définir la classe abstraite de mission pour le bras robotique QARM.
Les missions n'ont pas pour vocation de gérer la boucle de contrôle, mais simplement de fournir des points de consigne (position, vitesse, accélération) à suivre à chaque instant t.
"""

from abc import ABC, abstractmethod

from robot_control.utils.types import Waypoint


class Mission(ABC):
    """
    Classe abstraite représentant une mission pour le bras robotique QARM.
    """

    def __init__(self):
        name = self.__class__.__name__
        hashtags = "#" * len(name)
        print(
            "\n"
            f"###############################{hashtags}###\n"
            f"#  Nouvelle mission en créée : {name}  #\n"
            f"###############################{hashtags}###"
        )

        self.start_time = (
            None  # Temps de début de la mission, à initialiser lors de l'empilement de la mission
        )
        self.finished = False  # Indique si la mission est terminée

        self.load = 0.0  # Charge utile associée à la mission, peut être utilisée pour ajuster les gains du contrôleur en fonction de la charge transportée

    def stop(self):
        """Permet de terminer la mission et de passer à la mission suivante."""
        self.finished = True

    def is_finished(self):
        """Indique si la mission est terminée."""
        return self.finished

    @abstractmethod
    def finish_condition(self, t: float, waypoint_mes: Waypoint) -> bool:
        """Condition de fin de la mission, à implémenter selon les besoins spécifiques de chaque mission."""

    @abstractmethod
    def set_ini_waypoint(self, waypoint: Waypoint):
        """Permet de définir le waypoint initial de la mission, nécessaire pour toutes les missions, à part celle de ne rien faire."""

    @abstractmethod
    def get_waypoint_at_t(self, t: float) -> Waypoint:
        """Retourne le point de consigne (position, vitesse, accélération) à l'instant t ou DoNothing."""

    def say_hello(self):
        """Message de bienvenue spécifique à la mission, peut être utilisé pour indiquer le début d'une nouvelle phase de la tâche."""
        print(
            "\n"
            f"###############################{'#' * len(self.__class__.__name__)}\n"
            f"#  Mission {self.__class__.__name__} : Starting now !  #\n"
            f"###############################{'#' * len(self.__class__.__name__)}"
        )

    def say_goodbye(self):
        """Message d'au revoir spécifique à la mission, peut être utilisé pour indiquer la fin d'une phase de la tâche."""
        print(
            "\n"
            f"###############################{'#' * len(self.__class__.__name__)}###\n"
            f"#  Mission {self.__class__.__name__} : Finished ! Moving to the next one...  #\n"
            f"###############################{'#' * len(self.__class__.__name__)}###"
        )
