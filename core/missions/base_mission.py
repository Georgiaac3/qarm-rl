"""
Module pour définir la classe abstraite de mission pour le bras robotique QARM.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from numpy.typing import NDArray


class Mission(ABC):
    """
    Classe abstraite représentant une mission pour le bras robotique QARM.
    """

    @abstractmethod
    def get_target(
        self, t: float, q_mes: NDArray, p_mes: NDArray
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Retourne (pos_des, vel_des, accl_des)."""

    @abstractmethod
    def is_finished(self, t: float, p_mes: NDArray) -> bool:
        """Condition d'arrêt de la mission."""
