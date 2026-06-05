import numpy as np

from src.core.controller.base_controller import PIDController
from src.core.dynamics.qarm import QArmDynamics
from src.core.kinematics.qarm import QArmKinematics
from src.utils.types import CommandEnum


class QArmReal(PIDController):
    def __init__(
        self,
        dynamics: QArmDynamics,
        kinematics: QArmKinematics,
        command_type: CommandEnum,
        Kp: NDArray[np.float64],
        Kd: NDArray[np.float64],
        Ki: NDArray[np.float64],
    ):
        super().__init__(dynamics, kinematics, command_type, Kp, Kd, Ki)
