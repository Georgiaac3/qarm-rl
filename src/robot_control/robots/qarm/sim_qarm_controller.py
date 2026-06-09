from genesis.engine.entities import RigidEntity
from robot_control.utils import CommandEnum, Matrix3x3, Vector5x1

from .base_qarm_controller import BaseQArmController


class SimQArmController(BaseQArmController):

    def __init__(
        self,
        timestep: float,
        Kp: Matrix3x3,
        Kd: Matrix3x3,
        Ki: Matrix3x3,
        qarm_entity: RigidEntity,
        dofs_idx: list[int],
    ):
        BaseQArmController.__init__(
            self, timestep, CommandEnum.TORQUES, Kp, Kd, Ki, display=False, display_data_queue=None
        )
        self.qarm_entity = qarm_entity
        self.dofs_idx = dofs_idx

    def _update_packet(self):
        """
        last_packet est de type list avec 8 doubles (4 pour les angles, 4 pour les vitesses).
        """
        # Un transfert direct via NumPy est souvent plus optimisé par le backend de Genesis
        pos_np = self.qarm_entity.get_dofs_position(self.dofs_idx[:4]).cpu().numpy()
        vel_np = self.qarm_entity.get_dofs_velocity(self.dofs_idx[:4]).cpu().numpy()
        self.last_packet = tuple(pos_np) + tuple(vel_np)

    def _send_command(self, cmd: Vector5x1):
        self.qarm_entity.control_dofs_force(cmd[:4].flatten(), self.dofs_idx[:4])

    def _close(self):
        pass

    def connect(self):
        pass
