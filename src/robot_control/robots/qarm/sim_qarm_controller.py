import torch

from genesis.engine.entities import RigidEntity
from robot_control.utils import CommandEnum, Matrix3x3, Vector5x1, Waypoint

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
        gs,
        metrics: bool = False,
    ):
        BaseQArmController.__init__(
            self,
            timestep,
            CommandEnum.TORQUES,
            Kp,
            Kd,
            Ki,
            display=False,
            display_data_queue=None,
            metrics=metrics,
        )
        self.qarm_entity = qarm_entity
        self.dofs_idx = dofs_idx
        self.gs = gs

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

    def set_pos(self, waypoint: Waypoint):
        """
        Permet de définir l'état du robot à partir d'un waypoint.
        """
        local_offset = torch.tensor(
            [0.0, 0.0, self.L5], dtype=self.gs.tc_float, device=self.gs.device
        )  # 10cm along Z-axis
        angles = self.qarm_entity.inverse_kinematics(
            link=self.qarm_entity.get_link("END_EFFECTOR_BASE"),
            pos=waypoint.position.flatten(),
            quat=None,
            local_point=local_offset,
        )

        if not torch.is_tensor(angles):
            raise ValueError("Inverse kinematics did not return a valid tensor for joint angles.")
        angles[3] = 0.0
        self.qarm_entity.set_dofs_position(angles, self.dofs_idx[:4])

        zeros = [0.0] * 4
        self.qarm_entity.set_dofs_velocity(zeros, self.dofs_idx[:4])
