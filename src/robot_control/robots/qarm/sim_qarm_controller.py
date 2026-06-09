# from genesis.engine.entities import RigidEntity

from .base_qarm_controller import BaseQArmController


class SimQArmController(BaseQArmController):  # , RigidEntity):
    def __init__(self):
        pass

    def connect(self):
        print("Connecting to the simulated robot...")

    def init_stationnary(self):
        print("Initializing stationnary mission for the simulated robot...")

    def update_packet(self):
        pass

    def read_angles(self):
        return [0.0, 0.0, 0.0, 0.0]

    def read_speeds(self):
        return [0.0, 0.0, 0.0, 0.0]

    def update(self, t, phi, dphi):
        pass
