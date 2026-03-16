# from rclpy.node import Node
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray

# from core.qarm.interface import QARMInterface

class QARMSim:
    """Interface QARM pour ROS / Gazebo"""

    def __init__(self):
        # Simulateur de bras robotique
        self.angles = [0.0, 0.0, 0.0, 0.0]

    def send_speeds(self, v, grip):
        """Simuler l'envoi des vitesses au robot simulé"""
        print(f"Simulated speeds: {v} with grip: {grip}")

    def read_angles(self):
        """Retourne les angles simulés"""
        return self.angles

    def read_camera(self):
        """Optionnel : Simuler la lecture d'une caméra"""
        return None

# class QARMSim(QARMInterface, Node):
#     """Interface QARM pour ROS / Gazebo"""

#     def __init__(self, node_name="qarm_node"):
#         super().__init__(node_name)

#         # Publisher pour envoyer les vitesses au robot simulé
#         self.pub_speeds = self.create_publisher(Float64MultiArray, "/qarm/joint_vel_cmd", 10)

#         # Subscriber pour recevoir les positions actuelles
#         self.angles = [0.0, 0.0, 0.0, 0.0]
#         self.create_subscription(JointState, "/qarm/joint_states", self._angles_callback, 10)

#     def _angles_callback(self, msg: JointState):
#         self.angles = list(msg.position[:4])  # On prend juste les 4 joints principaux

#     def send_speeds(self, v, grip):
#         """Publier les vitesses sur le topic ROS"""
#         msg = Float64MultiArray()
#         msg.data = v + [grip]  # vitesse + pince
#         self.pub_speeds.publish(msg)

#     def read_angles(self):
#         """Retourne les dernières positions reçues par le subscriber"""
#         return self.angles

#     def read_camera(self):
#         """Optionnel : ROS CameraPublisher / subscriber"""
#         # Pour RealSense ou camera Gazebo, on pourrait utiliser cv_bridge
#         return None
