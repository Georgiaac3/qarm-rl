import mujoco

model = mujoco.MjModel.from_xml_path("genesis/QARM/urdf/qarm_with_gripper.urdf")

mujoco.mj_saveLastXML("genesis/QARM/mjcf/qarm_with_gripper.xml", model)
