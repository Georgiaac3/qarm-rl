"""
This is an example code for using genesis-world directly with the SimQArmController. It's the base for RL algorithms.
"""

import os
import time

import numpy as np

import genesis as gs
from robot_control.missions import CircleMission, MultiTrajectoryMission, SquareMission
from robot_control.robots.qarm import SimQArmController
from robot_control.utils import Waypoint

# import gs_nyx.nyx_py_renderer as npr
# from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions


# ------------------------------ init Genesis ------------------------------
gs.init(
    backend=gs.gpu,
    precision="32",
    seed=None,
    debug=False,
    performance_mode=False,  # set to True when training (to gain 30% of performance)
    logging_level="warning",
    theme="light",
    logger_verbose_time=False,
)

# ------------------------------ create scene ------------------------------
time_per_step = 0.01  # 100hz
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        gravity=(0, 0, -9.80665),
        dt=time_per_step,
        substeps=1,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,  # visualize the coordinate frame of `world` at its origin
        world_frame_size=1.0,  # length of the world frame in meter
        show_link_frame=False,  # visualizing the coordinate frames of entity links
        show_cameras=False,  # do not visualize mesh and frustum of the cameras added
        plane_reflection=False,  # turn off plane reflection
        ambient_light=(0.1, 0.1, 0.1),  # ambient light setting
    ),
    # viewer_options=gs.options.ViewerOptions(
    #    res=(1280, 960),  # (2560, 1920),
    #    camera_pos=(-1.5, 1.5, 1.5),
    #    camera_lookat=(0.0, 0.0, 0.5),
    #    camera_fov=40,
    #    max_FPS=103,
    # ),
    # renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
    show_viewer=True,
)
# ------------------------------- add entities ------------------------------
plane = scene.add_entity(gs.morphs.Plane())
qarm_entity = scene.add_entity(
    # gs.morphs.MJCF(file="genesis/QARM/mjcf/qarm_with_gripper.xml"),
    gs.morphs.URDF(file="genesis/QARM/urdf/qarm_gripper_com.urdf", fixed=True),
    # gs.morphs.URDF(file="genesis/QARM/urdf/QARM.urdf", fixed=True),
)

box0 = scene.add_entity(gs.morphs.Box(pos=(1, 0, 0.0), size=(0.1, 0.1, 0.1), fixed=True))

box = scene.add_entity(gs.morphs.Box(pos=(0.3, 0, 0.0), size=(0.01, 0.01, 0.01), fixed=True))
box2 = scene.add_entity(gs.morphs.Box(pos=(0.3, 0.3, 0.5), size=(0.01, 0.01, 0.01), fixed=True))
box3 = scene.add_entity(gs.morphs.Box(pos=(0.3, 0.3, 0.1), size=(0.01, 0.01, 0.01), fixed=True))
box4 = scene.add_entity(gs.morphs.Box(pos=(-0.3, -0.3, 0.6), size=(0.01, 0.01, 0.01), fixed=True))
box5 = scene.add_entity(gs.morphs.Box(pos=(0, -0.5, 0.1), size=(0.01, 0.01, 0.01), fixed=True))
box6 = scene.add_entity(gs.morphs.Box(pos=(-0.5, -0.2, 0.1), size=(0.01, 0.01, 0.01), fixed=True))
box7 = scene.add_entity(gs.morphs.Box(pos=(-0.3, 0.3, 0.4), size=(0.01, 0.01, 0.01), fixed=True))
box8 = scene.add_entity(gs.morphs.Box(pos=(0.4, 0.0, 0.2), size=(0.01, 0.01, 0.01), fixed=True))

# ---------------------------------- la cam ----------------------------------
HERE = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(HERE, "out", "02_attached_camera.mp4")

FPS = 30

# Wrist camera pose in the QARM "hand" link frame. The camera sits at
# (10 cm, 0 cm, 0 cm) in the hand frame and is rotated to look at the gripper
# fingertip at (0, 0, 10 cm) — the rotation is a standard look-at with
# hand -Z as the up hint, which is world up during the grasp because the
# wrist's R_x(180°) pose flips Z between hand and world. That keeps the
# closing jaws centered as the gripper descends onto the cube.
WRIST_OFFSET_T = np.array(
    [
        [1.0, 0.0, 0.0, 0.10],
        [0.0, -1.0, 0.0, 0.00],
        [0.000000, 0.0, -1.0, 0.00],
        [0.000000, 0.000000, 0.000000, 1.00],
    ],
    dtype=np.float64,
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
# Camera intrinsics + attachment to the QARM wrist + lighting. The
# ``lights`` list is plain Genesis sensor config; the plugin converts each
# dict into an Nyx ``LightAsset`` at build time.

cam = scene.add_sensor(
    gs.sensors.RasterizerCameraOptions(
        res=(1274, 956),
        fov=60.0,
        near=0.02,
        far=50.0,
        entity_idx=qarm_entity.idx,
        link_idx_local=qarm_entity.get_link("END_EFFECTOR").idx_local,
        offset_T=WRIST_OFFSET_T,
        lights=[
            {
                "type": "directional",
                "dir": (-0.4, -0.4, -0.8),
                "color": (1.0, 1.0, 1.0),
                "intensity": 5.0,
                "shadow": True,
            }
        ],
    )
)
# Stream RGB frames to an MP4 file. ``cam.read().rgb`` is shape
# (H, W, 3) uint8 in single-env mode.
scene.start_recording(
    data_func=lambda: cam.read().rgb,
    rec_options=gs.recorders.VideoFile(filename=OUTPUT_PATH, fps=FPS),
)
# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    "YAW",
    "SHOULDER",
    "ELBOW",
    "WRIST",
    "JOINT1A",
    "JOINT2A",
    "JOINT1B",
    "JOINT2B",
]

dofs_idx = [qarm_entity.get_joint(name).dofs_idx_local[0] for name in joint_names]

qarm_entity.set_dofs_force_range(
    lower=np.array([-87]),
    upper=np.array([87]),
    dofs_idx_local=dofs_idx,
)

qarm_controller = SimQArmController(
    timestep=time_per_step,
    # Kp=441.09732585520067 * np.diag([1, 1, 1]),
    # Kd=195.40127271879317 * np.diag([1, 1, 1]),
    # Ki=104.63321198526842 * np.diag([1, 1, 1]),
    Kp=747.6153381864258 * np.diag([1, 1, 1]),
    Kd=219.43740842413885 * np.diag([1, 1, 1]),
    Ki=114.4689596149386 * np.diag([1, 1, 1]),
    qarm_entity=qarm_entity,
    dofs_idx=dofs_idx,
    gs=gs,
)

ini_waypoint = Waypoint(
    position=np.array([0.2, 0.0, 0.5]).reshape(3, 1),
    velocity=None,
    acceleration=None,
)

waypoints = [
    Waypoint(
        position=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([0.3, 0.3, 0.5]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([0.3, 0.3, 0.1]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([-0.3, -0.3, 0.6]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([0, -0.5, 0.1]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([-0.5, -0.2, 0.1]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([-0.3, 0.3, 0.4]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([0.4, 0.0, 0.2]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
]

# waypoints = [Waypoint(
#         position=np.array([0.3, 0.0, 0.]).reshape(3, 1),
#         velocity=None,
#         acceleration=None,
# ),]

multi_trajectory_mission = MultiTrajectoryMission(
    waypoints=waypoints,
    ini_waypoint=ini_waypoint,
)
qarm_controller.add_mission(multi_trajectory_mission)

waypoint_start = Waypoint(
    position=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
    velocity=None,
    acceleration=None,
)
qarm_controller.set_pos(waypoint_start)
start_time = time.perf_counter()

# target_dof = qarm_controller.qarm_entity.get_joint("WRIST").dofs_idx_local[0]

# To stop the gripper parts from moving, part 1
gripper_dofs = dofs_idx[3:]
print("gripper_dofs", gripper_dofs)

qarm_controller.qarm_entity.set_dofs_kp([4000.0] * len(gripper_dofs), dofs_idx_local=gripper_dofs)
qarm_controller.qarm_entity.set_dofs_kv([100.0] * len(gripper_dofs), dofs_idx_local=gripper_dofs)

##############
# Go spurs go
for _ in range(10 * 1000):
    qarm_controller._update_packet()

    cmd = qarm_controller.compute_command(scene.cur_t)
    if cmd is not None:
        # cmd[4, 0] = -1*np.sin(2 * np.pi * 0.05 * scene.cur_t)  # Gripper command oscillation
        qarm_controller._send_command(cmd)

        # To stop the gripper parts from moving, part 2
        current_pos = qarm_controller.qarm_entity.get_dofs_position(dofs_idx_local=gripper_dofs)
        qarm_controller.qarm_entity.control_dofs_position(current_pos, dofs_idx_local=gripper_dofs)

    scene.step()

    if qarm_controller.finish_asked_missions:
        break

scene.stop_recording()
print(f"Saved {OUTPUT_PATH}")
