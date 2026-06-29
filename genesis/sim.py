"""
This is an example code for using genesis-world directly with the SimQArmController. It's the base for RL algorithms.
"""

import time

import numpy as np

import genesis as gs
from robot_control.missions import CircleMission, MultiTrajectoryMission, SquareMission
from robot_control.robots.qarm import SimQArmController
from robot_control.utils import Waypoint

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
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        gravity=(0, 0, -9.80665),
        dt=0.01,
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
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),  # (2560, 1920),
        camera_pos=(-1.5, 1.5, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
    show_viewer=True,
)
# ------------------------------- add entities ------------------------------
plane = scene.add_entity(gs.morphs.Plane())
qarm_entity = scene.add_entity(
    # genesis/QARM/urdf/qarm_with_gripper.urdf
    gs.morphs.URDF(file="genesis/QARM/urdf/QARM.urdf", fixed=True),
)
# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    "YAW",
    "SHOULDER",
    "ELBOW",
    "WRIST",
    # "JOINT1A",
    # "JOINT2A",
    # "JOINT1B",
    # "JOINT2B",
]

dofs_idx = [qarm_entity.get_joint(name).dofs_idx_local[0] for name in joint_names]

qarm_entity.set_dofs_force_range(
    lower=np.array([-87]),
    upper=np.array([87]),
    dofs_idx_local=dofs_idx,
)
qarm_controller = SimQArmController(
    timestep=0.01,
    Kp=np.diag([100, 100, 100]),
    Kd=np.diag([10, 10, 10]),
    Ki=np.diag([0, 0, 0]),
    qarm_entity=qarm_entity,
    dofs_idx=dofs_idx,
)

qarm_controller.connect()

# ################################
# Creating the missions sequence
# # 5 squares #######
# mission_square = SquareMission(
#     center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
#     side_length=0.4,
#     plane=np.array([1.0, 0.0, 1.0]).reshape(3, 1),
#     nb_of_squares=1,
#     time_per_side=5.0,
# )
# qarm_controller.add_mission(mission_square)
# # 5 circles #######
# mission_circle = CircleMission(
#     center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
#     radius=0.2,
#     plane=np.array([1.0, 0.0, 0.0]).reshape(3, 1),
#     nb_of_circles=1,
#     time_per_circle=10.0,
#     timestep=qarm_controller.timestep,
# )
# qarm_controller.add_mission(mission_circle)

waypoints = [
    Waypoint(
        position=np.array([0.0, 0.0, 0.5]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
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
        position=np.array([-0.5, 0.0, 0.1]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([-0.3, 0.3, 0.4]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([0.0, 0.0, 0.2]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
]

waypoints = [
    Waypoint(
        position=np.array([0.3, 0.3, 0.2]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
    Waypoint(
        position=np.array([-0.3, -0.3, 0.2]).reshape(3, 1),
        velocity=None,
        acceleration=None,
    ),
]

multi_trajectory_mission = MultiTrajectoryMission(
    waypoints=waypoints,
    ini_waypoint=waypoints[0],
    time_per_segment=5.0,
)

qarm_controller.add_mission(multi_trajectory_mission)

strat_time = time.perf_counter()
##############
# Go spurs go
for _ in range(10 * 1000):
    # print(
    #    f"Time: {scene.cur_t:.2f} s vs Real Time: {time.perf_counter() - strat_time:.2f} s",
    #    end="\r",
    # )

    qarm_controller._update_packet()

    cmd = qarm_controller.compute_command(scene.cur_t)
    if cmd is not None:
        qarm_controller._send_command(cmd)

    scene.step()

    if qarm_controller.finish_asked_missions:
        break
