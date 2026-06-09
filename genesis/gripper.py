import socket

import numpy as np

import genesis as gs

# ------------------------------ init Genesis ------------------------------
gs.init(
    backend=gs.gpu,
    precision="32",
    seed=None,
    debug=False,
    performance_mode=False,  # to change when training (to gain 30% of performance)
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
        show_link_frame=True,  # visualizing the coordinate frames of entity links
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
gripper = scene.add_entity(
    gs.morphs.URDF(file="QARM/urdf/gripper.urdf", fixed=True),
)
# cube = scene.add_entity(
#    gs.morphs.Box(size=(0.1, 0.3, 0.4), pos=(0.3, 0, 0.2)),
# )
# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    "JOINT1A",
    "JOINT2A",
    "JOINT1B",
    "JOINT2B",
]

dofs_idx = [gripper.get_joint(name).dofs_idx_local[0] for name in joint_names]

gripper.set_dofs_force_range(
    lower=np.array([-87]),
    upper=np.array([87]),
    dofs_idx_local=dofs_idx,
)

for _ in range(3000):
    scene.step()
