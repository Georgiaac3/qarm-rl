import genesis as gs
import numpy as np

# ------------------------------ init Genesis ------------------------------
gs.init(
    backend             = gs.gpu,
    precision           = '32',
    seed                = None,
    debug               = False,
    performance_mode    = False, # to change when training (to gain 30% of performance)
    logging_level       = 'warning',
    theme               = 'light',
    logger_verbose_time = False,
)

# ------------------------------ create scene ------------------------------
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        gravity=(0, 0, -9.80665),
        dt=0.01,
        substeps=1,
    ),  
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = True, # visualizing the coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = False, # turn off plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),#(2560, 1920),
        camera_pos    = (-1.5, 1.5, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    renderer       = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
    show_viewer    = True,
)
# ------------------------------- add entities ------------------------------
plane = scene.add_entity(gs.morphs.Plane())
cylinder = scene.add_entity(
    gs.morphs.URDF(file="QARM/urdf/test.urdf", fixed=True),
)
# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    'world_base_joint',
]

dofs_idx = [cylinder.get_joint(name).dofs_idx_local[0] for name in joint_names]

cylinder.set_dofs_force_range(
    lower          = np.array([-87]),
    upper          = np.array([ 87]),
    dofs_idx_local = dofs_idx,
)

# ------------------------------- simulate ------------------------------
def counter_gravity():
    phi_mes = np.array(cylinder.get_dofs_position(dofs_idx).cpu().detach().numpy())[0]
    cylinder.control_dofs_force([torque(phi_mes)], dofs_idx)

def torque(phi_mes):
    return (0.6/2)*9.80665*np.cos(np.pi/2+phi_mes)

cylinder.set_dofs_position(np.array([-0.78]),dofs_idx)
counter_gravity()
scene.step()

for i in range(300):
    counter_gravity()
    scene.step()