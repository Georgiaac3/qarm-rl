import numpy as np

import genesis as gs
from robot_control.robots.qarm import SimQArmController


def create_black_box_function(qarm_controller):
    """
    Creates a black-box function that gives the performance of real PID gains for a given sequence of predefined trajectory missions.

    Returns:
    - A function that takes Kp, Ki, and Kd as inputs and returns the performance metric.
    """

    def black_box_function(Kp, Ki, Kd):
        """
        A black-box function that gives the performance of real PID gains for a given sequence of predefined trajectory missions.

        Parameters:
        - Kp: Proportional gain
        - Ki: Integral gain
        - Kd: Derivative gain

        Returns:
        - The reward (error metric)
        """
        # Set the PID gains in the QArm controller
        qarm_controller.Kp = Kp * np.eye(3)
        qarm_controller.Ki = Ki * np.eye(3)
        qarm_controller.Kd = Kd * np.eye(3)

        # Reset the QArm controller and the missions
        # Create the missions sequence

        qarm_controller.set_state(waypoint_start)

        return performance_metric

    return black_box_function


def main():
    ######################################################
    # Create the genesis environment (scene, entity, etc)
    # ------------------------------ init Genesis ------------------------------
    gs.init(
        backend=gs.gpu,
        precision="32",
        seed=None,
        debug=False,
        performance_mode=True,  # set to True when training (to gain 30% of performance)
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
        show_viewer=False,
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

    #######################################################################################################################
    # Create our custom QArm sim controller from the genesis qarm entity that will interact directly with the genesis QArm
    qarm_controller = SimQArmController(
        timestep=0.01,
        Kp=np.diag([100, 100, 100]),
        Kd=np.diag([10, 10, 10]),
        Ki=np.diag([0, 0, 0]),
        qarm_entity=qarm_entity,
        dofs_idx=dofs_idx,
    )

    ###############################################################
    # Create the black-boc function using this QArm sim controller
    black_box_function = create_black_box_function(qarm_controller)


if __name__ == "__learn__":
    main()
