import numpy as np
from bayes_opt import BayesianOptimization

import genesis as gs
from robot_control.missions import MultiTrajectoryMission
from robot_control.robots.qarm import SimQArmController
from robot_control.utils import Waypoint


def get_computed_trajectory():
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

    trajectory_mission = MultiTrajectoryMission(waypoints, ini_waypoint)
    trajectory_mission.compute_trajectory()

    return trajectory_mission, ini_waypoint


def create_black_box_function(qarm_controller: SimQArmController, scene: gs.Scene):
    """
    Creates a black-box function that gives the performance of real PID gains for a given sequence of predefined trajectory missions.

    Returns:
    - A function that takes Kp, Ki, and Kd as inputs and returns the performance metric.
    """

    computed_trajectory, waypoint_start = (
        get_computed_trajectory()
    )  # Assuming this function is defined elsewhere to get the computed trajectory

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
        print(f"Evaluating PID gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        # Set the PID gains in the QArm controller
        qarm_controller.Kp = Kp * np.eye(3)
        qarm_controller.Ki = Ki * np.eye(3)
        qarm_controller.Kd = Kd * np.eye(3)

        # Reset the missions that are allready computed
        qarm_controller.missions.clear()
        qarm_controller.missions.append(computed_trajectory)
        qarm_controller.finish_asked_missions = False

        # Reset the QArm controller to its initial state
        qarm_controller.set_pos(waypoint_start)

        # The performance metric is negative = (position error + velocity error)/total number of steps
        perf = 0
        nb_steps = 0

        # Run the simulation for the duration of the trajectory
        for i in range(
            100 * 1000
        ):  # This is a large number to ensure the simulation runs long enough for the trajectory to complete
            qarm_controller._update_packet()

            cmd = qarm_controller.compute_command(scene.cur_t)
            if cmd is not None:
                qarm_controller._send_command(cmd)

            # Computes the performance metric
            last_X_mes = qarm_controller.last_X_mes
            last_X_des = qarm_controller.last_X_des
            last_dX_mes = qarm_controller.last_dX_mes
            last_dX_des = qarm_controller.last_dX_des

            perf -= np.linalg.norm(last_X_mes - last_X_des) + np.linalg.norm(
                last_dX_mes - last_dX_des
            )

            scene.step()

            if qarm_controller.finish_asked_missions:
                nb_steps = i + 1
                break

        perf /= nb_steps

        return float(perf)

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
        gs=gs,
        metrics=True,
    )

    ###############################################################
    # Create the black-boc function using this QArm sim controller
    black_box_function = create_black_box_function(qarm_controller, scene)

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds={
            "Kp": (0, 800),
            "Ki": (0, 800),
            "Kd": (0, 800),
        },
        random_state=42,
    )
    optimizer.maximize(
        init_points=5,
        n_iter=25,
    )

    print("Best PID gains found:")
    print(optimizer.max)


if __name__ == "__main__":
    main()
