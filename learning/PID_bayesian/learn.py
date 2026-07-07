import numpy as np
from bayes_opt import BayesianOptimization

import genesis as gs
from robot_control.missions import CircleMission, MultiTrajectoryMission, SquareMission
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


def get_computed_circles(timestep):
    mission_circle = CircleMission(
        center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
        radius=0.2,
        plane=np.array([1.0, 0.0, 0.0]).reshape(3, 1),
        nb_of_circles=5,
        time_per_circle=10.0,
        timestep=timestep,
    )
    mission_circle.compute_trajectory()

    return mission_circle


def get_computed_squares():
    mission_square = SquareMission(
        center=np.array([0.3, 0.0, 0.5]).reshape(3, 1),
        side_length=0.4,
        plane=np.array([1.0, 0.0, 1.0]).reshape(3, 1),
        nb_of_squares=50,
        time_per_side=5.0,
    )
    mission_square.compute_trajectory()

    return mission_square


def create_black_box_function(qarm_controller: SimQArmController, scene: gs.Scene):
    """
    Creates a black-box function that gives the performance of real PID gains for a given sequence of predefined trajectory missions.

    Returns:
    - A function that takes Kp, Ki, and Kd as inputs and returns the performance metric.
    """

    computed_trajectory, waypoint_start = get_computed_trajectory()

    def black_box_function(Kp1, Kp2, Kp3, Ki1, Ki2, Ki3, Kd1, Kd2, Kd3):
        """
        A black-box function that gives the performance of real PID gains for a given sequence of predefined trajectory missions.

        Parameters:
        - Kp1, Kp2, Kp3: Proportional gains
        - Ki1, Ki2, Ki3: Integral gains
        - Kd1, Kd2, Kd3: Derivative gains

        Returns:
        - The reward (error metric)
        """
        # Set the PID gains in the QArm controller
        qarm_controller.Kp = np.array([[Kp1, 0, 0], [0, Kp2, 0], [0, 0, Kp3]])
        qarm_controller.Ki = np.array([[Ki1, 0, 0], [0, Ki2, 0], [0, 0, Ki3]])
        qarm_controller.Kd = np.array([[Kd1, 0, 0], [0, Kd2, 0], [0, 0, Kd3]])

        # Reset the scene
        scene.reset()

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

            perf -= i * np.linalg.norm(last_X_mes - last_X_des) + i * np.linalg.norm(
                last_dX_mes - last_dX_des
            )

            scene.step()

            if qarm_controller.finish_asked_missions:
                nb_steps = i + 1
                break

        # perf /= nb_steps

        return float(perf)

    return black_box_function


def main():
    ######################################################
    # Create the genesis environment (scene, entity, etc)
    # ------------------------------ init Genesis ------------------------------
    gs.init(
        performance_mode=True,  # set to True when training (to gain 30% of performance)
        logging_level="warning",
    )

    # ------------------------------ create scene ------------------------------
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0, 0, -9.80665),
            dt=0.02,
            substeps=2,
        ),
        # renderer=gs.renderers.Rasterizer(),  # using rasterizer for camera rendering
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
            "Kp1": (0, 800),
            "Kp2": (0, 800),
            "Kp3": (0, 800),
            "Ki1": (0, 800),
            "Ki2": (0, 800),
            "Ki3": (0, 800),
            "Kd1": (0, 800),
            "Kd2": (0, 800),
            "Kd3": (0, 800),
        },
        random_state=42,
    )

    optimizer.load_state("learning/PID_bayesian/PID_bayesian_optimization_state_9var.json")

    optimizer.maximize(
        init_points=20,
        n_iter=800,
    )

    optimizer.save_state("learning/PID_bayesian/PID_bayesian_optimization_state_9var.json")

    print("Best PID gains found:")
    print(optimizer.max)


if __name__ == "__main__":
    main()
