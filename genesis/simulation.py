import genesis as gs
from genesis.utils.misc import tensor_to_array
import numpy as np
import time
import socket
import struct
import csv
from datetime import datetime
from pathlib import Path

from dynamics import L5

from helper import get_tau_from_pwm

# Choose what to display in the live Genesis plot: "cartesian", "joint", or "both"
LIVE_PLOT_MODE = "joint"


def save_plot_records(csv_path: Path, records):
    """Save live-plot samples to a dedicated CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "x_vel", "y_vel", "z_vel",
            "base_vel", "shoulder_vel", "elbow_vel", "wrist_vel",
        ])
        writer.writerows(records)

# ------------------------------ UDP Server Setup ------------------------------
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
if hasattr(socket, 'SO_REUSEPORT'):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

fmt_recv = "5d"   # 5 doubles reçus (pwm[4] + grip)
fmt_send = "8d"   # 8 doubles envoyés (phi_mes[4] + dphi_mes[4])

print(f"Serveur UDP actif sur le port {UDP_PORT}")

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
qarm = scene.add_entity(
    gs.morphs.URDF(file="QARM/urdf/QARM_new.urdf", fixed=True),
)
cube = scene.add_entity(
    gs.morphs.Box(size=(0.1, 0.3, 0.4), pos=(0.3, 0, 0.2)),
)

# ------------------------------- add camera ------------------------------
cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 3.5, 2.5),
    lookat = (0, 0, 0),
    fov    = 30,
    GUI    = True
)

# ------------------------------- plot configuration ------------------------------
end_effector_link = qarm.get_link('END_EFFECTOR')
plot_records = []
plot_start_t = time.perf_counter()

def get_plot_data():
    # Compute end-effector linear velocity from Jacobian.
    local_offset = np.array([0.0, 0.0, L5])
    jacobian = tensor_to_array(qarm.get_jacobian(end_effector_link, local_point=local_offset))
    joint_vels = tensor_to_array(qarm.get_dofs_velocity())
    target_vel = jacobian @ joint_vels

    linear_velocity = tuple(target_vel[0:3].tolist())
    joint_velocity = tuple(joint_vels[:4].tolist())
    plot_records.append((time.perf_counter() - plot_start_t, *linear_velocity, *joint_velocity))

    if LIVE_PLOT_MODE == "cartesian":
        return {"velocity": linear_velocity}
    if LIVE_PLOT_MODE == "joint":
        return {"joint_velocities": joint_velocity}
    return {
        "velocity": linear_velocity,
        "joint_velocities": joint_velocity,
    }


plot_labels = {}
if LIVE_PLOT_MODE in ("cartesian", "both"):
    plot_labels["velocity"] = ("x_vel", "y_vel", "z_vel")
if LIVE_PLOT_MODE in ("joint", "both"):
    plot_labels["joint_velocities"] = ("base_vel", "shoulder_vel", "elbow_vel", "wrist_vel")

# Start live plotter  
#plotter = scene.start_recording(
#    data_func=get_plot_data,
#    rec_options=gs.recorders.MPLLinePlot(
#        labels=plot_labels,
#        title="Box Dynamics",
#        history_length=100,
#        window_size=(600, 400),
#        hz=10,  # Update frequency (Hz)
#        show_window=True
#    )
#)

# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    'YAW',
    'SHOULDER',
    'ELBOW',
    'WRIST',
    #'FINGER_MASTER_JOINT',
    #'FINGER_MIMIC_JOINT'
]

dofs_idx = [qarm.get_joint(name).dofs_idx_local[0] for name in joint_names]

qarm.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87]),
    upper          = np.array([ 87,  87,  87,  87]),
    dofs_idx_local = dofs_idx,
)
# ------------------------------- simulate ------------------------------
#cam.start_recording()

for i in range(300):
    scene.step()

for i in range(300):
    phi_mes = np.array(qarm.get_dofs_position(dofs_idx).cpu().detach().numpy()).reshape(4, 1)
    dphi_mes = np.array(qarm.get_dofs_velocity(dofs_idx).cpu().detach().numpy()).reshape(4, 1)
    print(dphi_mes)
    
    # ============ Réception via UDP ============
    pwm = np.array([[0], [0], [0], [0]])  # Valeur par défaut
    
    try:
        data, addr = sock.recvfrom(struct.calcsize(fmt_recv))
        if len(data) == struct.calcsize(fmt_recv):
            received = struct.unpack(fmt_recv, data)
            # received[0:4] = pwm values
            pwm = np.array(received[0:4]).reshape(-1, 1)
            #print(f"Reçu du client {addr} - PWM: {received[0:4]}")
            
            # ============ Envoi via UDP ============
            # Préparer la réponse : phi_mes[4] + dphi_mes[4]
            response_data = tuple(phi_mes.flatten().tolist()) + tuple(dphi_mes.flatten().tolist())
            response_msg = struct.pack(fmt_send, *response_data)
            sock.sendto(response_msg, addr)
            
    except BlockingIOError:
        # Pas de réception, utiliser valeur par défaut
        pass
    except Exception as e:
        print(f"Erreur UDP: {e}")

    if i >= 200:
        if i == 200:
            print("Début de l'application du contrôle à partir de l'itération 100")
        tau_cmd = get_tau_from_pwm(
            pwm = pwm,
            dphi_mes = dphi_mes
        ).reshape(-1)
        qarm.control_dofs_force(
            tau_cmd,
            dofs_idx,
        )
        if i >= 200:
            cube.set_pos(
                pos=(0.3 , 0, -0.3),
            )
    #else:
    #    qarm.set_dofs_position(
    #        np.array([0, np.pi/4, -np.pi/4, 0]),
    #        dofs_idx,
    #    )
    #    qarm.set_dofs_velocity(
    #        np.zeros(4),
    #        dofs_idx,
    #    )
    scene.step()

    #print('dofs position:', np.array(qarm.get_dofs_position(dofs_idx).cpu().detach().numpy()).reshape(-1, 1))

    #print('dofs velocity:', np.array(qarm.get_dofs_velocity(dofs_idx).cpu().detach().numpy()).reshape(-1, 1))

    #print('control force:', qarm.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    #print('internal force:', qarm.get_dofs_force(dofs_idx))

    # change camera position
    #cam.set_pose(
    #    pos    = (2 * np.sin(i / 60), 2 * np.cos(i / 60), 1.5),
    #    lookat = (0, 0, 0.2),
    #    up     = (0, 0, 1)
    #)
    
    #cam.render()

    # Get link state  
    pos = tensor_to_array(end_effector_link.get_pos()).reshape(1, 3)
    quat = tensor_to_array(end_effector_link.get_quat()).reshape(1, 4)
    vel = tensor_to_array(end_effector_link.get_vel()).reshape(1, 3)
  
    # Apply local offset  
    local_offset = np.array([0.0, 0.0, L5])
    world_offset = gs.utils.geom.transform_by_quat(local_offset, quat.reshape(-1))  # Rotate offset to world frame using link orientation
    target_point = pos + world_offset
      
    # Highlight the link position  
    scene.draw_debug_sphere(  
        pos=target_point.flatten(),  
        radius=0.01,  
        color=(1.0, 0.0, 0.0, 1.0)  
    )
      
    ## Optional: Draw velocity vector  
    #if np.linalg.norm(vel) > 0.01:  
    #    vel_end = pos + vel * 0.1  # Scale velocity for visualization  
    #    scene.draw_debug_path(  
    #        qposs=np.array([pos, vel_end]),  
    #        entity=qarm,  
    #        link_idx=end_effector_link.idx  
    #    )

# Fermeture du socket UDP
sock.close()
print("Serveur UDP fermé")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = Path("plot_data") / f"live_plot_{timestamp}.csv"
latest_csv = Path("plot_data") / "live_plot_latest.csv"

save_plot_records(output_csv, plot_records)
save_plot_records(latest_csv, plot_records)
print(f"Données du live plot sauvegardées dans: {output_csv}")
print(f"Dernière capture mise à jour dans: {latest_csv}")

# stop recording and save video. If `filename` is not specified, a name will be auto-generated using the caller file name.
#cam.stop_recording(save_to_filename='video.mp4', fps=60)