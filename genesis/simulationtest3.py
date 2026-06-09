import genesis as gs
import numpy as np
import socket
import struct
from time import perf_counter

from dynamics import get_gravity_vector
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
# ------------------------------- build scene ------------------------------
scene.build()

# ------------------------------- getting ready ------------------------------
joint_names = [
    'YAW',
    'SHOULDER',
    'ELBOW',
    'WRIST',
    #'JOINT1A',
    #'JOINT2A',
    #'JOINT1B',
    #'JOINT2B',
]

dofs_idx = [qarm.get_joint(name).dofs_idx_local[0] for name in joint_names]

qarm.set_dofs_force_range(
    lower          = np.array([-87]),
    upper          = np.array([ 87]),
    dofs_idx_local = dofs_idx,
)

# ------------------------------- simulate ------------------------------
def get_phi_mes():
    phi_mes = np.array(qarm.get_dofs_position(dofs_idx).cpu().detach().numpy())
    return phi_mes

def get_dphi_mes():
    dphi_mes = np.array(qarm.get_dofs_velocity(dofs_idx).cpu().detach().numpy())
    return dphi_mes

def get_response():
    phi_mes = get_phi_mes()
    dphi_mes = get_dphi_mes()
    # Préparer la réponse : phi_mes[4] + dphi_mes[4]
    response_data = tuple(phi_mes.flatten().tolist()) + tuple(dphi_mes.flatten().tolist())
    response_msg = struct.pack(fmt_send, *response_data)
    return response_msg


def try_connect():
    try:
        data, addr = sock.recvfrom(struct.calcsize(fmt_recv))
        if len(data) == struct.calcsize(fmt_recv):
            received = struct.unpack(fmt_recv, data)
            
            # send a response any ways
            response_msg = get_response()
            sock.sendto(response_msg, addr)

            if received != (0.0, -0.1, -0.1, 0.0, 0.0):
                apply_torques(np.array(received[0:4]))
                return True
    except BlockingIOError:
        # Pas de connexion, continuer à attendre
        pass
    except Exception as e:
        print(f"Erreur lors de la tentative de connexion: {e}")
    
    return False

def get_torques():
    #g =  get_gravity_vector(get_phi_mes().reshape(4, 1)).reshape(4)

    torques = np.zeros(4)  # Valeur par défaut
    
    try:
        data, addr = sock.recvfrom(struct.calcsize(fmt_recv))
        if len(data) == struct.calcsize(fmt_recv):
            received = struct.unpack(fmt_recv, data)
            # received[0:4] = torques values
            torques = np.array(received[0:4])

            # ============ Envoi via UDP ============
            response_msg = get_response()
            sock.sendto(response_msg, addr)
        else:
            print(f"Message reçu de taille inattendue: {len(data)} octets")
    except BlockingIOError:
        # Pas de réception, utiliser valeur par défaut
        pass
    except Exception as e:
        print(f"Erreur UDP: {e}")

    # return g

    return torques

def apply_torques(torques):
    qarm.control_dofs_force(torques, dofs_idx)

for _ in range(10*3000):
    #qarm.set_dofs_position(np.array([0., np.pi/8, np.pi/4, 0.]),dofs_idx)
    
    scene.step()
    #print("Attente de connexion du client...")
    connexion = try_connect()
    if connexion:
        print("Connexion établie avec le client.")
        break

    #com_pos = qarm.get_links_pos(links_idx_local=[4], ref="link_com")
  #
    ## Draw a sphere at the COM position  
    #scene.draw_debug_sphere(pos=com_pos, radius=0.01, color=(1, 0, 0, 1))

scene.step()

cube.set_pos(pos=(0.3 , 0, -0.3))

for _ in range(1000*10):
    torques_to_apply = get_torques()
    apply_torques(torques_to_apply)

    scene.step()
