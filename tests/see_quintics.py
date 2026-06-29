import matplotlib.pyplot as plt
import numpy as np

from robot_control.utils import Waypoint, get_desired_state, get_quintic_coeffs_and_time

##############
# Preparation
waypoint1 = Waypoint(
    position=np.array([0.3, 0.3, 0.2]).reshape(3, 1),
    velocity=None,
    acceleration=None,
)
waypoint2 = Waypoint(
    position=np.array([-0.3, -0.3, 0.2]).reshape(3, 1),
    velocity=None,
    acceleration=None,
)

A, tf = get_quintic_coeffs_and_time(waypoint1, waypoint2, tf=5.0)
waypoints = [get_desired_state(t, A) for t in np.linspace(0, tf, num=100)]
positions = [wp.position.flatten() for wp in waypoints]
x_pos = [pos[0] for pos in positions]
y_pos = [pos[1] for pos in positions]
z_pos = [pos[2] for pos in positions]
velocities = [
    wp.velocity.flatten() if wp.velocity is not None else np.zeros_like(wp.position.flatten())
    for wp in waypoints
]
x_vel = [vel[0] for vel in velocities]
y_vel = [vel[1] for vel in velocities]
z_vel = [vel[2] for vel in velocities]
accelerations = [
    (
        wp.acceleration.flatten()
        if wp.acceleration is not None
        else np.zeros_like(wp.position.flatten())
    )
    for wp in waypoints
]
x_acc = [acc[0] for acc in accelerations]
y_acc = [acc[1] for acc in accelerations]
z_acc = [acc[2] for acc in accelerations]

###########
# Plotting

# Création d'une figure avec une disposition spécifique
# 'A' est le 3D, 'B' et 'C' sont la vitesse et l'accélération
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2)

# Graphique 3D à gauche (occupe les deux lignes de la première colonne)
ax_3d = fig.add_subplot(gs[:, 0], projection="3d")
ax_3d.set_title("Trajectory in 3D Space")
ax_3d.plot(x_pos, y_pos, z_pos, alpha=0.7)
start_point = (x_pos[0], y_pos[0], z_pos[0])
end_point = (x_pos[-1], y_pos[-1], z_pos[-1])

# Rotation du graphique 3D
# azim=135 permet de basculer X à gauche et Y à droite
ax_3d.view_init(elev=20, azim=135)

# Ajout des vecteurs de base (repère X, Y, Z) à l'origine (0,0,0)
# quiver(x, y, z, dx, dy, dz)
length = 0.1  # Longueur des vecteurs
ax_3d.quiver(0, 0, 0, length, 0, 0, color="red", label="X")
ax_3d.quiver(0, 0, 0, 0, length, 0, color="green", label="Y")
ax_3d.quiver(0, 0, 0, 0, 0, length, color="blue", label="Z")

# Ajout des points sur le graphique 3D
# 's' définit la taille du point, 'c' la couleur, 'label' pour la légende
ax_3d.scatter(*start_point, color="green", s=50, label="Départ", marker="o")
ax_3d.scatter(*end_point, color="red", s=50, label="Arrivée", marker="x")

# Graphique Vitesse en haut à droite
ax_vel = fig.add_subplot(gs[0, 1])
ax_vel.set_title("Velocity Components")
ax_vel.plot(x_vel, label="Vx", alpha=0.5)
ax_vel.plot(y_vel, label="Vy", alpha=0.5)
ax_vel.plot(z_vel, label="Vz", alpha=0.5)
ax_vel.legend()

# Graphique Accélération en bas à droite
ax_acc = fig.add_subplot(gs[1, 1])
ax_acc.set_title("Acceleration Components")
ax_acc.plot(x_acc, label="Ax", alpha=0.5)
ax_acc.plot(y_acc, label="Ay", alpha=0.5)
ax_acc.plot(z_acc, label="Az", alpha=0.5)
ax_acc.legend()

plt.tight_layout()
plt.show()
