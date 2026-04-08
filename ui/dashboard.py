from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np

from core.config import settings
from core.dynamics import L1, L2, L3, L4, L5


class RealTimeApp:
    def __init__(self, queue):
        self.queue = queue

        self.loop_delay = settings.timestep
        self.engine_frequency = 1 / self.loop_delay
        self.max_seconds = 5
        self.max_points = int(self.max_seconds * self.engine_frequency)

        self.vars_2d_groups = settings.graphs_2d
        self.all_2d_vars = [var for group in self.vars_2d_groups.values() for var in group]
        self.vars_3d = settings.graphs_3d

        self.x_data = list(np.linspace(-self.max_seconds, 0, self.max_points))
        self.y_data = {
            var: deque([0.0] * self.max_points, maxlen=self.max_points) for var in self.all_2d_vars
        }

        # Pour la 3D : on stocke les coordonnées projetées (u, v)
        self.data_3d = {
            var: {"u": deque(maxlen=500), "v": deque(maxlen=500)} for var in self.vars_3d
        }

        # Rotation autour de l'axe Z (en radians)
        self.rotation_z = 0

        # Longueurs du bras robot
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.L5 = L5

        self.setup_dpg()

    def project_3d_to_2d(self, x, y, z):
        """
        Projection isométrique simple avec rotation autour de l'axe Z :
        u = projection horizontale
        v = projection verticale (incluant la hauteur Z)
        """
        # Appliquer la rotation autour de l'axe Z
        cos_rz, sin_rz = np.cos(self.rotation_z), np.sin(self.rotation_z)
        x_rot = x * cos_rz - y * sin_rz
        y_rot = x * sin_rz + y * cos_rz

        # Angles de vue (30° environ pour un bel effet 3D)
        cos_a, sin_a = 0.866, 0.5
        cos_b, sin_b = 0.866, 0.5

        u = x_rot * cos_a - y_rot * sin_a
        v = x_rot * sin_a * sin_b + y_rot * cos_a * sin_b + z * cos_b
        return u, v

    def get_arm_joints(self, angles):
        """
        Calcule les positions 3D de toutes les articulations du bras.
        angles: array [angle0, angle1, angle2, angle3]

        Cinématique du bras:
        - Angle 0: rotation de la base autour de l'axe Z (angle0=0 → bras dans plan (x,z))
        - L1: vertical fixe
        - Angle 1: rotation de L2 autour du joint L1-L2 (angle1=0 → L2 vertical)
        - L2: dans le plan défini par angle0
        - Coude: fixe à π/2 → L3 perpendiculaire à L2
        - L3: perpendiculaire à L2 dans le même plan
        - Angle 2: rotation de L4 autour du joint L3-L4 (angle2=0 → L4 dans continuité de L3)
        - L4: rotation de L3 par angle2, dans le même plan
        - L5: même direction que L4
        - Angle 3: pince (ignoré)

        Retourne: liste de positions [p0, p1, p2, p3, p4, p5] en 3D
        """
        positions = []
        angle0, angle1, angle2 = angles[0], angles[1], angles[2]

        # Position 0: base (origine)
        p0 = np.array([0.0, 0.0, 0.0])
        positions.append(p0)

        # Position 1: après L1 (vertical)
        p1 = p0 + np.array([0.0, 0.0, self.L1])
        positions.append(p1)

        # Position 2: après L2
        # L2 s'étend dans le plan défini par angle0, avec déviation angle1 par rapport à la verticale
        L2_dir = np.array(
            [
                self.L2 * np.sin(angle1) * np.cos(angle0),
                self.L2 * np.sin(angle1) * np.sin(angle0),
                self.L2 * np.cos(angle1),
            ]
        )
        p2 = p1 + L2_dir
        positions.append(p2)

        # Position 3: après L3 (perpendiculaire à L2, coude fixe π/2)
        # L3 est perpendiculaire à L2 dans le même plan
        L3_dir = np.array(
            [
                self.L3 * np.cos(angle1) * np.cos(angle0),
                self.L3 * np.cos(angle1) * np.sin(angle0),
                -self.L3 * np.sin(angle1),
            ]
        )
        p3 = p2 + L3_dir
        positions.append(p3)

        # Position 4: après L4
        # L4 tourne par angle2 dans le même plan (angle1 + angle2)
        L4_dir = np.array(
            [
                self.L4 * np.cos(angle1 + angle2) * np.cos(angle0),
                self.L4 * np.cos(angle1 + angle2) * np.sin(angle0),
                -self.L4 * np.sin(angle1 + angle2),
            ]
        )
        p4 = p3 + L4_dir
        positions.append(p4)

        # Position 5: après L5 (même direction que L4)
        L5_dir = np.array(
            [
                self.L5 * np.cos(angle1 + angle2) * np.cos(angle0),
                self.L5 * np.cos(angle1 + angle2) * np.sin(angle0),
                -self.L5 * np.sin(angle1 + angle2),
            ]
        )
        p5 = p4 + L5_dir
        positions.append(p5)

        return positions

    def setup_dpg(self):
        dpg.create_context()

        # --- THÈME CLAIR DÉFINITIF (Testé DPG 1.x) ---
        with dpg.theme() as global_theme:
            # 1. Style général (Fenêtres, Texte, Boutons)
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (240, 240, 240))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (255, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))  # Texte noir
                dpg.add_theme_color(dpg.mvThemeCol_Header, (200, 200, 200))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (255, 255, 255))

            # 2. Style spécifique aux graphiques (Plots)
            with dpg.theme_component(dpg.mvPlot):
                # Couleur du fond du graphique
                dpg.add_theme_color(
                    dpg.mvPlotCol_PlotBg, (255, 255, 255), category=dpg.mvThemeCat_Plots
                )
                # Couleur de la bordure du cadre
                dpg.add_theme_color(
                    dpg.mvPlotCol_PlotBorder, (150, 150, 150), category=dpg.mvThemeCat_Plots
                )
                # LES GRILLES : Dans DPG 1.x, ce sont ces constantes :
                dpg.add_theme_color(
                    dpg.mvPlotCol_AxisGrid, (200, 200, 200), category=dpg.mvThemeCat_Plots
                )
                # Légende
                dpg.add_theme_color(
                    dpg.mvPlotCol_LegendBg, (240, 240, 240, 150), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_LegendText, (0, 0, 0), category=dpg.mvThemeCat_Plots
                )

        dpg.bind_theme(global_theme)

        dpg.create_viewport(title="QARM Dashboard - Perspective 3D", width=1300, height=1000)
        dpg.setup_dearpygui()

        with dpg.window(label="Contrôle & Visualisation", width=1280, height=980):

            # --- SECTION 2D ---
            with dpg.collapsing_header(label="Mesures Temporelles (2D)", default_open=True):
                with dpg.child_window(height=400, border=False):
                    for group_name, signals in self.vars_2d_groups.items():
                        with dpg.plot(label=group_name, height=180, width=-1):
                            dpg.add_plot_legend()
                            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Temps (s)")
                            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Valeur")
                            for var in signals:
                                dpg.add_line_series(
                                    self.x_data,
                                    list(self.y_data[var]),
                                    label=var,
                                    tag=f"series_{var}",
                                    parent=y_axis,
                                )
                            dpg.set_axis_limits(x_axis, -self.max_seconds, 0)
                            dpg.set_axis_limits_auto(y_axis)

            # --- SECTION 3D PERSPECTIVE ---
            with dpg.collapsing_header(
                label="Espace de Travail (Perspective 3D)", default_open=True
            ):
                for var in self.vars_3d:
                    # On agrandit la fenêtre (height=500)
                    with dpg.plot(label=f"Vue 3D simulée: {var}", height=500, width=-1):
                        dpg.add_plot_legend()
                        # On retire les labels d'axes car ils sont projetés
                        dpg.add_plot_axis(dpg.mvXAxis, label="Plan Horizontal", no_tick_labels=True)
                        y_ax_3d = dpg.add_plot_axis(
                            dpg.mvYAxis, label="Hauteur Z", no_tick_labels=True
                        )

                        # Axes du repère (projection 3D)
                        # Axe X (horizontal) - [0,0,0] -> [1, 0, 0]
                        u_x_origin, v_x_origin = self.project_3d_to_2d(0, 0, 0)
                        u_x_end, v_x_end = self.project_3d_to_2d(0.1, 0, 0)
                        dpg.add_line_series(
                            [u_x_origin, u_x_end],
                            [v_x_origin, v_x_end],
                            label="Axe X",
                            tag=f"axis_x_{var}",
                            parent=y_ax_3d,
                        )

                        # Axe Y (horizontal) - [0,0,0] -> [0, 1, 0]
                        u_y_origin, v_y_origin = self.project_3d_to_2d(0, 0, 0)
                        u_y_end, v_y_end = self.project_3d_to_2d(0, 0.1, 0)
                        dpg.add_line_series(
                            [u_y_origin, u_y_end],
                            [v_y_origin, v_y_end],
                            label="Axe Y",
                            tag=f"axis_y_{var}",
                            parent=y_ax_3d,
                        )

                        # Axe Z (vertical - axe de rotation) - [0,0,0] -> [0, 0, 1]
                        u_z_origin, v_z_origin = self.project_3d_to_2d(0, 0, 0)
                        u_z_end, v_z_end = self.project_3d_to_2d(0, 0, 0.1)
                        dpg.add_line_series(
                            [u_z_origin, u_z_end],
                            [v_z_origin, v_z_end],
                            label="Axe Z (rotation)",
                            tag=f"axis_z_{var}",
                            parent=y_ax_3d,
                        )

                        # Segments du bras robot (5 segments)
                        for i in range(5):
                            dpg.add_line_series(
                                [],
                                [],
                                label=f"Segment {i+1}",
                                tag=f"arm_segment_{i}_{var}",
                                parent=y_ax_3d,
                            )

                        # Trajectoire (ligne)
                        dpg.add_line_series(
                            [], [], label="Historique", tag=f"series_3d_line_{var}", parent=y_ax_3d
                        )
                        # Position actuelle (gros point)
                        dpg.add_scatter_series(
                            [],
                            [],
                            label="Position TCP",
                            tag=f"series_3d_head_{var}",
                            parent=y_ax_3d,
                        )

                        # Fixer les limites pour éviter que le graphique ne "saute" tout le temps
                        dpg.set_axis_limits(y_ax_3d, -0.5, 1.2)

            # --- CONTRÔLES 3D ---
            with dpg.collapsing_header(label="Contrôles 3D", default_open=True):
                dpg.add_slider_float(
                    label="Rotation Z (radians)",
                    default_value=0,
                    min_value=0,
                    max_value=2 * np.pi,
                    tag="rotation_z_slider",
                    width=-1,
                    callback=lambda s, v: setattr(self, "rotation_z", v),
                )

    def update_data(self):
        current_angles = None

        while not self.queue.empty():
            packet = self.queue.get()

            # Récupérer les angles du robot
            if "Angles Articulations mesurés (rad)" in packet:
                current_angles = packet["Angles Articulations mesurés (rad)"]

            # 1. Mise à jour 2D
            for group_name, signals in self.vars_2d_groups.items():
                if group_name in packet:
                    data_recue = packet[group_name]
                    if isinstance(data_recue, (list, tuple)):
                        for i, var_name in enumerate(signals):
                            if i < len(data_recue):
                                self.y_data[var_name].append(data_recue[i])
                    else:
                        self.y_data[signals[0]].append(data_recue)

            # 2. Mise à jour 3D avec Projection
            for var in self.vars_3d:
                if var in packet:
                    pos = packet[var]  # [x, y, z]
                    u, v = self.project_3d_to_2d(pos[0], pos[1], pos[2])
                    self.data_3d[var]["u"].append(u)
                    self.data_3d[var]["v"].append(v)

        # 3. Rafraîchissement DPG
        for var in self.all_2d_vars:
            dpg.set_value(f"series_{var}", [self.x_data, list(self.y_data[var])])

        # 4. Mise à jour des axes 3D (rotatifs)
        for var in self.vars_3d:
            # Recalculer les positions des axes en fonction de la rotation Z actuelle
            # Axe X: [0,0,0] -> [1, 0, 0]
            u_x_origin, v_x_origin = self.project_3d_to_2d(0, 0, 0)
            u_x_end, v_x_end = self.project_3d_to_2d(1, 0, 0)
            dpg.set_value(f"axis_x_{var}", [[u_x_origin, u_x_end], [v_x_origin, v_x_end]])

            # Axe Y: [0,0,0] -> [0, 1, 0]
            u_y_origin, v_y_origin = self.project_3d_to_2d(0, 0, 0)
            u_y_end, v_y_end = self.project_3d_to_2d(0, 1, 0)
            dpg.set_value(f"axis_y_{var}", [[u_y_origin, u_y_end], [v_y_origin, v_y_end]])

            # Axe Z: [0,0,0] -> [0, 0, 1]
            u_z_origin, v_z_origin = self.project_3d_to_2d(0, 0, 0)
            u_z_end, v_z_end = self.project_3d_to_2d(0, 0, 1)
            dpg.set_value(f"axis_z_{var}", [[u_z_origin, u_z_end], [v_z_origin, v_z_end]])

            # 5. Mise à jour des segments du bras robot
            if current_angles is not None:
                angles_array = np.array(current_angles)
                joints = self.get_arm_joints(angles_array)

                # Mettre à jour chaque segment (liaison entre deux articulations)
                for i in range(5):
                    p_start = joints[i]
                    p_end = joints[i + 1]

                    u_start, v_start = self.project_3d_to_2d(p_start[0], p_start[1], p_start[2])
                    u_end, v_end = self.project_3d_to_2d(p_end[0], p_end[1], p_end[2])

                    dpg.set_value(f"arm_segment_{i}_{var}", [[u_start, u_end], [v_start, v_end]])

            # 6. Mise à jour de la trajectoire 3D
            u_list = list(self.data_3d[var]["u"])
            v_list = list(self.data_3d[var]["v"])

            # Mise à jour de la ligne
            dpg.set_value(f"series_3d_line_{var}", [u_list, v_list])

            # Mise à jour du point de tête (dernier point reçu)
            if u_list:
                dpg.set_value(f"series_3d_head_{var}", [[u_list[-1]], [v_list[-1]]])

    def run(self):
        dpg.show_viewport()
        try:
            while dpg.is_dearpygui_running():
                self.update_data()
                dpg.render_dearpygui_frame()
        except KeyboardInterrupt:
            pass
        finally:
            dpg.destroy_context()
