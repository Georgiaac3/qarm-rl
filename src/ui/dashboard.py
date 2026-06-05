# ===============================
# File generated primarily by AI
# ===============================
import queue as pyqueue
from collections import deque

import dearpygui.dearpygui as dpg
import numpy as np

from src.core.config import settings
from src.core.dynamics_hold import L1, L2, L3, L4, L5


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

        self.x_data = deque(
            np.linspace(-self.max_seconds, 0, self.max_points).tolist(), maxlen=self.max_points
        )
        self.y_data = {
            var: deque([0.0] * self.max_points, maxlen=self.max_points) for var in self.all_2d_vars
        }
        self.x_axes_2d = {}

        # Pour la 3D : on stocke les positions brutes (x, y, z)
        self.data_3d = {
            var: {"x": deque(maxlen=500), "y": deque(maxlen=500), "z": deque(maxlen=500)}
            for var in self.vars_3d
        }

        # Rotation autour de l'axe Z (en radians)
        self.rotation_z = 0
        self.projection_mode = "plan_bras"
        self.view_angle0 = 0.0
        self.last_angles_measured = None

        # Longueurs du bras robot
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.L5 = L5

        # Bornes géométriques du robot pour fixer des limites de vue stables.
        self.arm_reach = self.L2 + self.L3 + self.L4 + self.L5
        self.z_min = -(self.L3 + self.L4 + self.L5)
        self.z_max = self.L1 + self.arm_reach
        self.u_min, self.u_max, self.v_min, self.v_max = self.get_projection_limits()
        self.floor_extent = self.arm_reach
        self.floor_step = 0.2
        self.floor_values = np.arange(
            -self.floor_extent, self.floor_extent + self.floor_step * 0.5, self.floor_step
        )
        self.floor_line_count = 2 * len(self.floor_values)

        self.setup_dpg()

    def project_3d_to_2d(self, x, y, z):
        """
        Projection 3D -> 2D selon le mode actif :
        u = projection horizontale
        v = projection verticale (incluant la hauteur Z)
        """
        if self.projection_mode == "plan_bras":
            # Vue orthographique dans le plan radial du bras (longueurs fideles).
            # On projette sur l'axe horizontal aligne avec l'azimut de base courant.
            cos_a0 = np.cos(self.view_angle0)
            sin_a0 = np.sin(self.view_angle0)
            u = x * cos_a0 + y * sin_a0
            v = z
            return u, v

        # Rotation autour de Z dans l'espace 3D (comportement historique).
        cos_rz, sin_rz = np.cos(self.rotation_z), np.sin(self.rotation_z)
        x_rot = x * cos_rz - y * sin_rz
        y_rot = x * sin_rz + y * cos_rz

        # Projection isométrique standard (orthographique):
        # u = (x - y) * cos(30°)
        # v = z + (x + y) * sin(30°)
        cos_30 = np.sqrt(3.0) / 2.0
        sin_30 = 0.5

        u = (x_rot - y_rot) * cos_30
        v = z + (x_rot + y_rot) * sin_30
        return u, v

    def get_projection_limits(self):
        """Calcule des limites 2D stables pour eviter tout effet de zoom dynamique."""
        if self.projection_mode == "plan_bras":
            margin = 0.1
            u_min, u_max = -self.arm_reach - margin, self.arm_reach + margin
            v_min, v_max = self.z_min - margin, self.z_max + margin
            return u_min, u_max, v_min, v_max

        # Avec la projection:
        # u = (x - y) * cos(30°), v = z + (x + y) * sin(30°)
        # et un disque horizontal de rayon arm_reach:
        # |x - y| et |x + y| <= arm_reach * sqrt(2)
        max_xy_combo = self.arm_reach * np.sqrt(2.0)
        cos_30 = np.sqrt(3.0) / 2.0
        sin_30 = 0.5

        max_u = max_xy_combo * cos_30
        v_xy_offset = max_xy_combo * sin_30

        margin = 0.1
        u_min, u_max = -max_u - margin, max_u + margin
        v_min, v_max = self.z_min - v_xy_offset - margin, self.z_max + v_xy_offset + margin
        return u_min, u_max, v_min, v_max

    def get_equal_axis_limits(self):
        """Retourne des bornes X/Y avec la meme amplitude en unites de donnees."""
        u_center = 0.5 * (self.u_min + self.u_max)
        v_center = 0.5 * (self.v_min + self.v_max)
        span = max(self.u_max - self.u_min, self.v_max - self.v_min)
        half = 0.5 * span
        return u_center - half, u_center + half, v_center - half, v_center + half

    def get_isotropic_limits(self, var):
        """Ajuste les limites pour avoir la meme echelle en pixels sur X et Y."""
        x_min, x_max, y_min, y_max = self.get_equal_axis_limits()
        width, height = dpg.get_item_rect_size(f"plot_3d_{var}")
        if width <= 0 or height <= 0:
            return x_min, x_max, y_min, y_max

        x_center = 0.5 * (x_min + x_max)
        y_center = 0.5 * (y_min + y_max)
        base_span = max(x_max - x_min, y_max - y_min)

        if width >= height:
            x_span = base_span * (width / height)
            y_span = base_span
        else:
            x_span = base_span
            y_span = base_span * (height / width)

        return (
            x_center - 0.5 * x_span,
            x_center + 0.5 * x_span,
            y_center - 0.5 * y_span,
            y_center + 0.5 * y_span,
        )

    def set_projection_mode(self, _sender, app_data):
        self.projection_mode = app_data
        self.u_min, self.u_max, self.v_min, self.v_max = self.get_projection_limits()

        # Eviter de melanger des points historiques issus de deux projections differentes.
        for var in self.vars_3d:
            self.data_3d[var]["x"].clear()
            self.data_3d[var]["y"].clear()
            self.data_3d[var]["z"].clear()

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

    def get_arm_joints_planar(self, angles):
        """Cinématique 2D directe dans le plan du bras (u, z), sans projection."""
        angle1, angle2 = angles[1], angles[2]

        p0 = np.array([0.0, 0.0])
        p1 = p0 + np.array([0.0, self.L1])
        p2 = p1 + np.array([self.L2 * np.sin(angle1), self.L2 * np.cos(angle1)])
        p3 = p2 + np.array([self.L3 * np.cos(angle1), -self.L3 * np.sin(angle1)])
        p4 = p3 + np.array([self.L4 * np.cos(angle1 + angle2), -self.L4 * np.sin(angle1 + angle2)])
        p5 = p4 + np.array([self.L5 * np.cos(angle1 + angle2), -self.L5 * np.sin(angle1 + angle2)])
        return [p0, p1, p2, p3, p4, p5]

    def get_floor_lines(self):
        """Construit les lignes du plancher XY (z=0) dans le repere projete."""
        if self.projection_mode == "plan_bras":
            # En vue planaire, le plan XY se confond avec l'axe horizontal z=0.
            lines = [([-self.floor_extent, self.floor_extent], [0.0, 0.0])]
            # Garder un nombre de series constant entre les modes pour eviter les tags manquants.
            while len(lines) < self.floor_line_count:
                lines.append(([], []))
            return lines

        lines = []

        # Lignes paralleles a X (y constant)
        for y in self.floor_values:
            u1, v1 = self.project_3d_to_2d(-self.floor_extent, y, 0.0)
            u2, v2 = self.project_3d_to_2d(self.floor_extent, y, 0.0)
            lines.append(([u1, u2], [v1, v2]))

        # Lignes paralleles a Y (x constant)
        for x in self.floor_values:
            u1, v1 = self.project_3d_to_2d(x, -self.floor_extent, 0.0)
            u2, v2 = self.project_3d_to_2d(x, self.floor_extent, 0.0)
            lines.append(([u1, u2], [v1, v2]))

        return lines

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

        # Theme dédié pour la grille du plancher (gris discret)
        with dpg.theme() as floor_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (150, 150, 150, 120), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 1.0, category=dpg.mvThemeCat_Plots
                )

        # Theme pour TCP_Trajectoire (rouge/orange)
        with dpg.theme() as tcp_scatter_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill, (255, 127, 0, 255), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline, (255, 127, 0, 255), category=dpg.mvThemeCat_Plots
                )

        # Theme pour Wanted_TCP_Trajectoire (bleu)
        with dpg.theme() as wanted_scatter_theme:
            with dpg.theme_component(dpg.mvScatterSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerFill, (0, 149, 255, 255), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_MarkerOutline, (0, 149, 255, 255), category=dpg.mvThemeCat_Plots
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
                            self.x_axes_2d[group_name] = x_axis
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
                # Skip Wanted_TCP_Trajectoire as it will be plotted with TCP_trajectoire
                vars_to_plot = [v for v in self.vars_3d if v != "Wanted_TCP_Trajectoire"]

                for var in vars_to_plot:
                    floor_lines = self.get_floor_lines()
                    # On agrandit la fenêtre (height=500)
                    with dpg.plot(
                        label=f"Vue 3D simulée: {var}",
                        height=500,
                        width=-1,
                        equal_aspects=True,
                        tag=f"plot_3d_{var}",
                    ):
                        dpg.add_plot_legend()
                        # On retire les labels d'axes car ils sont projetés
                        x_ax_3d = dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label="Plan Horizontal",
                            no_tick_labels=True,
                            tag=f"x_axis_3d_{var}",
                        )
                        y_ax_3d = dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label="Hauteur Z",
                            no_tick_labels=True,
                            tag=f"y_axis_3d_{var}",
                        )

                        # Grille du plancher XY (z=0)
                        for i in range(self.floor_line_count):
                            u_line, v_line = floor_lines[i]
                            floor_tag = f"floor_line_{i}_{var}"
                            dpg.add_line_series(
                                u_line,
                                v_line,
                                label="Plancher" if i == 0 else "",
                                tag=floor_tag,
                                parent=y_ax_3d,
                            )
                            dpg.bind_item_theme(floor_tag, floor_theme)

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

                        # Trajectoire actuelle (ligne)
                        dpg.add_line_series(
                            [], [], label=f"{var}", tag=f"series_3d_line_{var}", parent=y_ax_3d
                        )
                        # Position actuelle (gros point)
                        dpg.add_scatter_series(
                            [],
                            [],
                            label="Position TCP",
                            tag=f"series_3d_head_{var}",
                            parent=y_ax_3d,
                        )
                        dpg.bind_item_theme(f"series_3d_head_{var}", tcp_scatter_theme)

                        # Si "Wanted_TCP_Trajectoire" existe et var est "TCP_Trajectoire", ajouter la trajectoire voulue
                        if var == "TCP_Trajectoire" and "Wanted_TCP_Trajectoire" in self.vars_3d:
                            dpg.add_line_series(
                                [],
                                [],
                                label="Wanted_TCP_Trajectoire",
                                tag="series_3d_line_Wanted_TCP_Trajectoire",
                                parent=y_ax_3d,
                            )
                            dpg.add_scatter_series(
                                [],
                                [],
                                label="Position Wanted TCP",
                                tag="series_3d_head_Wanted_TCP_Trajectoire",
                                parent=y_ax_3d,
                            )
                            dpg.bind_item_theme(
                                "series_3d_head_Wanted_TCP_Trajectoire", wanted_scatter_theme
                            )

                        # Fixer les limites pour éviter la déformation visuelle et le "saut".
                        x_min, x_max, y_min, y_max = self.get_isotropic_limits(var)
                        dpg.set_axis_limits(x_ax_3d, x_min, x_max)
                        dpg.set_axis_limits(y_ax_3d, y_min, y_max)

            # --- CONTRÔLES 3D ---
            with dpg.collapsing_header(label="Contrôles 3D", default_open=True):
                dpg.add_radio_button(
                    items=["plan_bras", "isometrique"],
                    default_value=self.projection_mode,
                    label="Mode de projection",
                    horizontal=True,
                    callback=self.set_projection_mode,
                )
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

        while True:
            try:
                packet = self.queue.get_nowait()
            except pyqueue.Empty:
                break

            packet_time = packet.get("t_s")

            # Récupérer les angles du robot
            if "Angles Articulations mesurés (rad)" in packet:
                current_angles = packet["Angles Articulations mesurés (rad)"]
                if len(current_angles) >= 3:
                    self.last_angles_measured = np.array(current_angles, dtype=float)
                    # En mode plan du bras, on aligne la vue avec l'azimut de la base.
                    self.view_angle0 = float(self.last_angles_measured[0])

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
                    self.data_3d[var]["x"].append(float(pos[0]))
                    self.data_3d[var]["y"].append(float(pos[1]))
                    self.data_3d[var]["z"].append(float(pos[2]))

            if packet_time is not None:
                self.x_data.append(float(packet_time))

        # 3. Rafraîchissement DPG
        for var in self.all_2d_vars:
            dpg.set_value(f"series_{var}", [list(self.x_data), list(self.y_data[var])])

        current_time = float(self.x_data[-1]) if len(self.x_data) > 0 else 0.0
        x_min = current_time - self.max_seconds
        x_max = current_time
        for group_name in self.vars_2d_groups:
            x_axis = self.x_axes_2d.get(group_name)
            if x_axis is not None:
                dpg.set_axis_limits(x_axis, x_min, x_max)

        # 4. Mise à jour des axes 3D (rotatifs)
        vars_to_update = [v for v in self.vars_3d if v != "Wanted_TCP_Trajectoire"]
        for var in vars_to_update:
            x_min, x_max, y_min, y_max = self.get_isotropic_limits(var)
            dpg.set_axis_limits(f"x_axis_3d_{var}", x_min, x_max)
            dpg.set_axis_limits(f"y_axis_3d_{var}", y_min, y_max)

            floor_lines = self.get_floor_lines()
            for i, (u_line, v_line) in enumerate(floor_lines):
                dpg.set_value(f"floor_line_{i}_{var}", [u_line, v_line])

            # Recalculer les positions des axes en fonction de la rotation Z actuelle
            # Axe X: [0,0,0] -> [0.1, 0, 0]
            u_x_origin, v_x_origin = self.project_3d_to_2d(0, 0, 0)
            u_x_end, v_x_end = self.project_3d_to_2d(0.1, 0, 0)
            dpg.set_value(f"axis_x_{var}", [[u_x_origin, u_x_end], [v_x_origin, v_x_end]])

            # Axe Y: [0,0,0] -> [0, 0.1, 0]
            u_y_origin, v_y_origin = self.project_3d_to_2d(0, 0, 0)
            u_y_end, v_y_end = self.project_3d_to_2d(0, 0.1, 0)
            dpg.set_value(f"axis_y_{var}", [[u_y_origin, u_y_end], [v_y_origin, v_y_end]])

            # Axe Z: [0,0,0] -> [0, 0, 0.1]
            u_z_origin, v_z_origin = self.project_3d_to_2d(0, 0, 0)
            u_z_end, v_z_end = self.project_3d_to_2d(0, 0, 0.1)
            dpg.set_value(f"axis_z_{var}", [[u_z_origin, u_z_end], [v_z_origin, v_z_end]])

            # 5. Mise à jour des segments du bras robot
            if self.last_angles_measured is not None:
                angles_array = self.last_angles_measured
                if self.projection_mode == "plan_bras":
                    planar_joints = self.get_arm_joints_planar(angles_array)
                    for i in range(5):
                        p_start = planar_joints[i]
                        p_end = planar_joints[i + 1]

                        dpg.set_value(
                            f"arm_segment_{i}_{var}",
                            [[p_start[0], p_end[0]], [p_start[1], p_end[1]]],
                        )
                else:
                    joints = self.get_arm_joints(angles_array)

                    # Mettre à jour chaque segment (liaison entre deux articulations)
                    for i in range(5):
                        p_start = joints[i]
                        p_end = joints[i + 1]

                        u_start, v_start = self.project_3d_to_2d(p_start[0], p_start[1], p_start[2])
                        u_end, v_end = self.project_3d_to_2d(p_end[0], p_end[1], p_end[2])

                        dpg.set_value(
                            f"arm_segment_{i}_{var}", [[u_start, u_end], [v_start, v_end]]
                        )

            # 6. Mise à jour de la trajectoire 3D
            x_list = list(self.data_3d[var]["x"])
            y_list = list(self.data_3d[var]["y"])
            z_list = list(self.data_3d[var]["z"])

            # Reprojeter tout l'historique avec la vue courante.
            u_list = []
            v_list = []
            for x, y, z in zip(x_list, y_list, z_list):
                u, v = self.project_3d_to_2d(x, y, z)
                u_list.append(u)
                v_list.append(v)

            # Mise à jour de la ligne
            dpg.set_value(f"series_3d_line_{var}", [u_list, v_list])

            # Mise à jour du point de tête (dernier point reçu)
            if u_list:
                dpg.set_value(f"series_3d_head_{var}", [[u_list[-1]], [v_list[-1]]])

            # Si "Wanted_TCP_Trajectoire" existe et var est "TCP_Trajectoire", mettre à jour cette trajectoire aussi
            if var == "TCP_Trajectoire" and "Wanted_TCP_Trajectoire" in self.vars_3d:
                wanted_x_list = list(self.data_3d["Wanted_TCP_Trajectoire"]["x"])
                wanted_y_list = list(self.data_3d["Wanted_TCP_Trajectoire"]["y"])
                wanted_z_list = list(self.data_3d["Wanted_TCP_Trajectoire"]["z"])

                # Reprojeter tout l'historique avec la vue courante.
                wanted_u_list = []
                wanted_v_list = []
                for x, y, z in zip(wanted_x_list, wanted_y_list, wanted_z_list):
                    u, v = self.project_3d_to_2d(x, y, z)
                    wanted_u_list.append(u)
                    wanted_v_list.append(v)

                # Mise à jour de la ligne
                dpg.set_value(
                    "series_3d_line_Wanted_TCP_Trajectoire", [wanted_u_list, wanted_v_list]
                )

                # Mise à jour du point de tête (dernier point reçu)
                if wanted_u_list:
                    dpg.set_value(
                        "series_3d_head_Wanted_TCP_Trajectoire",
                        [[wanted_u_list[-1]], [wanted_v_list[-1]]],
                    )

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
