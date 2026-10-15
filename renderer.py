"""
Renderer Module — Engineering Motion Capture Animation Rig
===========================================================
Handles high-end Pygame rendering for the motion capture studio.
Features:
  • Octahedral diamond 3D animation bones (Maya/Blender armature style)
  • Spine vertebrae column, ribcage arches, and pelvis harness
  • Mechanical joint bearings with concentric calibration rings and tick marks
  • Real-time engineering telemetry (joint flexion angle readouts for knees & elbows)
  • Pelvis center-of-mass (COM) 3-axis RGB gizmo
  • Perspective floor grid with grounding shadow & contact rings
  • Height calibration ruler with metric scale
  • Camera picture-in-picture (PiP) feed with HUD border
  • Glow bloom, scanline overlay, and technical telemetry HUD
"""

import math
import random
import cv2
import pygame
import numpy as np

from pose_processor import (
    SKELETON_BONES, BONE_GROUPS, MAJOR_JOINTS, MINOR_JOINTS, Landmark
)


# ── Engineering Studio Color Palette ─────────────────────────────────────────

COLORS = {
    "bg":            (10, 12, 22),
    "grid":          (24, 30, 52),
    "grid_accent":   (40, 55, 95),
    "grid_major":    (55, 80, 140),
    "torso":         (0, 240, 255),       # Cyan
    "spine":         (0, 210, 255),       # Deep Cyan
    "left_arm":      (175, 75, 255),      # Neon Violet
    "right_arm":     (255, 65, 175),      # Hot Magenta
    "left_leg":      (50, 255, 170),      # Cyber Mint Green
    "right_leg":     (255, 205, 45),      # Tech Amber / Gold
    "head":          (255, 115, 65),      # Sunset Coral
    "joint_core":    (255, 255, 255),     # White center pin
    "joint_glow":    (0, 215, 255),       # Cyan glow
    "joint_ring":    (120, 170, 220),     # Metallic steel ring
    "telemetry":     (0, 245, 255),       # HUD telemetry cyan
    "telemetry_bg":  (15, 22, 38, 200),   # Pill background
    "particle":      (110, 210, 255),
    "trail":         (0, 190, 255),
    "hud_text":      (0, 230, 255),
    "hud_dim":       (80, 105, 145),
    "hud_active":    (50, 255, 140),      # Bright Green for Active Tracking
    "hud_standby":   (255, 190, 40),      # Amber for Standby Pose
    "recording":     (255, 55, 55),
    "playback":      (60, 255, 120),
    "gizmo_x":       (255, 60, 60),       # Red = X axis
    "gizmo_y":       (60, 255, 80),       # Green = Y axis
    "gizmo_z":       (60, 140, 255),      # Blue = Z axis
}

# Map bone groups to their colors
BONE_COLOR_MAP = {}
for group, bones in BONE_GROUPS.items():
    for bone in bones:
        BONE_COLOR_MAP[bone] = COLORS.get(group, COLORS["torso"])


class Particle:
    """A single glowing particle for trail effects."""

    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size']

    def __init__(self, x, y, color, size=2.5):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.3, 1.8)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.4, 1.2)
        self.max_life = self.life
        self.color = color
        self.size = size

    def update(self, dt):
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60
        self.vy += 0.02 * dt * 60
        self.life -= dt
        return self.life > 0

    def draw(self, surface):
        alpha = max(0, self.life / self.max_life)
        r = int(self.color[0] * alpha)
        g = int(self.color[1] * alpha)
        b = int(self.color[2] * alpha)
        radius = max(1, int(self.size * alpha))
        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)), radius)


class TrailPoint:
    """A point in a motion trail."""

    __slots__ = ['x', 'y', 'life', 'color']

    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.life = 1.0
        self.color = color

    def update(self, dt):
        self.life -= dt * 1.8
        return self.life > 0


class Renderer:
    """
    Engineering-grade motion capture visualization renderer.
    Renders 3D animation octahedral bone armatures, mechanical joints,
    telemetry angles, height rulers, and HUD overlays.
    """

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.particles = []
        self.trails = {i: [] for i in range(33)}
        self.time = 0.0
        self.frame_count = 0

        # Layer surfaces
        self.glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.bone_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.ui_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Typography
        try:
            self.font_large = pygame.font.SysFont("Menlo", 24, bold=True)
            self.font_medium = pygame.font.SysFont("Menlo", 16, bold=True)
            self.font_small = pygame.font.SysFont("Menlo", 13)
            self.font_tiny = pygame.font.SysFont("Menlo", 11)
        except Exception:
            self.font_large = pygame.font.Font(None, 28)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_small = pygame.font.Font(None, 16)
            self.font_tiny = pygame.font.Font(None, 14)

        # Scanline overlay
        self.scanline_surface = self._create_scanline_surface()

    def _create_scanline_surface(self):
        """Create a subtle technical CRT scanline overlay."""
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(0, self.height, 3):
            pygame.draw.line(surf, (0, 0, 0, 18), (0, y), (self.width, y))
        return surf

    def render_frame(self, screen, landmarks, processor, mode="live",
                     fps=0, recording=False, frame_num=0, total_frames=0,
                     cam_frame=None, is_tracking=False):
        """
        Render a full motion capture studio frame.
        """
        dt = 1.0 / max(fps, 30)
        self.time += dt
        self.frame_count += 1

        # Clear base screen
        screen.fill(COLORS["bg"])

        # Perspective studio floor grid
        self._draw_grid(screen)

        # Height calibration metric ruler on left edge
        self._draw_calibration_ruler(screen)

        # Ambient floating atmosphere particles
        self._update_ambient_particles(screen, dt)

        # Clear overlay surfaces
        self.glow_surface.fill((0, 0, 0, 0))
        self.bone_surface.fill((0, 0, 0, 0))
        self.ui_surface.fill((0, 0, 0, 0))

        if landmarks is not None:
            # Floor contact shadow under feet
            self._draw_floor_shadow(screen, landmarks)

            # Update & draw motion trails
            self._update_trails(landmarks, processor, dt)
            self._draw_trails(screen)

            # Draw octahedral animation rig bones & spine
            self._draw_bones(screen, landmarks, processor)

            # Draw mechanical bearing joints
            self._draw_joints(screen, landmarks, processor)

            # Draw engineering joint angle telemetry (knees, elbows)
            self._draw_engineering_telemetry(screen, landmarks)

            # Draw Pelvis Center-of-Mass (COM) 3-axis RGB gizmo
            self._draw_root_gizmo(screen, landmarks)

            # Draw cranial rig visor & sensors
            self._draw_head(screen, landmarks)

            # Spawn velocity particles at fast joints
            self._spawn_particles(landmarks, processor)

        # Blit bone surface (translucent octahedral facets)
        screen.blit(self.bone_surface, (0, 0))

        # Update and render particles
        self._update_and_draw_particles(screen, dt)

        # Blit glow surface (additive bloom)
        screen.blit(self.glow_surface, (0, 0), special_flags=pygame.BLEND_ADD)

        # Blit telemetry UI overlay
        screen.blit(self.ui_surface, (0, 0))

        # Scanline overlay
        screen.blit(self.scanline_surface, (0, 0))

        # Camera picture-in-picture
        if cam_frame is not None:
            self._draw_pip(screen, cam_frame, is_tracking)

        # HUD telemetry overlay
        self._draw_hud(screen, mode, fps, recording, frame_num, total_frames,
                       is_tracking, landmarks is not None)

        # Corner vignette
        self._draw_vignette(screen)

    def _draw_grid(self, screen):
        """Draw an engineering perspective floor and studio grid."""
        spacing = 50
        cx, cy = self.width // 2, self.height // 2

        # Vertical grid lines
        for x in range(0, self.width + spacing, spacing):
            dist_from_center = abs(x - cx) / cx
            alpha = max(0.2, 0.45 - dist_from_center * 0.25)
            col = tuple(int(c * alpha) for c in COLORS["grid_accent"])
            pygame.draw.line(screen, col, (x, 0), (x, self.height), 1)

        # Horizontal grid lines
        for y in range(0, self.height + spacing, spacing):
            dist_from_center = abs(y - cy) / cy
            alpha = max(0.2, 0.45 - dist_from_center * 0.25)
            col = tuple(int(c * alpha) for c in COLORS["grid_accent"])
            pygame.draw.line(screen, col, (0, y), (self.width, y), 1)

        # Primary Studio Axis Lines
        pygame.draw.line(screen, COLORS["grid_major"], (cx, 0), (cx, self.height), 2)
        pygame.draw.line(screen, COLORS["grid_major"], (0, cy), (self.width, cy), 1)

        # Floor reference line (around foot level: Y = 0.86 * height)
        floor_y = int(self.height * 0.86)
        pygame.draw.line(screen, (45, 75, 130), (0, floor_y), (self.width, floor_y), 2)

    def _draw_calibration_ruler(self, screen):
        """Draw vertical engineering height metric scale on the left margin."""
        ruler_x = 24
        floor_y = int(self.height * 0.86)
        top_y = int(self.height * 0.12)
        total_h = floor_y - top_y

        # Ruler vertical line
        pygame.draw.line(screen, COLORS["hud_dim"], (ruler_x, top_y), (ruler_x, floor_y), 1)

        # Scale marks from 0.0m up to 2.0m
        marks = [
            (0.00, "0.0m [GND]"),
            (0.50, "0.5m"),
            (1.00, "1.0m [PELVIS]"),
            (1.50, "1.5m [CHEST]"),
            (1.80, "1.8m [HEAD]"),
            (2.00, "2.0m"),
        ]

        for meters, label in marks:
            frac = meters / 2.0
            my = int(floor_y - frac * total_h)
            tick_w = 8 if meters in (0.0, 1.0, 1.8) else 4
            pygame.draw.line(screen, COLORS["hud_dim"], (ruler_x - tick_w, my), (ruler_x + tick_w, my), 1)
            txt = self.font_tiny.render(label, True, COLORS["hud_dim"])
            screen.blit(txt, (ruler_x + 12, my - 6))

    def _draw_floor_shadow(self, screen, landmarks):
        """Draw grounded floor contact shadows and pressure rings under the feet."""
        for foot_idx in (27, 28):  # Left & right ankles/feet
            lm = landmarks[foot_idx]
            if lm.visibility < 0.3:
                continue

            pos = lm.to_screen(self.width, self.height)
            floor_y = int(self.height * 0.86)

            # Shadow ellipse on floor
            dist = max(1, abs(pos[1] - floor_y))
            scale = max(0.2, 1.0 - (dist / 120.0))
            shadow_w = int(45 * scale)
            shadow_h = int(12 * scale)

            shadow_rect = pygame.Rect(pos[0] - shadow_w, floor_y - shadow_h // 2, shadow_w * 2, shadow_h)
            pygame.draw.ellipse(self.glow_surface, (0, 200, 255, int(40 * scale)), shadow_rect)
            pygame.draw.ellipse(screen, (5, 20, 35), shadow_rect)

            # Ground contact ring
            if dist < 20:
                pygame.draw.ellipse(screen, COLORS["left_leg" if foot_idx == 27 else "right_leg"],
                                    shadow_rect, 1)

    def _draw_bones(self, screen, landmarks, processor):
        """Draw 3D octahedral animation diamond armatures and spine structure."""
        # ── 1. Spine & Thoracic Ribcage (Core Armature) ──
        self._draw_spine_and_ribcage(screen, landmarks, processor)

        # ── 2. Primary Skeletal Bones as Octahedral Rig Armatures ──
        for bone in SKELETON_BONES:
            idx_a, idx_b = bone
            la = landmarks[idx_a]
            lb = landmarks[idx_b]

            if la.visibility < 0.3 or lb.visibility < 0.3:
                continue

            pa = la.to_screen(self.width, self.height)
            pb = lb.to_screen(self.width, self.height)

            color = BONE_COLOR_MAP.get(bone, COLORS["torso"])

            vel_a = processor.get_joint_velocity(idx_a)
            vel_b = processor.get_joint_velocity(idx_b)
            avg_vel = (vel_a + vel_b) / 2.0

            # Determine bone type for octahedral width
            if (idx_a, idx_b) in [(11, 13), (12, 14)]:
                # Upper arms
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=9, vel=avg_vel)
            elif (idx_a, idx_b) in [(13, 15), (14, 16)]:
                # Forearms
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=7, vel=avg_vel)
            elif (idx_a, idx_b) in [(23, 25), (24, 26)]:
                # Thighs (heavy structural bone)
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=12, vel=avg_vel)
            elif (idx_a, idx_b) in [(25, 27), (26, 28)]:
                # Shins
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=9, vel=avg_vel)
            elif (idx_a, idx_b) in [(11, 23), (12, 24)]:
                # Lateral torso pylons
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=10, vel=avg_vel)
            elif (idx_a, idx_b) == (11, 12):
                # Shoulder beam
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=8, vel=avg_vel)
            elif (idx_a, idx_b) == (23, 24):
                # Pelvis beam
                self._draw_octahedral_bone(screen, pa, pb, color, max_w=8, vel=avg_vel)
            else:
                # Sleek linkages for fingers, feet, and neck
                self._draw_linear_linkage(screen, pa, pb, color, vel=avg_vel)

    def _draw_octahedral_bone(self, screen, pa, pb, color, max_w=10, vel=0.0):
        """
        Draw a 3D animation diamond/octahedral bone armature (Blender/Maya style).
        Widest at ~28% from the parent joint, tapering to both ends with two faceted halves.
        """
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 5:
            return

        # Unit direction and normal
        ux = dx / length
        uy = dy / length
        nx = -uy
        ny = ux

        # Diamond widest point position
        t = 0.28
        mid_x = pa[0] + dx * t
        mid_y = pa[1] + dy * t

        w = max_w * (1.0 + min(1.0, vel * 30.0))

        v0 = pa
        v1 = (int(mid_x + nx * w), int(mid_y + ny * w))
        v2 = pb
        v3 = (int(mid_x - nx * w), int(mid_y - ny * w))

        # Facet 1 (Upper Highlight)
        col_high = tuple(min(255, int(c * 1.15) + 30) for c in color)
        pygame.draw.polygon(self.bone_surface, (*col_high, 95), [v0, v1, v2])

        # Facet 2 (Lower Shadow)
        col_low = tuple(max(0, int(c * 0.75)) for c in color)
        pygame.draw.polygon(self.bone_surface, (*col_low, 75), [v0, v2, v3])

        # Outer diamond wireframe edges
        pygame.draw.polygon(screen, color, [v0, v1, v2, v3], 1)

        # Center longitudinal structural axis
        pygame.draw.line(screen, (230, 250, 255), (int(v0[0]), int(v0[1])), (int(v2[0]), int(v2[1])), 1)

        # Outer glow bloom
        glow_c = (*tuple(max(0, min(255, int(c * 0.4))) for c in color), 45)
        pygame.draw.polygon(self.glow_surface, glow_c, [v0, v1, v2, v3])

    def _draw_linear_linkage(self, screen, pa, pb, color, vel=0.0):
        """Draw sleek glowing engineering linkage for hands, feet, and neck."""
        thickness = max(1, int(2 + vel * 30))
        # Core
        pygame.draw.line(screen, color, pa, pb, thickness)
        # Inner bright core
        bright = tuple(min(255, c + 70) for c in color)
        pygame.draw.line(screen, bright, pa, pb, 1)
        # Glow
        pygame.draw.line(self.glow_surface, (*color, 40), pa, pb, thickness + 4)

    def _draw_spine_and_ribcage(self, screen, landmarks, processor):
        """Draw segmented central spine vertebrae column and thoracic ribcage arches."""
        p_neck_x = (landmarks[11].x + landmarks[12].x) * 0.5
        p_neck_y = (landmarks[11].y + landmarks[12].y) * 0.5
        p_pelv_x = (landmarks[23].x + landmarks[24].x) * 0.5
        p_pelv_y = (landmarks[23].y + landmarks[24].y) * 0.5

        neck_pt = Landmark(p_neck_x, p_neck_y, 0, 1.0).to_screen(self.width, self.height)
        pelv_pt = Landmark(p_pelv_x, p_pelv_y, 0, 1.0).to_screen(self.width, self.height)

        # Main spine axis line
        pygame.draw.line(screen, COLORS["spine"], neck_pt, pelv_pt, 2)
        pygame.draw.line(self.glow_surface, (*COLORS["spine"], 60), neck_pt, pelv_pt, 6)

        # Vertebrae plates (4 articulated segments along the spine)
        num_vert = 4
        for i in range(1, num_vert + 1):
            t = i / (num_vert + 1)
            vx = neck_pt[0] + (pelv_pt[0] - neck_pt[0]) * t
            vy = neck_pt[1] + (pelv_pt[1] - neck_pt[1]) * t
            v_pt = (int(vx), int(vy))

            # Vertebrae disc
            disc_w = 12
            pygame.draw.line(screen, COLORS["joint_ring"], (v_pt[0] - disc_w, v_pt[1]), (v_pt[0] + disc_w, v_pt[1]), 2)
            pygame.draw.circle(screen, COLORS["joint_core"], v_pt, 3)

            # Ribcage horizontal arches at upper spine
            if i in (1, 2):
                rib_w = 26 if i == 1 else 20
                arch_pts = [
                    (int(vx - rib_w), int(vy + 4)),
                    v_pt,
                    (int(vx + rib_w), int(vy + 4)),
                ]
                pygame.draw.lines(screen, COLORS["grid_accent"], False, arch_pts, 1)

    def _draw_joints(self, screen, landmarks, processor):
        """Draw mechanical bearing pivot rings with rotary ticks and glow."""
        for i in range(33):
            lm = landmarks[i]
            if lm.visibility < 0.3:
                continue

            pos = lm.to_screen(self.width, self.height)
            vel = processor.get_joint_velocity(i)

            # Determine base radius based on anatomical hierarchy
            if i in MAJOR_JOINTS:
                base_r = 8
            elif i in MINOR_JOINTS:
                base_r = 5
            else:
                base_r = 6

            pulse = 1.0 + min(0.8, vel * 40.0)
            radius = int(base_r * pulse)

            # Determine joint color
            joint_col = COLORS["joint_glow"]
            for group, bones in BONE_GROUPS.items():
                for a, b in bones:
                    if i == a or i == b:
                        joint_col = COLORS.get(group, COLORS["joint_glow"])
                        break

            # 1. Outer Concentric Calibration Bearing Ring
            ring_r = radius + 4
            pygame.draw.circle(screen, COLORS["joint_ring"], pos, ring_r, 1)

            # Cardinal compass tick marks on major joints (0°, 90°, 180°, 270°)
            if i in MAJOR_JOINTS:
                for angle_deg in (0, 90, 180, 270):
                    rad = math.radians(angle_deg)
                    t1 = (int(pos[0] + math.cos(rad) * (ring_r - 2)),
                          int(pos[1] + math.sin(rad) * (ring_r - 2)))
                    t2 = (int(pos[0] + math.cos(rad) * (ring_r + 3)),
                          int(pos[1] + math.sin(rad) * (ring_r + 3)))
                    pygame.draw.line(screen, COLORS["hud_text"], t1, t2, 1)

            # 2. Glow Bloom Rings
            for r_mult, alpha in [(2.4, 25), (1.6, 60)]:
                gr = int(radius * r_mult)
                pygame.draw.circle(self.glow_surface, (*joint_col, alpha), pos, gr)

            # 3. Main Pivot Core
            pygame.draw.circle(screen, joint_col, pos, radius)

            # 4. White Center Pin
            inner_r = max(2, radius // 2)
            pygame.draw.circle(screen, COLORS["joint_core"], pos, inner_r)

    def _draw_engineering_telemetry(self, screen, landmarks):
        """
        Draw engineering articulation angle readouts and flexion arcs
        for knees, elbows, and torso alignment.
        """
        # Key joints to display live flexion angles
        angle_joints = [
            ("L-ELB", 11, 13, 15, (-48, -10)),   # Left Elbow (shoulder, elbow, wrist)
            ("R-ELB", 12, 14, 16, (22, -10)),    # Right Elbow
            ("L-KNEE", 23, 25, 27, (-58, 0)),    # Left Knee (hip, knee, ankle)
            ("R-KNEE", 24, 26, 28, (24, 0)),     # Right Knee
        ]

        for label, ja, jb, jc, offset in angle_joints:
            la = landmarks[ja]
            lb = landmarks[jb]
            lc = landmarks[jc]

            if la.visibility < 0.3 or lb.visibility < 0.3 or lc.visibility < 0.3:
                continue

            pb = lb.to_screen(self.width, self.height)

            # Compute 2D joint angle in degrees
            ba = np.array([la.x - lb.x, la.y - lb.y])
            bc = np.array([lc.x - lb.x, lc.y - lb.y])
            norm_a = np.linalg.norm(ba)
            norm_c = np.linalg.norm(bc)

            if norm_a < 1e-4 or norm_c < 1e-4:
                continue

            cos_ang = np.dot(ba, bc) / (norm_a * norm_c)
            angle_deg = int(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

            # Draw subtle flexion arc at joint B
            arc_r = 16
            pygame.draw.circle(screen, (70, 100, 150), pb, arc_r, 1)

            # Engineering Callout Tag
            tag_text = f"{label} {angle_deg}°"
            txt_surf = self.font_tiny.render(tag_text, True, COLORS["telemetry"])
            tw = txt_surf.get_width()
            th = txt_surf.get_height()

            tag_x = pb[0] + offset[0]
            tag_y = pb[1] + offset[1]

            # Tag pill background
            bg_rect = pygame.Rect(tag_x - 3, tag_y - 2, tw + 6, th + 4)
            pygame.draw.rect(self.ui_surface, (12, 18, 32, 210), bg_rect)
            pygame.draw.rect(self.ui_surface, COLORS["grid_accent"], bg_rect, 1)
            self.ui_surface.blit(txt_surf, (tag_x, tag_y))

            # Leader line connecting joint to tag
            lead_end = (tag_x if offset[0] > 0 else tag_x + tw, tag_y + th // 2)
            pygame.draw.line(self.ui_surface, COLORS["grid_accent"], pb, lead_end, 1)

    def _draw_root_gizmo(self, screen, landmarks):
        """Draw a 3-axis mini RGB gimbal gizmo at the pelvis center-of-mass (COM)."""
        l23 = landmarks[23]
        l24 = landmarks[24]
        if l23.visibility < 0.3 or l24.visibility < 0.3:
            return

        cx = (l23.x + l24.x) * 0.5
        cy = (l23.y + l24.y) * 0.5
        root_pt = Landmark(cx, cy, 0, 1.0).to_screen(self.width, self.height)

        axis_len = 22

        # Red = +X Axis (lateral right)
        pygame.draw.line(screen, COLORS["gizmo_x"], root_pt, (root_pt[0] + axis_len, root_pt[1]), 2)
        pygame.draw.circle(screen, COLORS["gizmo_x"], (root_pt[0] + axis_len, root_pt[1]), 2)

        # Green = +Y Axis (vertical down)
        pygame.draw.line(screen, COLORS["gizmo_y"], root_pt, (root_pt[0], root_pt[1] + axis_len), 2)
        pygame.draw.circle(screen, COLORS["gizmo_y"], (root_pt[0], root_pt[1] + axis_len), 2)

        # Blue = Z Axis (central depth circle)
        pygame.draw.circle(screen, COLORS["gizmo_z"], root_pt, 4, 1)
        pygame.draw.circle(screen, COLORS["joint_core"], root_pt, 2)

        # Label: ROOT COM
        root_lbl = self.font_tiny.render("ROOT COM", True, COLORS["hud_dim"])
        screen.blit(root_lbl, (root_pt[0] - 25, root_pt[1] - 16))

    def _draw_head(self, screen, landmarks):
        """Draw cranial animation rig visor with optical sensor reticle."""
        nose = landmarks[0]
        if nose.visibility < 0.3:
            return

        pos = nose.to_screen(self.width, self.height)
        head_radius = 28

        # Neck link from nose down to mid-shoulder
        p_neck_x = (landmarks[11].x + landmarks[12].x) * 0.5
        p_neck_y = (landmarks[11].y + landmarks[12].y) * 0.5
        neck_pt = Landmark(p_neck_x, p_neck_y, 0, 1.0).to_screen(self.width, self.height)
        pygame.draw.line(screen, COLORS["head"], pos, neck_pt, 2)

        # Cranial ring outline & glow
        pygame.draw.circle(self.glow_surface, (*COLORS["head"], 35), pos, head_radius + 6)
        pygame.draw.circle(screen, COLORS["head"], pos, head_radius, 2)
        pygame.draw.circle(screen, (25, 15, 20), pos, head_radius - 2)

        # Cranial orientation reticle (crosshairs)
        pygame.draw.line(screen, COLORS["grid_accent"], (pos[0] - head_radius, pos[1]), (pos[0] + head_radius, pos[1]), 1)
        pygame.draw.line(screen, COLORS["grid_accent"], (pos[0], pos[1] - head_radius), (pos[0], pos[1] + head_radius), 1)

        # Glowing Visor Band
        visor_w = int(head_radius * 1.3)
        visor_h = 7
        visor_rect = pygame.Rect(pos[0] - visor_w // 2, pos[1] - 4, visor_w, visor_h)
        pygame.draw.rect(screen, (0, 240, 255), visor_rect, border_radius=2)
        pygame.draw.rect(self.glow_surface, (0, 240, 255, 120), visor_rect, border_radius=2)

        # Eye optical sensors
        for eye_idx in (2, 5):
            lm_eye = landmarks[eye_idx]
            if lm_eye.visibility > 0.3:
                ep = lm_eye.to_screen(self.width, self.height)
                pygame.draw.circle(screen, (255, 255, 255), ep, 2)

    def _update_trails(self, landmarks, processor, dt):
        """Update motion trails for key end effectors."""
        trail_joints = [15, 16, 27, 28, 0]  # Wrists, Ankles, Head

        for i in trail_joints:
            lm = landmarks[i]
            if lm.visibility < 0.3:
                continue

            vel = processor.get_joint_velocity(i)
            if vel > 0.006:
                pos = lm.to_screen(self.width, self.height)
                color = COLORS["trail"]
                for group, bones in BONE_GROUPS.items():
                    for a, b in bones:
                        if i == a or i == b:
                            color = COLORS.get(group, COLORS["trail"])
                            break
                self.trails[i].append(TrailPoint(pos[0], pos[1], color))

            self.trails[i] = [tp for tp in self.trails[i] if tp.update(dt)]
            if len(self.trails[i]) > 35:
                self.trails[i] = self.trails[i][-35:]

    def _draw_trails(self, screen):
        """Draw fading motion trail ribbons."""
        for joint_id, trail in self.trails.items():
            if len(trail) < 2:
                continue
            for i in range(1, len(trail)):
                tp = trail[i]
                tp_prev = trail[i - 1]
                alpha = tp.life
                color = tuple(int(c * alpha) for c in tp.color)
                thickness = max(1, int(3 * alpha))
                pygame.draw.line(
                    screen, color,
                    (int(tp_prev.x), int(tp_prev.y)),
                    (int(tp.x), int(tp.y)),
                    thickness
                )

    def _spawn_particles(self, landmarks, processor):
        """Spawn velocity-driven particles at dynamic joints."""
        for i in [15, 16, 27, 28]:
            vel = processor.get_joint_velocity(i)
            if vel > 0.018:
                lm = landmarks[i]
                if lm.visibility < 0.3:
                    continue
                pos = lm.to_screen(self.width, self.height)
                color = COLORS["particle"]
                num = min(4, int(vel * 180))
                for _ in range(num):
                    self.particles.append(Particle(pos[0], pos[1], color, random.uniform(2, 4)))

    def _update_ambient_particles(self, screen, dt):
        """Spawn slow drifting studio dust particles."""
        if random.random() < 0.25:
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color = (
                random.randint(25, 45),
                random.randint(45, 75),
                random.randint(75, 120),
            )
            self.particles.append(Particle(x, y, color, random.uniform(1, 2)))

    def _update_and_draw_particles(self, screen, dt):
        """Update and draw active particles."""
        alive = []
        for p in self.particles:
            if p.update(dt):
                p.draw(screen)
                alive.append(p)
        self.particles = alive
        if len(self.particles) > 400:
            self.particles = self.particles[-400:]

    def _draw_pip(self, screen, cam_frame, is_tracking):
        """Draw high-definition camera picture-in-picture in the bottom-right corner."""
        pip_w, pip_h = 320, 200
        margin = 16

        try:
            # Downscale & mirror camera frame for user-friendly reflection
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (pip_w, pip_h))
            rgb = cv2.flip(rgb, 1)

            pip_surface = pygame.image.frombuffer(rgb.tobytes(), (pip_w, pip_h), 'RGB')

            x = self.width - pip_w - margin
            y = self.height - pip_h - margin

            # Dark container backing
            bg_rect = pygame.Rect(x - 3, y - 22, pip_w + 6, pip_h + 25)
            pygame.draw.rect(screen, (6, 8, 14), bg_rect)
            pygame.draw.rect(screen, COLORS["grid_accent"], bg_rect, 1)

            # Blit camera image
            screen.blit(pip_surface, (x, y))

            # Header banner above PiP
            status_dot_col = COLORS["hud_active"] if is_tracking else COLORS["hud_standby"]
            pygame.draw.circle(screen, status_dot_col, (x + 8, y - 11), 4)

            feed_title = "OPTICAL CAMERA FEED"
            lbl = self.font_tiny.render(feed_title, True, COLORS["hud_text"])
            screen.blit(lbl, (x + 18, y - 16))

            res_text = f"{cam_frame.shape[1]}x{cam_frame.shape[0]}"
            res_lbl = self.font_tiny.render(res_text, True, COLORS["hud_dim"])
            screen.blit(res_lbl, (x + pip_w - res_lbl.get_width() - 4, y - 16))

            # Frame boundary glow
            pygame.draw.rect(screen, COLORS["hud_text"], (x, y, pip_w, pip_h), 1)

        except Exception as e:
            x = self.width - pip_w - margin
            y = self.height - pip_h - margin
            err = self.font_small.render(f"Cam PiP Error: {e}", True, COLORS["recording"])
            screen.blit(err, (x, y))

    def _draw_hud(self, screen, mode, fps, recording, frame_num, total_frames,
                  is_tracking, has_landmarks):
        """Draw engineering telemetry HUD and studio status banners."""
        margin = 20

        # ─── Top-Left: Studio Title & Solver Metrics ───
        title = self.font_large.render("ENGINEERING MOTION CAPTURE STUDIO", True, COLORS["hud_text"])
        screen.blit(title, (margin, margin))

        sub = self.font_small.render("3D ARMATURE SOLVER  │  MEDIAPIPE POSE + EMA FILTER", True, COLORS["hud_dim"])
        screen.blit(sub, (margin, margin + 28))

        # ─── Top-Center: Rig Tracking State Banner ───
        cx = self.width // 2
        if is_tracking:
            rig_banner = "⚙ RIG STATUS: OPTICAL CAPTURE [LIVE TRACKING]"
            banner_col = COLORS["hud_active"]
        else:
            rig_banner = "⚙ RIG STATUS: CALIBRATED STANDING SKELETON [MEDIUM RIG]"
            banner_col = COLORS["hud_standby"]

        banner_surf = self.font_medium.render(rig_banner, True, banner_col)
        bw = banner_surf.get_width()
        bh = banner_surf.get_height()
        b_rect = pygame.Rect(cx - bw // 2 - 10, margin - 2, bw + 20, bh + 4)
        pygame.draw.rect(screen, (8, 12, 22), b_rect)
        pygame.draw.rect(screen, banner_col, b_rect, 1)
        screen.blit(banner_surf, (cx - bw // 2, margin))

        # ─── Top-Right: Telemetry Metrics (FPS, Joints, Rig Mode) ───
        fps_text = f"FPS: {int(fps)}  │  JOINTS: 33/33  │  BONES: 24"
        fps_surf = self.font_small.render(fps_text, True, COLORS["hud_text"])
        fps_rect = fps_surf.get_rect(topright=(self.width - margin, margin))
        screen.blit(fps_surf, fps_rect)

        mode_badge = f"MODE: {mode.upper()}"
        mode_surf = self.font_small.render(mode_badge, True, COLORS["hud_dim"])
        mode_rect = mode_surf.get_rect(topright=(self.width - margin, margin + 22))
        screen.blit(mode_surf, mode_rect)

        # ─── Recording Indicator ───
        if recording:
            if int(self.time * 3) % 2 == 0:
                rec_text = "⏺ RECORDING MOCAP DATA"
                rec_surf = self.font_medium.render(rec_text, True, COLORS["recording"])
                rec_rect = rec_surf.get_rect(topright=(self.width - margin, margin + 46))
                screen.blit(rec_surf, rec_rect)

        # ─── Playback Progress Bar ───
        if mode == "playback" and total_frames > 0:
            pb_text = f"FRAME: {frame_num}/{total_frames}"
            pb_surf = self.font_small.render(pb_text, True, COLORS["playback"])
            pb_rect = pb_surf.get_rect(topright=(self.width - margin, margin + 46))
            screen.blit(pb_surf, pb_rect)

            bar_w = 180
            bar_h = 5
            bar_x = self.width - margin - bar_w
            bar_y = margin + 68
            prog = frame_num / max(1, total_frames)
            pygame.draw.rect(screen, COLORS["hud_dim"], (bar_x, bar_y, bar_w, bar_h), 1)
            pygame.draw.rect(screen, COLORS["playback"], (bar_x, bar_y, int(bar_w * prog), bar_h))

        # ─── Bottom-Left: Keyboard Controls Bar ───
        controls = [
            "R: Record Mocap  │  P: Playback  │  S: Save  │  L: Load",
            "C: Camera Feed PiP  │  G: Floor Grid  │  ESC: Exit",
        ]
        for i, text in enumerate(controls):
            ctrl_surf = self.font_tiny.render(text, True, COLORS["hud_dim"])
            y = self.height - margin - (len(controls) - i) * 16
            screen.blit(ctrl_surf, (margin + 30, y))

        # ─── Technical Corner Brackets ───
        blen = 16
        bcol = COLORS["hud_text"]
        # TL
        pygame.draw.line(screen, bcol, (6, 6), (6 + blen, 6), 1)
        pygame.draw.line(screen, bcol, (6, 6), (6, 6 + blen), 1)
        # TR
        pygame.draw.line(screen, bcol, (self.width - 6, 6), (self.width - 6 - blen, 6), 1)
        pygame.draw.line(screen, bcol, (self.width - 6, 6), (self.width - 6, 6 + blen), 1)
        # BL
        pygame.draw.line(screen, bcol, (6, self.height - 6), (6 + blen, self.height - 6), 1)
        pygame.draw.line(screen, bcol, (6, self.height - 6), (6, self.height - 6 - blen), 1)
        # BR
        pygame.draw.line(screen, bcol, (self.width - 6, self.height - 6),
                         (self.width - 6 - blen, self.height - 6), 1)
        pygame.draw.line(screen, bcol, (self.width - 6, self.height - 6),
                         (self.width - 6, self.height - 6 - blen), 1)

    def _draw_vignette(self, screen):
        """Draw subtle technical corner vignette."""
        vignette = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        radius = int(math.sqrt(self.width ** 2 + self.height ** 2) * 0.28)
        for cx, cy in [(0, 0), (self.width, 0), (0, self.height), (self.width, self.height)]:
            pygame.draw.circle(vignette, (0, 0, 0, 35), (cx, cy), radius)
        screen.blit(vignette, (0, 0))

    def draw_no_camera(self, screen, fps):
        """Fallback screen (not normally used since standing rig is rendered)."""
        screen.fill(COLORS["bg"])
        self._draw_grid(screen)
        text = self.font_large.render("CONNECTING TO OPTICAL SENSORS...", True, COLORS["hud_standby"])
        rect = text.get_rect(center=(self.width // 2, self.height // 2))
        screen.blit(text, rect)
