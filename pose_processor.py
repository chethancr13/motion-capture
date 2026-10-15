"""
Pose Processor Module
Handles MediaPipe pose detection and landmark processing.
Provides smoothed, normalized skeleton data for the renderer.
"""

import math
import numpy as np

# MediaPipe Pose landmark indices
LANDMARK_NAMES = {
    0: "nose",
    1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
    4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
    7: "left_ear", 8: "right_ear",
    9: "mouth_left", 10: "mouth_right",
    11: "left_shoulder", 12: "right_shoulder",
    13: "left_elbow", 14: "right_elbow",
    15: "left_wrist", 16: "right_wrist",
    17: "left_pinky", 18: "right_pinky",
    19: "left_index", 20: "right_index",
    21: "left_thumb", 22: "right_thumb",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

# Skeleton bone connections (pairs of landmark indices)
SKELETON_BONES = [
    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left torso
    (12, 24),  # right torso
    (23, 24),  # hips
    # Left arm
    (11, 13),  # left upper arm
    (13, 15),  # left forearm
    # Right arm
    (12, 14),  # right upper arm
    (14, 16),  # right forearm
    # Left leg
    (23, 25),  # left thigh
    (25, 27),  # left shin
    # Right leg
    (24, 26),  # right thigh
    (26, 28),  # right shin
    # Hands (fingers)
    (15, 17),  # left pinky
    (15, 19),  # left index
    (15, 21),  # left thumb
    (16, 18),  # right pinky
    (16, 20),  # right index
    (16, 22),  # right thumb
    # Feet
    (27, 29),  # left heel
    (27, 31),  # left foot
    (28, 30),  # right heel
    (28, 32),  # right foot
    # Head/face
    (0, 11),   # nose to left shoulder (neck approx)
    (0, 12),   # nose to right shoulder (neck approx)
]

# Body region groups for coloring
BONE_GROUPS = {
    "torso": [(11, 12), (11, 23), (12, 24), (23, 24)],
    "left_arm": [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21)],
    "right_arm": [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22)],
    "left_leg": [(23, 25), (25, 27), (27, 29), (27, 31)],
    "right_leg": [(24, 26), (26, 28), (28, 30), (28, 32)],
    "head": [(0, 11), (0, 12)],
}

# Joint groups for rendering sizes
MAJOR_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}
MINOR_JOINTS = {0, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32}

# Primary skeleton bones for the clean engineering look (skip finger/toe detail)
PRIMARY_BONES = [
    # Spine/Torso
    (11, 12),  # shoulders
    (11, 23),  # left torso
    (12, 24),  # right torso
    (23, 24),  # hips
    # Left arm
    (11, 13),  # left upper arm
    (13, 15),  # left forearm
    # Right arm
    (12, 14),  # right upper arm
    (14, 16),  # right forearm
    # Left leg
    (23, 25),  # left thigh
    (25, 27),  # left shin
    # Right leg
    (24, 26),  # right thigh
    (26, 28),  # right shin
    # Neck
    (0, 11),   # nose to left shoulder
    (0, 12),   # nose to right shoulder
]


def create_default_standing_pose(time_val=0.0):
    """
    Create a default standing skeleton in a natural animation A-pose.
    All coordinates are normalized (0-1). Centered horizontally and
    positioned at medium size with subtle procedural breathing/balance dynamics.

    Returns a list of 33 Landmark objects.
    """
    landmarks = [None] * 33

    # Procedural idle animation (subtle breathing + balance sway)
    breath = math.sin(time_val * 2.0) * 0.004
    sway = math.sin(time_val * 0.9) * 0.002
    arm_sway = math.cos(time_val * 1.5) * 0.003

    # Center X, top of body Y
    cx = 0.50 + sway
    shoulder_w = 0.095  # natural shoulder width
    hip_w = 0.055       # natural hip width

    # Vertical positions (top to bottom, normalized, medium size ~70% screen height)
    head_y = 0.16 + breath * 0.3
    shoulder_y = 0.26 + breath
    elbow_y = 0.39 + breath * 0.7
    wrist_y = 0.52 + breath * 0.4
    hip_y = 0.49
    knee_y = 0.67
    ankle_y = 0.84
    foot_y = 0.86
    heel_y = 0.85

    # Eye/ear spread
    eye_spread = 0.014
    ear_spread = 0.024

    vis = 1.0  # full visibility

    # Head landmarks
    landmarks[0]  = Landmark(cx, head_y, 0, vis)                          # nose
    landmarks[1]  = Landmark(cx - eye_spread * 0.5, head_y - 0.015, 0, vis)  # left_eye_inner
    landmarks[2]  = Landmark(cx - eye_spread, head_y - 0.015, 0, vis)     # left_eye
    landmarks[3]  = Landmark(cx - eye_spread * 1.5, head_y - 0.015, 0, vis)  # left_eye_outer
    landmarks[4]  = Landmark(cx + eye_spread * 0.5, head_y - 0.015, 0, vis)  # right_eye_inner
    landmarks[5]  = Landmark(cx + eye_spread, head_y - 0.015, 0, vis)     # right_eye
    landmarks[6]  = Landmark(cx + eye_spread * 1.5, head_y - 0.015, 0, vis)  # right_eye_outer
    landmarks[7]  = Landmark(cx - ear_spread, head_y, 0, vis)             # left_ear
    landmarks[8]  = Landmark(cx + ear_spread, head_y, 0, vis)             # right_ear
    landmarks[9]  = Landmark(cx - 0.008, head_y + 0.015, 0, vis)          # mouth_left
    landmarks[10] = Landmark(cx + 0.008, head_y + 0.015, 0, vis)          # mouth_right

    # Shoulders
    landmarks[11] = Landmark(cx - shoulder_w, shoulder_y, 0, vis)         # left_shoulder
    landmarks[12] = Landmark(cx + shoulder_w, shoulder_y, 0, vis)         # right_shoulder

    # Arms (natural ready standing A-pose)
    landmarks[13] = Landmark(cx - shoulder_w - 0.035, elbow_y, 0, vis)    # left_elbow
    landmarks[14] = Landmark(cx + shoulder_w + 0.035, elbow_y, 0, vis)    # right_elbow
    landmarks[15] = Landmark(cx - shoulder_w - 0.045 + arm_sway, wrist_y, 0, vis)  # left_wrist
    landmarks[16] = Landmark(cx + shoulder_w + 0.045 - arm_sway, wrist_y, 0, vis)  # right_wrist

    # Fingers (hands in relaxed ready pose)
    landmarks[17] = Landmark(cx - shoulder_w - 0.05 + arm_sway, wrist_y + 0.02, 0, vis)  # left_pinky
    landmarks[18] = Landmark(cx + shoulder_w + 0.05 - arm_sway, wrist_y + 0.02, 0, vis)  # right_pinky
    landmarks[19] = Landmark(cx - shoulder_w - 0.045 + arm_sway, wrist_y + 0.025, 0, vis) # left_index
    landmarks[20] = Landmark(cx + shoulder_w + 0.045 - arm_sway, wrist_y + 0.025, 0, vis) # right_index
    landmarks[21] = Landmark(cx - shoulder_w - 0.035 + arm_sway, wrist_y + 0.018, 0, vis) # left_thumb
    landmarks[22] = Landmark(cx + shoulder_w + 0.035 - arm_sway, wrist_y + 0.018, 0, vis) # right_thumb

    # Hips (pelvis)
    landmarks[23] = Landmark(cx - hip_w, hip_y, 0, vis)                  # left_hip
    landmarks[24] = Landmark(cx + hip_w, hip_y, 0, vis)                  # right_hip

    # Legs (straight standing with feet grounded)
    landmarks[25] = Landmark(cx - hip_w - 0.008, knee_y, 0, vis)         # left_knee
    landmarks[26] = Landmark(cx + hip_w + 0.008, knee_y, 0, vis)         # right_knee
    landmarks[27] = Landmark(cx - hip_w - 0.01, ankle_y, 0, vis)         # left_ankle
    landmarks[28] = Landmark(cx + hip_w + 0.01, ankle_y, 0, vis)         # right_ankle

    # Feet
    landmarks[29] = Landmark(cx - hip_w - 0.02, heel_y, 0, vis)          # left_heel
    landmarks[30] = Landmark(cx + hip_w + 0.02, heel_y, 0, vis)          # right_heel
    landmarks[31] = Landmark(cx - hip_w - 0.005, foot_y, 0, vis)         # left_foot_index
    landmarks[32] = Landmark(cx + hip_w + 0.005, foot_y, 0, vis)         # right_foot_index

    return landmarks


class Landmark:
    """Represents a single body landmark with position and visibility."""

    __slots__ = ['x', 'y', 'z', 'visibility']

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def to_tuple(self):
        return (self.x, self.y, self.z)

    def to_screen(self, width, height, margin=60):
        """Convert normalized coordinates to screen coordinates."""
        sx = margin + self.x * (width - 2 * margin)
        sy = margin + self.y * (height - 2 * margin)
        return (int(sx), int(sy))


class PoseProcessor:
    """
    Processes MediaPipe pose landmarks with smoothing, filtering,
    and automatic fallback to default standing skeleton when no body is tracked.
    """

    def __init__(self, smoothing_factor=0.4, visibility_threshold=0.5):
        self.smoothing_factor = smoothing_factor
        self.visibility_threshold = visibility_threshold
        self.prev_landmarks = None
        self.velocity = None  # For velocity-based effects
        self.is_tracking = False
        self.blend_to_default_weight = 1.0  # 0.0 = live, 1.0 = default standing pose

    def process(self, mp_results, time_val=0.0):
        """
        Process MediaPipe results into smoothed Landmark objects.
        If no pose is detected, smoothly transitions to the default standing skeleton.

        Args:
            mp_results: MediaPipe pose detection results
            time_val: Current runtime in seconds (for idle animation)

        Returns:
            list of 33 Landmark objects
        """
        has_detection = (
            mp_results is not None
            and mp_results.pose_landmarks is not None
            and len(mp_results.pose_landmarks.landmark) == 33
        )

        default_pose = create_default_standing_pose(time_val)

        if has_detection:
            self.is_tracking = True
            # Gradually blend towards live pose
            self.blend_to_default_weight = max(0.0, self.blend_to_default_weight - 0.15)

            raw_landmarks = []
            for lm in mp_results.pose_landmarks.landmark:
                raw_landmarks.append(Landmark(lm.x, lm.y, lm.z, lm.visibility))

            # Apply smoothing
            if self.prev_landmarks is not None:
                smoothed = self._smooth(raw_landmarks)
            else:
                smoothed = raw_landmarks
        else:
            self.is_tracking = False
            # Gradually blend towards default standing pose
            self.blend_to_default_weight = min(1.0, self.blend_to_default_weight + 0.1)

            if self.prev_landmarks is not None and self.blend_to_default_weight < 0.99:
                smoothed = self._smooth(default_pose)
            else:
                smoothed = default_pose

        # Compute velocities for effects
        if self.prev_landmarks is not None:
            self.velocity = self._compute_velocity(self.prev_landmarks, smoothed)
        else:
            self.velocity = [0.0] * len(smoothed)

        self.prev_landmarks = [
            Landmark(l.x, l.y, l.z, l.visibility) for l in smoothed
        ]

        return smoothed

    def _smooth(self, current):
        """Apply exponential moving average smoothing."""
        smoothed = []
        alpha = self.smoothing_factor
        for i, (cur, prev) in enumerate(zip(current, self.prev_landmarks)):
            s = Landmark(
                x=prev.x + alpha * (cur.x - prev.x),
                y=prev.y + alpha * (cur.y - prev.y),
                z=prev.z + alpha * (cur.z - prev.z),
                visibility=cur.visibility,
            )
            smoothed.append(s)
        return smoothed

    def _compute_velocity(self, prev, current):
        """Compute per-joint velocity magnitude."""
        velocities = []
        for p, c in zip(prev, current):
            dx = c.x - p.x
            dy = c.y - p.y
            dz = c.z - p.z
            vel = math.sqrt(dx * dx + dy * dy + dz * dz)
            velocities.append(vel)
        return velocities

    def get_joint_velocity(self, index):
        """Get velocity of a specific joint."""
        if self.velocity and 0 <= index < len(self.velocity):
            return self.velocity[index]
        return 0.0

    def get_bone_midpoint(self, landmarks, idx_a, idx_b):
        """Get the midpoint between two landmarks."""
        a = landmarks[idx_a]
        b = landmarks[idx_b]
        return Landmark(
            (a.x + b.x) / 2,
            (a.y + b.y) / 2,
            (a.z + b.z) / 2,
            min(a.visibility, b.visibility),
        )

    def get_torso_center(self, landmarks):
        """Get the center of the torso."""
        indices = [11, 12, 23, 24]
        cx = sum(landmarks[i].x for i in indices) / 4
        cy = sum(landmarks[i].y for i in indices) / 4
        cz = sum(landmarks[i].z for i in indices) / 4
        return Landmark(cx, cy, cz, 1.0)

    def compute_angle(self, landmarks, idx_a, idx_b, idx_c):
        """Compute the angle at joint B between segments BA and BC (in degrees)."""
        a = np.array([landmarks[idx_a].x, landmarks[idx_a].y])
        b = np.array([landmarks[idx_b].x, landmarks[idx_b].y])
        c = np.array([landmarks[idx_c].x, landmarks[idx_c].y])

        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        return angle

    def serialize_frame(self, landmarks):
        """Serialize landmarks to a list of dicts for recording."""
        if landmarks is None:
            return None
        return [
            {"x": l.x, "y": l.y, "z": l.z, "v": l.visibility}
            for l in landmarks
        ]

    def deserialize_frame(self, data):
        """Deserialize a recorded frame back to Landmark objects."""
        if data is None:
            return None
        return [
            Landmark(d["x"], d["y"], d["z"], d["v"])
            for d in data
        ]
