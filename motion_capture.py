"""
Motion Capture System — Main Application
==========================================
Real-time body motion capture using webcam + MediaPipe Pose.
Renders a stylized neon skeleton character in Pygame that
mirrors your movements. Supports recording & playback.

Controls:
    R       — Start/Stop recording
    P       — Play back last recording
    S       — Save recording to file
    L       — Load recording from file
    C       — Toggle camera PiP view
    G       — Toggle background grid
    SPACE   — Pause/Resume (playback mode)
    ESC     — Quit

Requirements:
    pip install -r requirements.txt
"""

import sys
import os
import json
import time

import cv2
import mediapipe as mp
import pygame
import numpy as np

from pose_processor import PoseProcessor
from renderer import Renderer


# ── Configuration ────────────────────────────────────────────────────────────

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "Motion Capture Studio"
CAMERA_INDEX = 0
TARGET_FPS = 60

# MediaPipe config
MP_MODEL_COMPLEXITY = 1       # 0=lite, 1=full, 2=heavy
MP_MIN_DETECTION_CONF = 0.6
MP_MIN_TRACKING_CONF = 0.6
MP_SMOOTH_LANDMARKS = True


class MotionCaptureApp:
    """Main application class for the motion capture system."""

    def __init__(self):
        # ── Pygame Init ──
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)
        self.screen = pygame.display.set_mode(
            (WINDOW_WIDTH, WINDOW_HEIGHT),
            pygame.DOUBLEBUF | pygame.HWSURFACE
        )
        self.clock = pygame.time.Clock()

        # ── MediaPipe Pose ──
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MP_MODEL_COMPLEXITY,
            smooth_landmarks=MP_SMOOTH_LANDMARKS,
            min_detection_confidence=MP_MIN_DETECTION_CONF,
            min_tracking_confidence=MP_MIN_TRACKING_CONF,
        )

        # ── Camera ──
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.camera_ok = False
        self.cam_fail_count = 0

        if self.cap.isOpened():
            print("⏳ Initializing optical sensor (webcam)...")
            import time as _time
            warmup_start = _time.time()
            while _time.time() - warmup_start < 2.0:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.camera_ok = True
                    self.last_cam_frame = frame
                    print(f"✓ Optical camera online: {frame.shape[1]}x{frame.shape[0]}")
                    break
                _time.sleep(0.05)

            if not self.camera_ok:
                print("⚠ Camera stream warming up in background...")
                self.camera_ok = True
        else:
            print("⚠ Camera device not detected — studio standing armature active")

        # ── Modules ──
        self.processor = PoseProcessor(smoothing_factor=0.45)
        self.renderer = Renderer(WINDOW_WIDTH, WINDOW_HEIGHT)

        # ── State ──
        self.running = True
        self.mode = "live"              # "live" or "playback"
        self.show_pip = True            # camera picture-in-picture
        self.show_grid = True
        self.fps = 0.0

        # Recording
        self.recording = False
        self.recorded_frames = []       # list of serialized landmark frames
        self.playback_index = 0
        self.playback_paused = False

        # Last camera frame for PiP
        self.last_cam_frame = None

    def run(self):
        """Main application loop."""
        print("\n" + "=" * 50)
        print("  MOTION CAPTURE STUDIO")
        print("=" * 50)
        print("  Stand in front of your webcam.")
        print("  Your body will be tracked and rendered")
        print("  as a neon skeleton character in real time.")
        print()
        print("  Controls:")
        print("    R — Record / Stop recording")
        print("    P — Playback recording")
        print("    S — Save recording to file")
        print("    L — Load recording from file")
        print("    C — Toggle camera PiP")
        print("    G — Toggle grid")
        print("    SPACE — Pause playback")
        print("    ESC — Quit")
        print("=" * 50 + "\n")

        try:
            while self.running:
                self._handle_events()

                if self.mode == "live":
                    self._live_frame()
                elif self.mode == "playback":
                    self._playback_frame()

                pygame.display.flip()
                self.clock.tick(TARGET_FPS)
                self.fps = self.clock.get_fps()

        except KeyboardInterrupt:
            pass
        finally:
            self._cleanup()

    def _handle_events(self):
        """Handle Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_r:
                    self._toggle_recording()

                elif event.key == pygame.K_p:
                    self._toggle_playback()

                elif event.key == pygame.K_s:
                    self._save_recording()

                elif event.key == pygame.K_l:
                    self._load_recording()

                elif event.key == pygame.K_c:
                    self.show_pip = not self.show_pip
                    print(f"Camera PiP: {'ON' if self.show_pip else 'OFF'}")

                elif event.key == pygame.K_g:
                    self.show_grid = not self.show_grid
                    print(f"Grid: {'ON' if self.show_grid else 'OFF'}")

                elif event.key == pygame.K_SPACE:
                    if self.mode == "playback":
                        self.playback_paused = not self.playback_paused
                        state = "PAUSED" if self.playback_paused else "PLAYING"
                        print(f"Playback: {state}")

    def _live_frame(self):
        """Process one live frame from camera or standing rig."""
        results = None

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.last_cam_frame = frame
                self.cam_fail_count = 0
                self.camera_ok = True

                # Downscale frame for fast MediaPipe inference if needed
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640.0 / w
                    mp_frame = cv2.resize(frame, (640, int(h * scale)))
                else:
                    mp_frame = frame

                rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self.pose.process(rgb)
            else:
                self.cam_fail_count += 1
                if self.cam_fail_count > 60:
                    # Soft re-initialize capture if camera stalls
                    self.cam_fail_count = 0
                    try:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(CAMERA_INDEX)
                    except Exception:
                        pass

        # Always run processor: either tracks detected body or smoothly blends into default standing skeleton
        current_time = time.time()
        landmarks = self.processor.process(results, time_val=current_time)

        # Record if active and tracking real body
        if self.recording and self.processor.is_tracking:
            serialized = self.processor.serialize_frame(landmarks)
            self.recorded_frames.append(serialized)

        # Camera PiP in the corner
        cam_for_pip = self.last_cam_frame if (self.show_pip and self.last_cam_frame is not None) else None

        self.renderer.render_frame(
            self.screen, landmarks, self.processor,
            mode="live", fps=self.fps,
            recording=self.recording,
            cam_frame=cam_for_pip,
            is_tracking=self.processor.is_tracking,
        )

    def _playback_frame(self):
        """Play back a recorded motion."""
        if not self.recorded_frames:
            print("No recording to play back!")
            self.mode = "live"
            return

        if not self.playback_paused:
            self.playback_index += 1

        if self.playback_index >= len(self.recorded_frames):
            self.playback_index = 0  # loop

        frame_data = self.recorded_frames[self.playback_index]
        landmarks = self.processor.deserialize_frame(frame_data)

        # Feed through processor for velocity computation
        if landmarks is not None:
            if self.playback_index > 0:
                prev_data = self.recorded_frames[self.playback_index - 1]
                prev_landmarks = self.processor.deserialize_frame(prev_data)
                self.processor.prev_landmarks = prev_landmarks
                self.processor.velocity = self.processor._compute_velocity(
                    prev_landmarks, landmarks
                )
            else:
                self.processor.velocity = [0.0] * len(landmarks)

        self.renderer.render_frame(
            self.screen, landmarks, self.processor,
            mode="playback", fps=self.fps,
            frame_num=self.playback_index,
            total_frames=len(self.recorded_frames),
            is_tracking=True,
        )

    def _toggle_recording(self):
        """Toggle motion recording."""
        if self.mode != "live":
            print("Switch to live mode first (press P to stop playback)")
            return

        self.recording = not self.recording
        if self.recording:
            self.recorded_frames = []
            print("⏺ Recording started...")
        else:
            duration = len(self.recorded_frames) / max(1, self.fps)
            print(f"⏹ Recording stopped — {len(self.recorded_frames)} frames "
                  f"({duration:.1f}s)")

    def _toggle_playback(self):
        """Toggle between live and playback modes."""
        if self.mode == "live":
            if not self.recorded_frames:
                print("No recording to play! Press R to record first.")
                return
            self.mode = "playback"
            self.playback_index = 0
            self.playback_paused = False
            self.recording = False
            print(f"▶ Playback started ({len(self.recorded_frames)} frames)")
        else:
            self.mode = "live"
            self.playback_paused = False
            print("◼ Playback stopped — back to live")

    def _save_recording(self):
        """Save recorded motion to a JSON file."""
        if not self.recorded_frames:
            print("Nothing to save — record something first!")
            return

        # Create recordings directory
        rec_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(rec_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(rec_dir, f"motion_{timestamp}.json")

        data = {
            "version": 1,
            "fps": int(self.fps) if self.fps > 0 else 30,
            "frame_count": len(self.recorded_frames),
            "frames": self.recorded_frames,
        }

        with open(filename, 'w') as f:
            json.dump(data, f)

        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"💾 Saved: {filename} ({size_mb:.1f} MB)")

    def _load_recording(self):
        """Load a recorded motion from the most recent JSON file."""
        rec_dir = os.path.join(os.path.dirname(__file__), "recordings")

        if not os.path.isdir(rec_dir):
            print("No recordings directory found!")
            return

        files = sorted([
            f for f in os.listdir(rec_dir) if f.endswith('.json')
        ])

        if not files:
            print("No recordings found!")
            return

        # Load the most recent
        filepath = os.path.join(rec_dir, files[-1])
        print(f"Loading: {filepath}...")

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.recorded_frames = data["frames"]
        self.playback_index = 0
        self.mode = "playback"
        self.playback_paused = False
        print(f"▶ Loaded {data['frame_count']} frames — playback started")

    def _cleanup(self):
        """Release resources."""
        print("\nShutting down...")
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.pose.close()
        pygame.quit()
        print("Done.")


def main():
    """Entry point."""
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required!")
        sys.exit(1)

    app = MotionCaptureApp()
    app.run()


if __name__ == "__main__":
    main()
