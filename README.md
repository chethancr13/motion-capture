# Motion Capture Studio 🎬

A real-time body motion capture system that tracks your body using a webcam and renders a **cyberpunk neon skeleton character** that mirrors your movements in real time.

![demo](https://img.shields.io/badge/Python-3.8%2B-blue)
![status](https://img.shields.io/badge/Status-Working-brightgreen)

## Features

- **Engineering 3D Animation Skeleton** — Maya/Blender-style octahedral diamond armatures with faceted highlights and longitudinal axes
- **Real-Time Joint Telemetry** — Live flexion angle readouts (`L-ELB`, `R-ELB`, `L-KNEE`, `R-KNEE`) with technical callouts and arcs
- **Center-of-Mass (COM) Gizmo** — 3-axis RGB coordinate gimbal at the pelvis root
- **Calibrated Standing Rig (Medium Size)** — Anatomically proportioned standing armature centered on screen with procedural breathing and balance dynamics
- **Optical Camera PiP Feed** — 720p 1280x720 webcam feed in the bottom-right corner with mirror flip and tracking status
- **Studio Environment** — Perspective floor grid with contact shadows, pressure rings, and vertical height metric ruler (0.0m - 2.0m)
- **Motion Recording & Playback** — Record live sessions to JSON and replay them frame-by-frame
- **Particle Trails & Glow Bloom** — Dynamic velocity-responsive particle bursts and CRT scanlines

## Quick Start

```bash
# Run directly with the launcher:
./run.sh

# Or with Python 3.11:
/usr/local/bin/python3.11 motion_capture.py
```

## Controls

| Key     | Action                        |
|---------|-------------------------------|
| `R`     | Start / Stop recording        |
| `P`     | Toggle playback               |
| `S`     | Save recording to file        |
| `L`     | Load last recording           |
| `C`     | Toggle camera PiP             |
| `G`     | Toggle background grid        |
| `SPACE` | Pause / Resume playback       |
| `ESC`   | Quit                          |

## How It Works

1. **OpenCV** captures your webcam feed
2. **MediaPipe Pose** detects 33 body landmarks (shoulders, elbows, wrists, hips, knees, ankles, etc.)
3. **Pose Processor** applies exponential smoothing and computes per-joint velocity
4. **Renderer** draws a stylized skeleton with:
   - Color-coded body regions (cyan torso, purple/pink arms, green/gold legs, orange head)
   - Additive glow bloom on bones and joints
   - Motion trails on wrists, ankles, and nose
   - Velocity-responsive particle effects
   - CRT scanlines and vignette

## Project Structure

```
animation/
├── motion_capture.py    # Main application & game loop
├── pose_processor.py    # Pose detection, smoothing, skeleton data
├── renderer.py          # Pygame rendering engine (neon, particles, HUD)
├── requirements.txt     # Python dependencies
├── recordings/          # Saved motion recordings (JSON)
└── README.md            # This file
```

## Requirements

- Python 3.8+
- A working webcam
- macOS, Linux, or Windows
