#!/usr/bin/env bash
# Motion Capture Studio Launcher
# Uses Python 3.11 with OpenCV, MediaPipe, and Pygame installed

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if command -v /usr/local/bin/python3.11 &> /dev/null; then
    exec /usr/local/bin/python3.11 motion_capture.py "$@"
elif command -v python3.11 &> /dev/null; then
    exec python3.11 motion_capture.py "$@"
elif command -v python3 &> /dev/null; then
    exec python3 motion_capture.py "$@"
else
    exec python motion_capture.py "$@"
fi
