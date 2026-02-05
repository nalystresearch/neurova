# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
LIVE Face Detection with Native C++ Haar Cascade
Camera is ALWAYS released when window closes or script exits.
"""

import sys
import os
import time
import signal
import subprocess
import threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "Neurova"))

import numpy as np

# camera release - critical section

_camera_proc = None
_display_proc = None
_running = True

def release_camera():
    """Release camera hardware immediately."""
    global _camera_proc, _display_proc, _running
    _running = False
    
    # Kill camera process (ffmpeg)
    if _camera_proc:
        try:
            _camera_proc.terminate()
            _camera_proc.kill()
        except:
            pass
        _camera_proc = None
    
    # Kill display process (ffplay)
    if _display_proc:
        try:
            _display_proc.terminate()
            _display_proc.kill()
        except:
            pass
        _display_proc = None
    
    # Force kill any orphaned ffmpeg/ffplay
    os.system("pkill -9 -f 'ffmpeg.*avfoundation' 2>/dev/null")
    os.system("pkill -9 -f ffplay 2>/dev/null")

def signal_handler(sig, frame):
    """Handle Ctrl+C - release camera and exit."""
    print("\n\nExiting...")
    release_camera()
    os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Kill any old instances first
print("Releasing any old camera connections...")
os.system("pkill -9 -f 'ffmpeg.*avfoundation' 2>/dev/null")
os.system("pkill -9 -f ffplay 2>/dev/null")
time.sleep(0.3)

# main detection script

print("=" * 55)
print("NEUROVA LIVE FACE DETECTION (C++ Native)")
print("=" * 55)
print()

# load native haar module
try:
    from neurova.face import haar_native
    from neurova.face.haar_loader import load_cascade
    print("[ok] Native C++ Haar cascade loaded")
except ImportError as e:
    print(f"ERROR: Native module not available: {e}")
    sys.exit(1)

# load cascade from neurova/data/haarcascades
CASCADE_PATH = str(SCRIPT_DIR.parent.parent / "neurova" / "data" / "haarcascades" / "haarcascade_frontalface_default.xml")
if not load_cascade(CASCADE_PATH):
    print("ERROR: Failed to load cascade")
    sys.exit(1)

# Settings
WIDTH = 640
HEIGHT = 480
FPS = 30
CAMERA_DEVICE = "0"

def start_camera():
    """Start ffmpeg webcam capture."""
    global _camera_proc
    cmd = [
        "ffmpeg",
        "-f", "avfoundation",
        "-framerate", str(FPS),
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-i", f"{CAMERA_DEVICE}:",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-"
    ]
    _camera_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)
    return _camera_proc

def read_frame():
    """Read one frame from camera."""
    global _camera_proc
    if not _camera_proc or not _running:
        return None
    frame_size = WIDTH * HEIGHT * 3
    try:
        raw = _camera_proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3)).copy()
    except:
        return None

def rgb_to_gray(rgb):
    """Fast grayscale conversion."""
    r = rgb[:, :, 0].astype(np.uint16)
    g = rgb[:, :, 1].astype(np.uint16)
    b = rgb[:, :, 2].astype(np.uint16)
    return ((r * 77 + g * 150 + b * 29) >> 8).astype(np.uint8)

def draw_boxes(frame, faces):
    """Draw simple green boxes (30% larger, fast)."""
    for (x, y, w, h, conf) in faces:
        # 30% larger, centered
        expand = 0.30
        x1 = max(0, x - int(w * expand / 2))
        y1 = max(0, y - int(h * expand / 2))
        x2 = min(WIDTH-1, x + w + int(w * expand / 2))
        y2 = min(HEIGHT-1, y + h + int(h * expand / 2))
        # Simple 2px lines
        frame[y1:y1+2, x1:x2] = [0, 255, 0]
        frame[y2-2:y2, x1:x2] = [0, 255, 0]
        frame[y1:y2, x1:x1+2] = [0, 255, 0]
        frame[y1:y2, x2-2:x2] = [0, 255, 0]
    return frame

def run_with_matplotlib():
    """Run with matplotlib display - releases camera when window closes."""
    global _running
    
    try:
        import matplotlib
        matplotlib.use('macosx')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib error: {e}")
        return False
    
    print()
    print(f"Starting camera (device {CAMERA_DEVICE})...")
    print("Close window to quit (camera will be released)")
    print()
    
    start_camera()
    time.sleep(0.5)
    
    frame = read_frame()
    if frame is None:
        print("ERROR: Could not read from camera")
        release_camera()
        return False
    
    print("[ok] Camera streaming")
    print("-" * 55)
    
    # Create minimal figure - no extra decorations for speed
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.canvas.manager.set_window_title("Neurova Face Detection")
    im = ax.imshow(frame)
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])  # full figure, no margins
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.ion()
    plt.show()
    
    frame_count = 0
    start_time = time.time()
    last_faces = []
    
    # Watchdog: check if window closed
    def window_watchdog():
        while _running:
            time.sleep(0.1)
            if not plt.fignum_exists(fig.number):
                print("\nWindow closed - releasing camera...")
                release_camera()
                break
    
    watchdog = threading.Thread(target=window_watchdog, daemon=True)
    watchdog.start()
    
    try:
        while _running and plt.fignum_exists(fig.number):
            frame = read_frame()
            if frame is None:
                break
            
            frame_count += 1
            
            # Detect faces every 3 frames for speed
            if frame_count % 3 == 1:
                gray = rgb_to_gray(frame)
                faces = haar_native.detect(gray, scale_factor=1.3, min_neighbors=3, min_size=60)
                last_faces = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4]) for f in faces]
            
            frame = draw_boxes(frame, last_faces)
            im.set_data(frame)
            
            # Fast update - no pause
            fig.canvas.flush_events()
            
            elapsed = time.time() - start_time
            fps_val = frame_count / elapsed if elapsed > 0 else 0
            
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: FPS={fps_val:.1f}, Faces={len(last_faces)}")
    except:
        pass
    finally:
        release_camera()
        try:
            plt.close('all')
        except:
            pass
    
    elapsed = time.time() - start_time
    fps_val = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nDone! {frame_count} frames, {fps_val:.1f} FPS average")
    print("[ok] Camera released")
    return frame_count > 0

def run_with_ffplay():
    """Run with ffplay display - releases camera when window closes."""
    global _display_proc, _running
    
    print()
    print(f"Starting camera with ffplay (device {CAMERA_DEVICE})...")
    print("Press 'q' in window to quit (camera will be released)")
    print()
    
    start_camera()
    time.sleep(0.5)
    
    ffplay_cmd = [
        "ffplay",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}",
        "-framerate", str(FPS),
        "-window_title", "Neurova Face Detection - Press q to quit",
        "-"
    ]
    _display_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    print("[ok] Camera streaming")
    print("-" * 55)
    
    frame_count = 0
    start_time = time.time()
    last_faces = []
    
    # Watchdog: check if ffplay closed
    def ffplay_watchdog():
        while _running and _display_proc:
            if _display_proc.poll() is not None:
                print("\nWindow closed - releasing camera...")
                release_camera()
                break
            time.sleep(0.1)
    
    watchdog = threading.Thread(target=ffplay_watchdog, daemon=True)
    watchdog.start()
    
    try:
        while _running:
            frame = read_frame()
            if frame is None:
                break
            
            frame_count += 1
            
            if frame_count % 2 == 1:
                gray = rgb_to_gray(frame)
                faces = haar_native.detect(gray, scale_factor=1.2, min_neighbors=2, min_size=50)
                last_faces = [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4]) for f in faces]
            
            frame = draw_boxes(frame, last_faces)
            
            try:
                if _display_proc and _display_proc.stdin:
                    _display_proc.stdin.write(frame.tobytes())
            except:
                break
            
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_val = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}: FPS={fps_val:.1f}, Faces={len(last_faces)}")
    except:
        pass
    finally:
        release_camera()
    
    elapsed = time.time() - start_time
    fps_val = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nDone! {frame_count} frames, {fps_val:.1f} FPS average")
    print("[ok] Camera released")

# main entry point
if __name__ == "__main__":
    try:
        print("\nUsing matplotlib display...")
        success = run_with_matplotlib()
        
        if not success:
            print("\nMatplotlib failed, trying ffplay...")
            _running = True  # Reset flag
            run_with_ffplay()
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        release_camera()
        print("\n[ok] All camera resources released")
