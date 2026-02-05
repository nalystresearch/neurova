# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Chapter 10: Video Processing


This chapter covers:
- Video capture and reading
- Frame processing
- Video writing
- Motion detection
- Video analysis pipelines
- Webcam integration

Using Neurova's video processing tools!

Author: Neurova Team
"""

import numpy as np
from pathlib import Path

print("")
print("Chapter 10: Video Processing")
print("")

import neurova as nv

# 10.1 videocapture class overview
print(f"\n10.1 VideoCapture Class")

from neurova.video import VideoCapture

print("    VideoCapture provides:")
print("      - Video file reading")
print("      - Webcam capture")
print("      - Frame-by-frame processing")
print("      - Property access (fps, dimensions)")

# 10.2 opening video files
print(f"\n10.2 Opening Video Files")

# Note: This demonstrates the API - actual video file required
print("""
# open video file
    cap = VideoCapture('video.mp4')
    
# check if opened
    if cap.isOpened():
        print("Video opened successfully")
    
# get properties
    fps = cap.get('fps')
    width = cap.get('width')
    height = cap.get('height')
    frame_count = cap.get('frame_count')
    
    print(f"Video: {width}x{height} @ {fps} fps, {frame_count} frames")
""")

# 10.3 reading frames
print(f"\n10.3 Reading Frames")

print("""
# read single frame
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape}")
    
# read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    print(f"Read {len(frames)} frames")
""")

# 10.4 webcam capture
print(f"\n10.4 Webcam Capture")

print("""
    # Open webcam (device 0)
    webcam = VideoCapture(0)
    
# set resolution
    webcam.set('width', 640)
    webcam.set('height', 480)
    
# capture frames
    for i in range(10):
        ret, frame = webcam.read()
        if ret:
            process_frame(frame)
    
# release
    webcam.release()
""")

# 10.5 frame processing pipeline
print(f"\n10.5 Frame Processing Pipeline")

def process_frame(frame):
    """Example frame processing pipeline."""
    # 1. Convert to grayscale
    if len(frame.shape) == 3:
        gray = np.mean(frame, axis=2).astype(np.uint8)
    else:
        gray = frame
    
    # 2. Apply Gaussian blur
    from neurova.filters import gaussian_blur
    blurred = gaussian_blur(gray, kernel_size=5, sigma=1.0)
    
    # 3. Edge detection
    from neurova.filters import sobel, gradient_magnitude
    gx, gy = sobel(blurred)
    edges = gradient_magnitude(gx, gy)
    
    # 4. Threshold
    binary = (edges > 50).astype(np.uint8) * 255
    
    return binary

# demo with synthetic frame
synthetic_frame = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
processed = process_frame(synthetic_frame)
print(f"    Synthetic frame: {synthetic_frame.shape}")
print(f"    Processed: {processed.shape}")

# 10.6 video writing
print(f"\n10.6 Video Writing")

from neurova.video import VideoWriter

print("""
# create video writer
    writer = VideoWriter(
        'output.mp4',
        width=640,
        height=480,
        fps=30
    )
    
# write frames
    for frame in frames:
        writer.write(frame)
    
# release
    writer.release()
""")

# 10.7 motion detection
print(f"\n10.7 Motion Detection")

class MotionDetector:
    """Simple motion detection using frame differencing."""
    
    def __init__(self, threshold=25, min_area=500):
        self.threshold = threshold
        self.min_area = min_area
        self.prev_frame = None
    
    def detect(self, frame):
        """Detect motion between current and previous frame."""
# convert to grayscale
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2).astype(np.uint8)
        else:
            gray = frame
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return [], np.zeros_like(gray)
        
# frame difference
        diff = np.abs(gray.astype(np.int16) - self.prev_frame.astype(np.int16))
        diff = diff.astype(np.uint8)
        
# threshold
        motion_mask = (diff > self.threshold).astype(np.uint8) * 255
        
        # Find motion regions (simplified)
        motion_pixels = np.sum(motion_mask > 0)
        motion_boxes = []
        
        if motion_pixels > self.min_area:
# find bounding box of motion
            rows = np.any(motion_mask, axis=1)
            cols = np.any(motion_mask, axis=0)
            
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                motion_boxes.append((cmin, rmin, cmax - cmin, rmax - rmin))
        
        self.prev_frame = gray
        return motion_boxes, motion_mask

# demo motion detection
motion_detector = MotionDetector(threshold=30, min_area=100)

# create two frames with motion
frame1 = np.zeros((100, 100), dtype=np.uint8)
frame1[30:50, 30:50] = 200  # Object at position 1

frame2 = np.zeros((100, 100), dtype=np.uint8)
frame2[40:60, 40:60] = 200  # Object moved

# detect motion
boxes1, mask1 = motion_detector.detect(frame1)
boxes2, mask2 = motion_detector.detect(frame2)

print(f"    Frame 1: {len(boxes1)} motion regions")
print(f"    Frame 2: {len(boxes2)} motion regions")
if boxes2:
    print(f"      Motion box: {boxes2[0]}")

# 10.8 background subtraction
print(f"\n10.8 Background Subtraction")

class BackgroundSubtractor:
    """Running average background subtractor."""
    
    def __init__(self, alpha=0.05, threshold=30):
        self.alpha = alpha  # Learning rate
        self.threshold = threshold
        self.background = None
    
    def apply(self, frame):
        """Apply background subtraction."""
# convert to grayscale
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        
        if self.background is None:
            self.background = gray.copy()
            return np.zeros_like(gray, dtype=np.uint8)
        
# update background
        self.background = self.alpha * gray + (1 - self.alpha) * self.background
        
# compute foreground mask
        diff = np.abs(gray - self.background)
        fg_mask = (diff > self.threshold).astype(np.uint8) * 255
        
        return fg_mask

# demo background subtraction
bg_subtractor = BackgroundSubtractor(alpha=0.1, threshold=25)

# process frames
for i in range(5):
    frame = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
    if i >= 3:  # Add foreground object
        frame[40:60, 40:60] = 250
    
    fg_mask = bg_subtractor.apply(frame)

print(f"    Background subtractor initialized")
print(f"    Learning rate: {bg_subtractor.alpha}")
print(f"    Threshold: {bg_subtractor.threshold}")

# 10.9 optical flow (simplified)
print(f"\n10.9 Optical Flow (Block Matching)")

def block_matching_flow(prev_frame, curr_frame, block_size=16, search_range=8):
    """Simplified optical flow using block matching."""
    h, w = prev_frame.shape
    flow_u = np.zeros((h // block_size, w // block_size))
    flow_v = np.zeros((h // block_size, w // block_size))
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = prev_frame[i:i+block_size, j:j+block_size]
            
            best_match = (0, 0)
            best_sad = float('inf')
            
# search in neighborhood
            for di in range(-search_range, search_range + 1):
                for dj in range(-search_range, search_range + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h - block_size and 0 <= nj < w - block_size:
                        candidate = curr_frame[ni:ni+block_size, nj:nj+block_size]
                        sad = np.sum(np.abs(block.astype(np.int16) - candidate.astype(np.int16)))
                        
                        if sad < best_sad:
                            best_sad = sad
                            best_match = (di, dj)
            
            bi = i // block_size
            bj = j // block_size
            flow_v[bi, bj] = best_match[0]
            flow_u[bi, bj] = best_match[1]
    
    return flow_u, flow_v

# demo optical flow
prev = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
# shift frame to simulate motion
curr = np.roll(prev, shift=2, axis=1)  # Shift right by 2

flow_u, flow_v = block_matching_flow(prev, curr, block_size=8, search_range=4)
print(f"    Flow field shape: {flow_u.shape}")
print(f"    Mean horizontal flow: {flow_u.mean():.2f}")
print(f"    Mean vertical flow: {flow_v.mean():.2f}")

# 10.10 frame rate control
print(f"\n10.10 Frame Rate Control")

import time

class FrameRateController:
    """Control frame processing rate."""
    
    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.last_time = None
        self.frame_count = 0
        self.start_time = None
    
    def tick(self):
        """Call at start of each frame."""
        current_time = time.time()
        
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
        
        if self.start_time is None:
            self.start_time = current_time
        
        self.last_time = time.time()
        self.frame_count += 1
    
    def get_actual_fps(self):
        """Get actual achieved FPS."""
        if self.start_time is None or self.frame_count == 0:
            return 0
        elapsed = time.time() - self.start_time
        return self.frame_count / elapsed if elapsed > 0 else 0

# demo frame rate control
fps_controller = FrameRateController(target_fps=30)
print(f"    Target FPS: {fps_controller.target_fps}")
print(f"    Frame time: {fps_controller.frame_time*1000:.1f} ms")

# 10.11 video analysis pipeline
print(f"\n10.11 Video Analysis Pipeline")

class VideoAnalyzer:
    """Complete video analysis pipeline."""
    
    def __init__(self, detect_motion=True, detect_faces=False):
        self.detect_motion = detect_motion
        self.detect_faces = detect_faces
        self.motion_detector = MotionDetector() if detect_motion else None
        self.frame_count = 0
        self.results = []
    
    def process(self, frame):
        """Process a single frame."""
        self.frame_count += 1
        
        result = {
            'frame': self.frame_count,
            'shape': frame.shape
        }
        
# motion detection
        if self.motion_detector:
            motion_boxes, _ = self.motion_detector.detect(frame)
            result['motion'] = len(motion_boxes) > 0
            result['motion_boxes'] = motion_boxes
        
        self.results.append(result)
        return result
    
    def get_summary(self):
        """Get analysis summary."""
        motion_frames = sum(1 for r in self.results if r.get('motion', False))
        return {
            'total_frames': self.frame_count,
            'motion_frames': motion_frames,
            'motion_ratio': motion_frames / self.frame_count if self.frame_count > 0 else 0
        }

# demo pipeline
analyzer = VideoAnalyzer(detect_motion=True)

# process synthetic video
for i in range(10):
    frame = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
    if i % 3 == 0:  # Add motion every 3rd frame
        frame[30:50, 30:50] = 250
    analyzer.process(frame)

summary = analyzer.get_summary()
print(f"    Processed frames: {summary['total_frames']}")
print(f"    Motion frames: {summary['motion_frames']}")
print(f"    Motion ratio: {summary['motion_ratio']:.1%}")

# 10.12 real-time processing tips
print(f"\n10.12 Real-time Processing Tips")

print("""
    Tips for real-time video processing:
    
    1. Resize frames to smaller resolution:
       small = resize(frame, (320, 240))
    
    2. Process every Nth frame:
       if frame_count % 3 == 0:
           result = expensive_operation(frame)
    
    3. Use ROI (Region of Interest):
       roi = frame[y:y+h, x:x+w]
       result = process(roi)
    
    4. Cache intermediate results:
       if not cache_valid:
           cached_result = compute()
    
    5. Use appropriate data types:
       gray = frame.astype(np.uint8)  # Not float64
    
    6. Enable GPU acceleration:
       nv.device.set('gpu')
""")

# 10.13 video recording utility
print(f"\n10.13 Video Recording Utility")

class VideoRecorder:
    """Utility for recording video with timestamps."""
    
    def __init__(self, output_path, fps=30, resolution=(640, 480)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.timestamps = []
        self.start_time = None
    
    def start(self):
        """Start recording."""
        self.start_time = time.time()
        self.frames = []
        self.timestamps = []
    
    def add_frame(self, frame):
        """Add a frame to recording."""
        if self.start_time is None:
            self.start()
        
        self.frames.append(frame.copy())
        self.timestamps.append(time.time() - self.start_time)
    
    def stop(self):
        """Stop recording and return stats."""
        if not self.frames:
            return None
        
        duration = self.timestamps[-1] if self.timestamps else 0
        actual_fps = len(self.frames) / duration if duration > 0 else 0
        
        return {
            'frames': len(self.frames),
            'duration': duration,
            'actual_fps': actual_fps
        }

# demo recorder
recorder = VideoRecorder('test.mp4', fps=30)
recorder.start()

for i in range(30):
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    recorder.add_frame(frame)

stats = recorder.stop()
print(f"    Recorded: {stats['frames']} frames")
print(f"    Duration: {stats['duration']:.2f} seconds")

# summary
print("\n" + "=" * 60)
print("Chapter 10 Summary:")
print("   Learned VideoCapture for reading video/webcam")
print("   Processed frames through pipelines")
print("   Implemented motion detection")
print("   Applied background subtraction")
print("   Computed optical flow (block matching)")
print("   Controlled frame rate")
print("   Built video analysis pipeline")
print("   Created video recording utility")
print("")
