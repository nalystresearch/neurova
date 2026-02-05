// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/video.hpp - Video Capture and Processing
 * 
 * This header provides video functionality:
 * - Video capture from cameras and files
 * - Video writing
 * - Frame processing
 * - Display
 * - Background subtraction (MOG2, KNN)
 * - Optical flow (Lucas-Kanade, Farneback)
 * - Object tracking (MeanShift, CamShift, KalmanFilter)
 * - Object trackers (MIL, KCF, CSRT)
 */

#ifndef NEUROVA_VIDEO_HPP
#define NEUROVA_VIDEO_HPP

#include "core.hpp"
#include "video/background.hpp"
#include "video/optflow.hpp"
#include "video/tracking.hpp"
#include "video/trackers.hpp"
#include <string>
#include <functional>

namespace neurova {
namespace video {

// ============================================================================
// Video Properties
// ============================================================================

enum class FourCC {
    MJPG,
    XVID,
    H264,
    MP4V,
    DIVX,
    RAW
};

struct VideoProperties {
    int width;
    int height;
    double fps;
    int frame_count;
    double duration;      // in seconds
    std::string codec;
    
    VideoProperties() : width(0), height(0), fps(0), frame_count(0), duration(0) {}
};

// ============================================================================
// Video Capture
// ============================================================================

class VideoCapture {
public:
    VideoCapture();
    VideoCapture(int device);
    VideoCapture(const std::string& filename);
    ~VideoCapture();
    
    // Open camera or file
    bool open(int device);
    bool open(const std::string& filename);
    
    // Check if opened
    bool isOpened() const;
    
    // Close
    void release();
    
    // Read frame
    bool read(Image& frame);
    Image read();
    
    // Seek (for files only)
    bool set(int prop_id, double value);
    double get(int prop_id) const;
    
    // Properties
    int frameWidth() const;
    int frameHeight() const;
    double fps() const;
    int frameCount() const;
    int currentFrame() const;
    VideoProperties properties() const;
    
    // Set resolution (for cameras)
    bool setResolution(int width, int height);
    bool setFps(double fps);
    
    // Seek to frame (for files)
    bool seekFrame(int frame_number);
    bool seekTime(double time_seconds);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Video Writer
// ============================================================================

class VideoWriter {
public:
    VideoWriter();
    VideoWriter(const std::string& filename, FourCC codec,
                double fps, int width, int height, bool is_color = true);
    ~VideoWriter();
    
    // Open file for writing
    bool open(const std::string& filename, FourCC codec,
              double fps, int width, int height, bool is_color = true);
    
    // Check if opened
    bool isOpened() const;
    
    // Close
    void release();
    
    // Write frame
    void write(const Image& frame);
    VideoWriter& operator<<(const Image& frame);
    
    // Properties
    std::string filename() const;
    int width() const;
    int height() const;
    double fps() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Frame Processing
// ============================================================================

// Frame buffer for temporal operations
class FrameBuffer {
public:
    FrameBuffer(size_t max_frames = 30);
    
    void push(const Image& frame);
    Image get(size_t index) const;
    Image latest() const;
    Image oldest() const;
    
    size_t size() const;
    size_t maxSize() const;
    void clear();
    
    // Temporal operations
    Image average() const;
    Image median() const;
    Image difference(size_t idx1, size_t idx2) const;
    
private:
    std::vector<Image> buffer_;
    size_t max_frames_;
    size_t head_;
};

// Motion detection
class MotionDetector {
public:
    MotionDetector(double threshold = 25.0, int min_area = 500);
    
    // Process frame and detect motion
    bool detect(const Image& frame);
    
    // Get motion mask
    Image getMotionMask() const;
    
    // Get motion regions
    std::vector<std::tuple<int, int, int, int>> getMotionRegions() const;
    
    // Reset background model
    void reset();
    
    // Parameters
    void setThreshold(double threshold);
    void setMinArea(int min_area);

private:
    Image background_;
    Image motion_mask_;
    double threshold_;
    int min_area_;
    bool initialized_;
};

// Background subtraction
class BackgroundSubtractor {
public:
    BackgroundSubtractor(int history = 500, double var_threshold = 16.0,
                         bool detect_shadows = true);
    
    // Apply to frame
    Image apply(const Image& frame, double learning_rate = -1);
    
    // Get background model
    Image getBackgroundImage() const;
    
    // Reset
    void reset();

private:
    int history_;
    double var_threshold_;
    bool detect_shadows_;
    
    Tensor mean_;
    Tensor variance_;
    int frame_count_;
};

// Optical flow
class OpticalFlow {
public:
    enum class Method {
        FARNEBACK,
        LUCAS_KANADE
    };
    
    OpticalFlow(Method method = Method::FARNEBACK);
    
    // Compute flow between two frames
    std::pair<Image, Image> compute(const Image& prev, const Image& next);
    
    // Visualize flow
    Image visualize(const Image& flow_x, const Image& flow_y) const;
    
    // Parameters for Farneback
    void setFarnebackParams(int pyr_scale = 2, int levels = 5,
                            int winsize = 13, int iterations = 10,
                            int poly_n = 5, double poly_sigma = 1.1);

private:
    Method method_;
    int pyr_scale_, levels_, winsize_, iterations_, poly_n_;
    double poly_sigma_;
};

// Video stabilization
class VideoStabilizer {
public:
    VideoStabilizer(int smoothing_radius = 30);
    
    // Process frame
    Image stabilize(const Image& frame);
    
    // Reset
    void reset();

private:
    int smoothing_radius_;
    std::vector<Tensor> transforms_;
    Image prev_gray_;
    std::vector<std::pair<double, double>> prev_points_;
};

// ============================================================================
// Display (using FFplay or system display)
// ============================================================================

class Display {
public:
    Display(const std::string& window_name = "Neurova Display");
    ~Display();
    
    // Show image
    void show(const Image& image);
    
    // Wait for key
    int waitKey(int delay_ms = 0);
    
    // Check if window is open
    bool isOpen() const;
    
    // Close window
    void close();
    
    // Window properties
    void setTitle(const std::string& title);
    void resize(int width, int height);
    void move(int x, int y);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// Simple display function
void imshow(const std::string& window_name, const Image& image);
int waitKey(int delay_ms = 0);
void destroyAllWindows();
void destroyWindow(const std::string& window_name);

// ============================================================================
// Video Processing Pipeline
// ============================================================================

using FrameProcessor = std::function<Image(const Image&)>;
using FrameCallback = std::function<void(const Image&, int)>;

class VideoPipeline {
public:
    VideoPipeline();
    
    // Add processing step
    VideoPipeline& addProcessor(FrameProcessor processor);
    
    // Clear processors
    void clearProcessors();
    
    // Process video file
    void processFile(const std::string& input_file, 
                     const std::string& output_file = "",
                     FrameCallback callback = nullptr);
    
    // Process camera
    void processCamera(int device = 0, 
                       FrameCallback callback = nullptr,
                       int max_frames = -1);
    
    // Stop processing
    void stop();
    
    // Is running
    bool isRunning() const;

private:
    std::vector<FrameProcessor> processors_;
    bool running_;
};

// ============================================================================
// Utility Functions
// ============================================================================

// FourCC conversion
std::string fourccToString(FourCC code);
FourCC stringToFourCC(const std::string& str);
int fourccToInt(FourCC code);

// Frame rate control
class FrameRateController {
public:
    FrameRateController(double target_fps);
    
    void wait();
    void reset();
    
    double actualFps() const;
    void setTargetFps(double fps);

private:
    double target_fps_;
    std::chrono::steady_clock::time_point last_frame_;
    double actual_fps_;
    int frame_count_;
};

// Video info
VideoProperties getVideoInfo(const std::string& filename);

// Capture from URL (RTSP, HTTP)
VideoCapture captureFromUrl(const std::string& url);

// Screenshot
Image grabScreen(int x = 0, int y = 0, int width = 0, int height = 0);

} // namespace video
} // namespace neurova

#endif // NEUROVA_VIDEO_HPP
