/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova V4L2 Camera Capture Backend
 * Linux Video4Linux2 camera capture implementation
 */

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstring>

#ifdef __linux__

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <errno.h>

namespace neurova {
namespace videoio {
namespace v4l2 {

// Buffer structure for mmap
struct Buffer {
    void* start;
    size_t length;
};

// Pixel format info
struct PixelFormat {
    uint32_t fourcc;
    std::string name;
    int bytes_per_pixel;
    bool is_compressed;
};

// Frame callback type
using FrameCallback = std::function<void(const uint8_t* data, int width, int height, 
                                         int channels, int64_t timestamp)>;

// V4L2 Camera class
class V4L2Camera {
public:
    V4L2Camera() : fd_(-1), streaming_(false), capture_thread_running_(false) {
        buffers_.resize(4);  // Use 4 buffers by default
    }
    
    ~V4L2Camera() {
        close();
    }
    
    // Open camera device
    bool open(const std::string& device = "/dev/video0") {
        device_path_ = device;
        
        fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) {
            error_ = "Failed to open device: " + device;
            return false;
        }
        
        // Query capabilities
        struct v4l2_capability cap;
        if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            error_ = "VIDIOC_QUERYCAP failed";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            error_ = "Device doesn't support video capture";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            error_ = "Device doesn't support streaming I/O";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        
        driver_name_ = reinterpret_cast<char*>(cap.driver);
        card_name_ = reinterpret_cast<char*>(cap.card);
        
        return true;
    }
    
    // Close camera
    void close() {
        stopCapture();
        
        if (fd_ >= 0) {
            // Unmap buffers
            for (auto& buf : buffers_) {
                if (buf.start != MAP_FAILED && buf.start != nullptr) {
                    munmap(buf.start, buf.length);
                    buf.start = nullptr;
                }
            }
            
            ::close(fd_);
            fd_ = -1;
        }
    }
    
    // Check if open
    bool isOpened() const { return fd_ >= 0; }
    
    // Get/set resolution
    bool setResolution(int width, int height) {
        if (!isOpened()) return false;
        
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) {
            return false;
        }
        
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            return false;
        }
        
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;
        pixel_format_ = fmt.fmt.pix.pixelformat;
        
        return true;
    }
    
    void getResolution(int& width, int& height) const {
        width = width_;
        height = height_;
    }
    
    // Set pixel format
    bool setPixelFormat(uint32_t fourcc) {
        if (!isOpened()) return false;
        
        struct v4l2_format fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        if (ioctl(fd_, VIDIOC_G_FMT, &fmt) < 0) {
            return false;
        }
        
        fmt.fmt.pix.pixelformat = fourcc;
        
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            return false;
        }
        
        pixel_format_ = fmt.fmt.pix.pixelformat;
        
        return true;
    }
    
    // Set frame rate
    bool setFrameRate(int fps) {
        if (!isOpened()) return false;
        
        struct v4l2_streamparm parm = {};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = fps;
        
        if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
            return false;
        }
        
        fps_ = fps;
        return true;
    }
    
    // Enumerate supported formats
    std::vector<PixelFormat> getSupportedFormats() {
        std::vector<PixelFormat> formats;
        
        if (!isOpened()) return formats;
        
        struct v4l2_fmtdesc fmtdesc = {};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        
        while (ioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
            PixelFormat pf;
            pf.fourcc = fmtdesc.pixelformat;
            pf.name = reinterpret_cast<char*>(fmtdesc.description);
            pf.is_compressed = (fmtdesc.flags & V4L2_FMT_FLAG_COMPRESSED) != 0;
            
            // Estimate bytes per pixel
            switch (fmtdesc.pixelformat) {
                case V4L2_PIX_FMT_YUYV:
                case V4L2_PIX_FMT_UYVY:
                    pf.bytes_per_pixel = 2;
                    break;
                case V4L2_PIX_FMT_RGB24:
                case V4L2_PIX_FMT_BGR24:
                    pf.bytes_per_pixel = 3;
                    break;
                case V4L2_PIX_FMT_RGB32:
                case V4L2_PIX_FMT_BGR32:
                    pf.bytes_per_pixel = 4;
                    break;
                default:
                    pf.bytes_per_pixel = 0;
            }
            
            formats.push_back(pf);
            fmtdesc.index++;
        }
        
        return formats;
    }
    
    // Enumerate supported resolutions for a format
    struct Resolution {
        int width;
        int height;
    };
    
    std::vector<Resolution> getSupportedResolutions(uint32_t format) {
        std::vector<Resolution> resolutions;
        
        if (!isOpened()) return resolutions;
        
        struct v4l2_frmsizeenum frmsize = {};
        frmsize.pixel_format = format;
        
        while (ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
            if (frmsize.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                Resolution res;
                res.width = frmsize.discrete.width;
                res.height = frmsize.discrete.height;
                resolutions.push_back(res);
            } else if (frmsize.type == V4L2_FRMSIZE_TYPE_STEPWISE) {
                // Add common resolutions within range
                for (int w = frmsize.stepwise.min_width; 
                     w <= (int)frmsize.stepwise.max_width; 
                     w += frmsize.stepwise.step_width) {
                    for (int h = frmsize.stepwise.min_height;
                         h <= (int)frmsize.stepwise.max_height;
                         h += frmsize.stepwise.step_height) {
                        Resolution res;
                        res.width = w;
                        res.height = h;
                        resolutions.push_back(res);
                    }
                }
            }
            frmsize.index++;
        }
        
        return resolutions;
    }
    
    // Initialize memory-mapped buffers
    bool initBuffers(int num_buffers = 4) {
        if (!isOpened()) return false;
        
        // Request buffers
        struct v4l2_requestbuffers req = {};
        req.count = num_buffers;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            error_ = "VIDIOC_REQBUFS failed";
            return false;
        }
        
        if (req.count < 2) {
            error_ = "Insufficient buffer memory";
            return false;
        }
        
        buffers_.resize(req.count);
        
        // Map buffers
        for (size_t i = 0; i < buffers_.size(); i++) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                error_ = "VIDIOC_QUERYBUF failed";
                return false;
            }
            
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length,
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     fd_, buf.m.offset);
            
            if (buffers_[i].start == MAP_FAILED) {
                error_ = "mmap failed";
                return false;
            }
        }
        
        return true;
    }
    
    // Start capture
    bool startCapture() {
        if (!isOpened() || streaming_) return false;
        
        // Queue all buffers
        for (size_t i = 0; i < buffers_.size(); i++) {
            struct v4l2_buffer buf = {};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                error_ = "VIDIOC_QBUF failed";
                return false;
            }
        }
        
        // Start streaming
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            error_ = "VIDIOC_STREAMON failed";
            return false;
        }
        
        streaming_ = true;
        return true;
    }
    
    // Stop capture
    void stopCapture() {
        if (!streaming_) return;
        
        // Stop capture thread if running
        if (capture_thread_running_) {
            capture_thread_running_ = false;
            if (capture_thread_.joinable()) {
                capture_thread_.join();
            }
        }
        
        // Stop streaming
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        
        streaming_ = false;
    }
    
    // Read a single frame (synchronous)
    bool readFrame(std::vector<uint8_t>& data, int64_t& timestamp) {
        if (!streaming_) return false;
        
        // Wait for frame with select
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        
        struct timeval tv;
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        
        int r = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) {
            error_ = r == 0 ? "Timeout waiting for frame" : "Select error";
            return false;
        }
        
        // Dequeue buffer
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) return false;
            error_ = "VIDIOC_DQBUF failed";
            return false;
        }
        
        // Copy data
        data.resize(buf.bytesused);
        memcpy(data.data(), buffers_[buf.index].start, buf.bytesused);
        
        // Get timestamp
        timestamp = buf.timestamp.tv_sec * 1000000LL + buf.timestamp.tv_usec;
        
        // Re-queue buffer
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            error_ = "VIDIOC_QBUF failed";
            return false;
        }
        
        return true;
    }
    
    // Start asynchronous capture with callback
    bool startAsyncCapture(FrameCallback callback) {
        if (!streaming_ || capture_thread_running_) return false;
        
        frame_callback_ = callback;
        capture_thread_running_ = true;
        
        capture_thread_ = std::thread([this]() {
            std::vector<uint8_t> data;
            std::vector<uint8_t> rgb_data;
            
            while (capture_thread_running_) {
                int64_t timestamp;
                if (readFrame(data, timestamp)) {
                    // Convert to RGB if needed
                    int channels = 3;
                    convertToRGB(data.data(), data.size(), rgb_data);
                    
                    if (frame_callback_) {
                        frame_callback_(rgb_data.data(), width_, height_, 
                                       channels, timestamp);
                    }
                }
            }
        });
        
        return true;
    }
    
    // Get camera controls
    struct Control {
        uint32_t id;
        std::string name;
        int32_t minimum;
        int32_t maximum;
        int32_t default_value;
        int32_t step;
    };
    
    std::vector<Control> getControls() {
        std::vector<Control> controls;
        
        if (!isOpened()) return controls;
        
        // Standard controls
        struct v4l2_queryctrl queryctrl = {};
        queryctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
        
        while (ioctl(fd_, VIDIOC_QUERYCTRL, &queryctrl) == 0) {
            if (!(queryctrl.flags & V4L2_CTRL_FLAG_DISABLED)) {
                Control ctrl;
                ctrl.id = queryctrl.id;
                ctrl.name = reinterpret_cast<char*>(queryctrl.name);
                ctrl.minimum = queryctrl.minimum;
                ctrl.maximum = queryctrl.maximum;
                ctrl.default_value = queryctrl.default_value;
                ctrl.step = queryctrl.step;
                controls.push_back(ctrl);
            }
            queryctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
        }
        
        return controls;
    }
    
    // Get/set control value
    int32_t getControl(uint32_t id) {
        struct v4l2_control ctrl = {};
        ctrl.id = id;
        
        if (ioctl(fd_, VIDIOC_G_CTRL, &ctrl) < 0) {
            return -1;
        }
        
        return ctrl.value;
    }
    
    bool setControl(uint32_t id, int32_t value) {
        struct v4l2_control ctrl = {};
        ctrl.id = id;
        ctrl.value = value;
        
        return ioctl(fd_, VIDIOC_S_CTRL, &ctrl) >= 0;
    }
    
    // Common controls
    bool setBrightness(int value) { return setControl(V4L2_CID_BRIGHTNESS, value); }
    bool setContrast(int value) { return setControl(V4L2_CID_CONTRAST, value); }
    bool setSaturation(int value) { return setControl(V4L2_CID_SATURATION, value); }
    bool setExposure(int value) { return setControl(V4L2_CID_EXPOSURE, value); }
    bool setGain(int value) { return setControl(V4L2_CID_GAIN, value); }
    
    bool setAutoExposure(bool enabled) {
        return setControl(V4L2_CID_EXPOSURE_AUTO, 
                         enabled ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL);
    }
    
    bool setAutoWhiteBalance(bool enabled) {
        return setControl(V4L2_CID_AUTO_WHITE_BALANCE, enabled ? 1 : 0);
    }
    
    bool setAutoFocus(bool enabled) {
        return setControl(V4L2_CID_FOCUS_AUTO, enabled ? 1 : 0);
    }
    
    // Get device info
    std::string getDriverName() const { return driver_name_; }
    std::string getCardName() const { return card_name_; }
    std::string getDevicePath() const { return device_path_; }
    std::string getLastError() const { return error_; }

private:
    // Convert YUYV to RGB
    void convertToRGB(const uint8_t* src, size_t src_size, std::vector<uint8_t>& dst) {
        int num_pixels = width_ * height_;
        dst.resize(num_pixels * 3);
        
        if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
            for (int i = 0; i < num_pixels / 2; i++) {
                int y0 = src[i * 4 + 0];
                int u  = src[i * 4 + 1];
                int y1 = src[i * 4 + 2];
                int v  = src[i * 4 + 3];
                
                // YUV to RGB conversion
                int c0 = y0 - 16;
                int c1 = y1 - 16;
                int d = u - 128;
                int e = v - 128;
                
                auto clamp = [](int x) -> uint8_t {
                    return x < 0 ? 0 : (x > 255 ? 255 : x);
                };
                
                dst[i * 6 + 0] = clamp((298 * c0 + 409 * e + 128) >> 8);
                dst[i * 6 + 1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
                dst[i * 6 + 2] = clamp((298 * c0 + 516 * d + 128) >> 8);
                
                dst[i * 6 + 3] = clamp((298 * c1 + 409 * e + 128) >> 8);
                dst[i * 6 + 4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
                dst[i * 6 + 5] = clamp((298 * c1 + 516 * d + 128) >> 8);
            }
        } else if (pixel_format_ == V4L2_PIX_FMT_RGB24) {
            memcpy(dst.data(), src, num_pixels * 3);
        } else if (pixel_format_ == V4L2_PIX_FMT_BGR24) {
            for (int i = 0; i < num_pixels; i++) {
                dst[i * 3 + 0] = src[i * 3 + 2];
                dst[i * 3 + 1] = src[i * 3 + 1];
                dst[i * 3 + 2] = src[i * 3 + 0];
            }
        }
    }

private:
    int fd_;
    std::string device_path_;
    std::string driver_name_;
    std::string card_name_;
    std::string error_;
    
    int width_ = 640;
    int height_ = 480;
    int fps_ = 30;
    uint32_t pixel_format_ = V4L2_PIX_FMT_YUYV;
    
    std::vector<Buffer> buffers_;
    std::atomic<bool> streaming_;
    
    std::thread capture_thread_;
    std::atomic<bool> capture_thread_running_;
    FrameCallback frame_callback_;
};

// Enumerate available cameras
inline std::vector<std::string> enumerateCameras() {
    std::vector<std::string> cameras;
    
    for (int i = 0; i < 10; i++) {
        std::string path = "/dev/video" + std::to_string(i);
        int fd = ::open(path.c_str(), O_RDWR);
        if (fd >= 0) {
            struct v4l2_capability cap;
            if (ioctl(fd, VIDIOC_QUERYCAP, &cap) >= 0) {
                if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) {
                    cameras.push_back(path);
                }
            }
            ::close(fd);
        }
    }
    
    return cameras;
}

} // namespace v4l2
} // namespace videoio
} // namespace neurova

#endif // __linux__
