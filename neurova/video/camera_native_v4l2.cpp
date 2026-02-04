/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

#ifdef __linux__
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>
#include <cerrno>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

inline int xioctl(int fd, unsigned long request, void* arg) {
    int r;
    do {
        r = ioctl(fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r;
}

struct Buffer {
    void* start = nullptr;
    size_t length = 0;
};

static void yuyv_to_bgr(const uint8_t* src, uint8_t* dst, int width, int height) {
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    auto clamp = [](int val) {
        return static_cast<uint8_t>(val < 0 ? 0 : (val > 255 ? 255 : val));
    };

    for (size_t i = 0, j = 0; i < pixels; i += 2, j += 4) {
        const int y0 = src[j + 0] - 16;
        const int u = src[j + 1] - 128;
        const int y1 = src[j + 2] - 16;
        const int v = src[j + 3] - 128;

        const int c0 = 298 * y0 + 128;
        const int c1 = 298 * y1 + 128;
        const int d = 516 * u;
        const int e = 409 * v;
        const int f = -100 * u - 208 * v;

        const uint8_t b0 = clamp((c0 + d) >> 8);
        const uint8_t g0 = clamp((c0 + f) >> 8);
        const uint8_t r0 = clamp((c0 + e) >> 8);

        const uint8_t b1 = clamp((c1 + d) >> 8);
        const uint8_t g1 = clamp((c1 + f) >> 8);
        const uint8_t r1 = clamp((c1 + e) >> 8);

        dst[i * 3 + 0] = r0;
        dst[i * 3 + 1] = g0;
        dst[i * 3 + 2] = b0;

        dst[i * 3 + 3] = r1;
        dst[i * 3 + 4] = g1;
        dst[i * 3 + 5] = b1;
    }
}

}  // namespace

class CameraCaptureV4L2 {
public:
    CameraCaptureV4L2(int device = 0, int width = 640, int height = 480, int fps = 30)
        : device_index_(device), width_(width), height_(height), fps_(fps) {}

    ~CameraCaptureV4L2() { release(); }

    bool open() {
        if (opened_) {
            return true;
        }

        const std::string device_path = "/dev/video" + std::to_string(device_index_);
        fd_ = ::open(device_path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) {
            last_error_ = "Failed to open " + device_path;
            return false;
        }

        v4l2_capability cap{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            last_error_ = "VIDIOC_QUERYCAP failed";
            cleanup();
            return false;
        }
        if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
            last_error_ = "Device does not support video capture";
            cleanup();
            return false;
        }

        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            last_error_ = "VIDIOC_S_FMT failed";
            cleanup();
            return false;
        }
        width_ = fmt.fmt.pix.width;
        height_ = fmt.fmt.pix.height;

        v4l2_streamparm parm{};
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator = 1;
        parm.parm.capture.timeperframe.denominator = fps_;
        xioctl(fd_, VIDIOC_S_PARM, &parm);

        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            last_error_ = "VIDIOC_REQBUFS failed";
            cleanup();
            return false;
        }
        if (req.count < 2) {
            last_error_ = "Insufficient buffer memory";
            cleanup();
            return false;
        }

        buffers_.resize(req.count);
        for (size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<uint32_t>(i);
            if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                last_error_ = "VIDIOC_QUERYBUF failed";
                cleanup();
                return false;
            }
            buffers_[i].length = buf.length;
            buffers_[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED) {
                last_error_ = "mmap failed";
                cleanup();
                return false;
            }
        }

        for (size_t i = 0; i < buffers_.size(); ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = static_cast<uint32_t>(i);
            if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                last_error_ = "VIDIOC_QBUF failed";
                cleanup();
                return false;
            }
        }

        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            last_error_ = "VIDIOC_STREAMON failed";
            cleanup();
            return false;
        }

        streaming_ = true;
        opened_ = true;
        return true;
    }

    py::array_t<uint8_t> read() {
        if (!opened_) {
            if (!open()) {
                return py::array_t<uint8_t>();
            }
        }

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd_, &fds);
        timeval tv{};
        tv.tv_sec = 2;
        int r = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
        if (r == -1) {
            return py::array_t<uint8_t>();
        } else if (r == 0) {
            last_error_ = "select timeout";
            return py::array_t<uint8_t>();
        }

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            return py::array_t<uint8_t>();
        }

        const uint8_t* src = static_cast<uint8_t*>(buffers_[buf.index].start);
        py::array_t<uint8_t> frame({height_, width_, 3});
        py::buffer_info info = frame.request();
        auto* dst = static_cast<uint8_t*>(info.ptr);
        yuyv_to_bgr(src, dst, width_, height_);

        if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            last_error_ = "VIDIOC_QBUF failed";
        }
        return frame;
    }

    void release() {
        cleanup();
        opened_ = false;
    }

    bool isOpened() const { return opened_; }
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    std::string last_error() const { return last_error_; }

private:
    int device_index_ = 0;
    int width_ = 640;
    int height_ = 480;
    int fps_ = 30;
    int fd_ = -1;
    bool opened_ = false;
    bool streaming_ = false;
    std::string last_error_;
    std::vector<Buffer> buffers_;

    void cleanup() {
        if (streaming_ && fd_ >= 0) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(fd_, VIDIOC_STREAMOFF, &type);
        }
        streaming_ = false;
        for (auto& buf : buffers_) {
            if (buf.start && buf.length) {
                munmap(buf.start, buf.length);
            }
        }
        buffers_.clear();
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
};

PYBIND11_MODULE(camera_native, m) {
    py::class_<CameraCaptureV4L2>(m, "CameraCapture")
        .def(py::init<int, int, int, int>(),
             py::arg("device") = 0,
             py::arg("width") = 640,
             py::arg("height") = 480,
             py::arg("fps") = 30)
        .def("open", &CameraCaptureV4L2::open)
        .def("read", &CameraCaptureV4L2::read)
        .def("release", &CameraCaptureV4L2::release)
        .def("isOpened", &CameraCaptureV4L2::isOpened)
        .def("get_width", &CameraCaptureV4L2::get_width)
        .def("get_height", &CameraCaptureV4L2::get_height);
}

#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(camera_native, m) {
    m.attr("CameraCapture") = py::none();
}
#endif
