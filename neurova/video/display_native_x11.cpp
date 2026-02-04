/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <vector>
#include <cstdint>

namespace py = pybind11;

class NativeDisplayX11 {
public:
    NativeDisplayX11(int width = 640, int height = 480, const std::string& title = "Neurova Display")
        : width_(width), height_(height), title_(title) {}

    ~NativeDisplayX11() { close(); }

    bool open() {
        if (display_) {
            return true;
        }
        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            return false;
        }
        screen_ = DefaultScreen(display_);
        window_ = XCreateSimpleWindow(
            display_,
            RootWindow(display_, screen_),
            0,
            0,
            width_,
            height_,
            0,
            BlackPixel(display_, screen_),
            WhitePixel(display_, screen_));
        XStoreName(display_, window_, title_.c_str());
        XSelectInput(display_, window_, ExposureMask | StructureNotifyMask | KeyPressMask);
        XMapWindow(display_, window_);
        gc_ = DefaultGC(display_, screen_);
        return true;
    }

    bool show(py::array array) {
        if (!open()) {
            return false;
        }
        py::buffer_info buf = array.request();
        if (buf.ndim != 3) {
            return false;
        }
        const int h = static_cast<int>(buf.shape[0]);
        const int w = static_cast<int>(buf.shape[1]);
        const int c = static_cast<int>(buf.shape[2]);
        if (c != 3 && c != 4) {
            return false;
        }

        if (w != width_ || h != height_) {
            width_ = w;
            height_ = h;
            XResizeWindow(display_, window_, width_, height_);
        }

        buffer_.resize(static_cast<size_t>(w) * h * 4);
        const uint8_t* src = static_cast<const uint8_t*>(buf.ptr);
        uint8_t* dst = buffer_.data();

        if (c == 3) {
            for (int i = 0; i < w * h; ++i) {
                dst[i * 4 + 0] = src[i * 3 + 2];
                dst[i * 4 + 1] = src[i * 3 + 1];
                dst[i * 4 + 2] = src[i * 3 + 0];
                dst[i * 4 + 3] = 0xFF;
            }
        } else {
            for (int i = 0; i < w * h; ++i) {
                dst[i * 4 + 0] = src[i * 4 + 2];
                dst[i * 4 + 1] = src[i * 4 + 1];
                dst[i * 4 + 2] = src[i * 4 + 0];
                dst[i * 4 + 3] = src[i * 4 + 3];
            }
        }

        if (!image_) {
            image_ = XCreateImage(
                display_,
                DefaultVisual(display_, screen_),
                24,
                ZPixmap,
                0,
                nullptr,
                w,
                h,
                32,
                0);
        }
        image_->width = w;
        image_->height = h;
        image_->bytes_per_line = w * 4;
        image_->bits_per_pixel = 32;
        image_->data = reinterpret_cast<char*>(buffer_.data());

        XPutImage(display_, window_, gc_, image_, 0, 0, 0, 0, w, h);
        image_->data = nullptr;  // prevent XDestroyImage from freeing our buffer
        process_events();
        return true;
    }

    bool isOpened() const { return display_ && window_; }

    void close() {
        if (image_) {
            image_->data = nullptr;
            XDestroyImage(image_);
            image_ = nullptr;
        }
        if (display_) {
            if (window_) {
                XDestroyWindow(display_, window_);
                window_ = 0;
            }
            XCloseDisplay(display_);
            display_ = nullptr;
        }
    }

private:
    void process_events() {
        while (XPending(display_)) {
            XEvent event;
            XNextEvent(display_, &event);
            if (event.type == DestroyNotify) {
                window_ = 0;
            }
        }
    }

    Display* display_ = nullptr;
    Window window_ = 0;
    GC gc_ = 0;
    int screen_ = 0;
    int width_ = 640;
    int height_ = 480;
    std::string title_;
    XImage* image_ = nullptr;
    std::vector<uint8_t> buffer_;
};

PYBIND11_MODULE(display_native, m) {
    py::class_<NativeDisplayX11>(m, "NativeDisplay")
        .def(py::init<int, int, const std::string&>(),
             py::arg("width") = 640,
             py::arg("height") = 480,
             py::arg("title") = "Neurova Display")
        .def("open", &NativeDisplayX11::open)
        .def("show", &NativeDisplayX11::show)
        .def("isOpened", &NativeDisplayX11::isOpened)
        .def("close", &NativeDisplayX11::close);
}

#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(display_native, m) {
    m.attr("NativeDisplay") = py::none();
}
#endif
