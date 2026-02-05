/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Windows placeholder: implement MediaFoundation/DirectShow later.
class CameraCaptureWin {
public:
    CameraCaptureWin(int device = 0, int width = 640, int height = 480)
        : device_(device), width_(width), height_(height), opened_(false) {}

    bool open() {
        opened_ = false;  // not implemented yet
        return false;
    }

    py::array_t<uint8_t> read() { return py::array_t<uint8_t>(); }
    py::array_t<uint8_t> read_bgra() { return py::array_t<uint8_t>(); }

    void release() { opened_ = false; }
    bool isOpened() const { return opened_; }
    int get_width() const { return width_; }
    int get_height() const { return height_; }

private:
    int device_;
    int width_;
    int height_;
    bool opened_;
};

PYBIND11_MODULE(camera_native, m) {
    m.doc() = "Neurova camera capture stub for Windows (MediaFoundation TODO)";

    py::class_<CameraCaptureWin>(m, "CameraCapture")
        .def(py::init<int, int, int>(),
             py::arg("device") = 0,
             py::arg("width") = 640,
             py::arg("height") = 480)
        .def("open", &CameraCaptureWin::open)
        .def("read", &CameraCaptureWin::read)
        .def("read_bgra", &CameraCaptureWin::read_bgra)
        .def("release", &CameraCaptureWin::release)
        .def("isOpened", &CameraCaptureWin::isOpened)
        .def("get_width", &CameraCaptureWin::get_width)
        .def("get_height", &CameraCaptureWin::get_height);
}
