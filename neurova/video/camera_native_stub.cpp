/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class CameraCaptureStub {
public:
    CameraCaptureStub(int device = 0, int width = 640, int height = 480)
        : device_(device), width_(width), height_(height), opened_(false) {}

    bool open() {
        opened_ = false;  // stub cannot open
        return false;
    }

    py::array_t<uint8_t> read() {
        return py::array_t<uint8_t>();
    }

    py::array_t<uint8_t> read_bgra() {
        return py::array_t<uint8_t>();
    }

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
    m.doc() = "Neurova camera stub (non-mac platforms)";

    py::class_<CameraCaptureStub>(m, "CameraCapture")
        .def(py::init<int, int, int>(),
             py::arg("device") = 0,
             py::arg("width") = 640,
             py::arg("height") = 480)
        .def("open", &CameraCaptureStub::open)
        .def("read", &CameraCaptureStub::read)
        .def("read_bgra", &CameraCaptureStub::read_bgra)
        .def("release", &CameraCaptureStub::release)
        .def("isOpened", &CameraCaptureStub::isOpened)
        .def("get_width", &CameraCaptureStub::get_width)
        .def("get_height", &CameraCaptureStub::get_height);
}
