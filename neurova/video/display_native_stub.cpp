/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class NativeDisplayStub {
public:
    NativeDisplayStub(int width = 640, int height = 480, const std::string& title = "Neurova Display")
        : width_(width), height_(height), title_(title), opened_(false) {}

    bool open() {
        opened_ = false;
        return false;
    }

    bool show(py::array /*frame*/) {
        return false;
    }

    bool isOpened() const { return opened_; }

    void close() { opened_ = false; }

private:
    int width_;
    int height_;
    std::string title_;
    bool opened_;
};

PYBIND11_MODULE(display_native, m) {
    m.doc() = "Neurova display stub (non-mac platforms)";

    py::class_<NativeDisplayStub>(m, "NativeDisplay")
        .def(py::init<int, int, const std::string&>(),
             py::arg("width") = 640,
             py::arg("height") = 480,
             py::arg("title") = "Neurova Display")
        .def("open", &NativeDisplayStub::open)
        .def("show", &NativeDisplayStub::show)
        .def("isOpened", &NativeDisplayStub::isOpened)
        .def("close", &NativeDisplayStub::close);
}
