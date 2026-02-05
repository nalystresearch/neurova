/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}

ATOM RegisterDisplayClass(HINSTANCE instance) {
    static ATOM atom = 0;
    if (atom != 0) {
        return atom;
    }
    WNDCLASS wc{};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = instance;
    wc.lpszClassName = TEXT("NeurovaDisplayWindow");
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
    atom = RegisterClass(&wc);
    return atom;
}
}  // namespace

class NativeDisplayWin {
public:
    NativeDisplayWin(int width = 640, int height = 480, const std::string& title = "Neurova Display")
        : width_(width), height_(height), title_(title) {}

    ~NativeDisplayWin() { close(); }

    bool open() {
        if (opened_) {
            return true;
        }
        HINSTANCE instance = GetModuleHandle(nullptr);
        if (!RegisterDisplayClass(instance)) {
            return false;
        }
        const std::wstring wide_title(title_.begin(), title_.end());
        hwnd_ = CreateWindowEx(
            0,
            TEXT("NeurovaDisplayWindow"),
            wide_title.c_str(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            width_,
            height_,
            nullptr,
            nullptr,
            instance,
            nullptr);
        if (!hwnd_) {
            return false;
        }
        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);
        opened_ = true;
        return true;
    }

    bool show(py::array frame) {
        if (!open()) {
            return false;
        }

        py::buffer_info buf = frame.request();
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
            RECT rect{0, 0, width_, height_};
            AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
            SetWindowPos(
                hwnd_,
                nullptr,
                0,
                0,
                rect.right - rect.left,
                rect.bottom - rect.top,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }

        BITMAPINFO bmi{};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = w;
        bmi.bmiHeader.biHeight = -h;  // top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 24;
        bmi.bmiHeader.biCompression = BI_RGB;

        const uint8_t* src = static_cast<const uint8_t*>(buf.ptr);
        buffer_.resize(static_cast<size_t>(w) * h * 3);
        uint8_t* dst = buffer_.data();

        if (c == 3) {
            for (int i = 0; i < w * h; ++i) {
                dst[i * 3 + 0] = src[i * 3 + 2];
                dst[i * 3 + 1] = src[i * 3 + 1];
                dst[i * 3 + 2] = src[i * 3 + 0];
            }
        } else {
            for (int i = 0; i < w * h; ++i) {
                dst[i * 3 + 0] = src[i * 4 + 2];
                dst[i * 3 + 1] = src[i * 4 + 1];
                dst[i * 3 + 2] = src[i * 4 + 0];
            }
        }

        HDC dc = GetDC(hwnd_);
        StretchDIBits(
            dc,
            0,
            0,
            width_,
            height_,
            0,
            0,
            w,
            h,
            buffer_.data(),
            &bmi,
            DIB_RGB_COLORS,
            SRCCOPY);
        ReleaseDC(hwnd_, dc);
        process_events();
        return true;
    }

    bool isOpened() const { return opened_ && hwnd_ != nullptr; }

    void close() {
        if (hwnd_) {
            DestroyWindow(hwnd_);
            hwnd_ = nullptr;
        }
        opened_ = false;
    }

private:
    void process_events() {
        MSG msg{};
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    HWND hwnd_ = nullptr;
    int width_ = 640;
    int height_ = 480;
    std::string title_;
    bool opened_ = false;
    std::vector<uint8_t> buffer_;
};

PYBIND11_MODULE(display_native, m) {
    py::class_<NativeDisplayWin>(m, "NativeDisplay")
        .def(py::init<int, int, const std::string&>(),
             py::arg("width") = 640,
             py::arg("height") = 480,
             py::arg("title") = "Neurova Display")
        .def("open", &NativeDisplayWin::open)
        .def("show", &NativeDisplayWin::show)
        .def("isOpened", &NativeDisplayWin::isOpened)
        .def("close", &NativeDisplayWin::close);
}

#else
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(display_native, m) {
    m.attr("NativeDisplay") = py::none();
}
#endif
