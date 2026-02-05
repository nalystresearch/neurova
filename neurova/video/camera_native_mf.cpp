/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

namespace py = pybind11;
using Microsoft::WRL::ComPtr;

class CameraCaptureWin {
public:
    CameraCaptureWin(int device = 0, int width = 640, int height = 480)
        : device_index_(device), width_(width), height_(height), opened_(false) {}

    ~CameraCaptureWin() { release(); }

    bool open() {
        if (opened_) return true;

        HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_FULL);
        if (FAILED(hr)) {
            last_error_ = "MFStartup failed";
            return false;
        }

        hr = MFCreateAttributes(&attributes_, 1);
        if (FAILED(hr)) {
            last_error_ = "MFCreateAttributes failed";
            return false;
        }
        attributes_->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

        IMFActivate** devices = nullptr;
        UINT32 count = 0;
        hr = MFEnumDeviceSources(attributes_.Get(), &devices, &count);
        if (FAILED(hr) || count == 0 || device_index_ >= static_cast<int>(count)) {
            last_error_ = "No MediaFoundation camera devices";
            if (devices) {
                for (UINT32 i = 0; i < count; ++i) devices[i]->Release();
                CoTaskMemFree(devices);
            }
            return false;
        }

        ComPtr<IMFActivate> activate(devices[device_index_]);
        media_source_.Reset();
        hr = activate->ActivateObject(IID_PPV_ARGS(&media_source_));
        for (UINT32 i = 0; i < count; ++i) devices[i]->Release();
        CoTaskMemFree(devices);
        if (FAILED(hr)) {
            last_error_ = "ActivateObject failed";
            return false;
        }

        hr = MFCreateSourceReaderFromMediaSource(media_source_.Get(), nullptr, &reader_);
        if (FAILED(hr)) {
            last_error_ = "MFCreateSourceReaderFromMediaSource failed";
            return false;
        }

        // Configure media type
        ComPtr<IMFMediaType> media_type;
        hr = MFCreateMediaType(&media_type);
        if (FAILED(hr)) {
            last_error_ = "MFCreateMediaType failed";
            return false;
        }

        media_type->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        media_type->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        MFSetAttributeSize(media_type.Get(), MF_MT_FRAME_SIZE, width_, height_);
        MFSetAttributeRatio(media_type.Get(), MF_MT_FRAME_RATE, 30, 1);

        hr = reader_->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, media_type.Get());
        if (FAILED(hr)) {
            last_error_ = "SetCurrentMediaType failed";
            return false;
        }

        opened_ = true;
        return true;
    }

    py::array_t<uint8_t> read_bgra() {
        if (!opened_) {
            return py::array_t<uint8_t>();
        }

        DWORD stream_index = 0;
        DWORD flags = 0;
        ComPtr<IMFSample> sample;
        HRESULT hr = reader_->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &stream_index, &flags, nullptr, &sample);
        if (FAILED(hr) || !sample) {
            return py::array_t<uint8_t>();
        }

        ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr)) {
            return py::array_t<uint8_t>();
        }

        BYTE* data = nullptr;
        DWORD max_length = 0;
        DWORD current_length = 0;
        hr = buffer->Lock(&data, &max_length, &current_length);
        if (FAILED(hr) || !data) {
            return py::array_t<uint8_t>();
        }

        py::array_t<uint8_t> frame({height_, width_, 4});
        auto buf = frame.request();
        const size_t copy_bytes = std::min<size_t>(static_cast<size_t>(current_length), buf.size);
        std::memcpy(buf.ptr, data, copy_bytes);
        buffer->Unlock();
        return frame;
    }

    py::array_t<uint8_t> read() {
        auto bgra = read_bgra();
        if (bgra.size() == 0) return bgra;
        auto buf = bgra.request();
        py::array_t<uint8_t> rgb({height_, width_, 3});
        auto dst = rgb.mutable_data();
        const uint8_t* src = static_cast<const uint8_t*>(buf.ptr);
        for (int i = 0; i < width_ * height_; ++i) {
            dst[i * 3 + 0] = src[i * 4 + 2];
            dst[i * 3 + 1] = src[i * 4 + 1];
            dst[i * 3 + 2] = src[i * 4 + 0];
        }
        return rgb;
    }

    void release() {
        reader_.Reset();
        media_source_.Reset();
        attributes_.Reset();
        if (opened_) {
            MFShutdown();
        }
        opened_ = false;
    }

    bool isOpened() const { return opened_; }
    int get_width() const { return width_; }
    int get_height() const { return height_; }
    std::string last_error() const { return last_error_; }

private:
    int device_index_;
    int width_;
    int height_;
    bool opened_;
    std::string last_error_;

    ComPtr<IMFAttributes> attributes_;
    ComPtr<IMFMediaSource> media_source_;
    ComPtr<IMFSourceReader> reader_;
};

PYBIND11_MODULE(camera_native, m) {
    py::class_<CameraCaptureWin>(m, "CameraCapture")
        .def(py::init<int, int, int>(), py::arg("device") = 0, py::arg("width") = 640, py::arg("height") = 480)
        .def("open", &CameraCaptureWin::open)
        .def("read", &CameraCaptureWin::read)
        .def("read_bgra", &CameraCaptureWin::read_bgra)
        .def("release", &CameraCaptureWin::release)
        .def("isOpened", &CameraCaptureWin::isOpened)
        .def("get_width", &CameraCaptureWin::get_width)
        .def("get_height", &CameraCaptureWin::get_height);
}

#else
// Non-Windows placeholder to avoid build issues on other platforms
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(camera_native, m) {
    m.attr("CameraCapture") = py::none();
}
#endif
