/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/**
 * Neurova Media Foundation Camera Capture Backend
 * Windows Media Foundation camera capture implementation
 */

#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>

#ifdef _WIN32

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <comdef.h>
#include <wrl/client.h>

#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

using Microsoft::WRL::ComPtr;

namespace neurova {
namespace videoio {
namespace mf {

// Frame callback type
using FrameCallback = std::function<void(const uint8_t* data, int width, int height, 
                                         int channels, int64_t timestamp)>;

// Camera device info
struct CameraInfo {
    std::wstring device_name;
    std::wstring symbolic_link;
    int index;
};

// Media Foundation initialization helper
class MFInitializer {
public:
    MFInitializer() : initialized_(false) {
        HRESULT hr = MFStartup(MF_VERSION);
        initialized_ = SUCCEEDED(hr);
    }
    
    ~MFInitializer() {
        if (initialized_) {
            MFShutdown();
        }
    }
    
    bool isInitialized() const { return initialized_; }
    
private:
    bool initialized_;
};

// Global MF initializer
static MFInitializer g_mfInit;

// Media Foundation Camera class
class MFCamera {
public:
    MFCamera() : width_(640), height_(480), fps_(30) {}
    
    ~MFCamera() {
        close();
    }
    
    // Enumerate available cameras
    static std::vector<CameraInfo> enumerateCameras() {
        std::vector<CameraInfo> cameras;
        
        if (!g_mfInit.isInitialized()) return cameras;
        
        ComPtr<IMFAttributes> attributes;
        HRESULT hr = MFCreateAttributes(&attributes, 1);
        if (FAILED(hr)) return cameras;
        
        hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                  MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        if (FAILED(hr)) return cameras;
        
        IMFActivate** devices = nullptr;
        UINT32 count = 0;
        
        hr = MFEnumDeviceSources(attributes.Get(), &devices, &count);
        if (FAILED(hr)) return cameras;
        
        for (UINT32 i = 0; i < count; i++) {
            CameraInfo info;
            info.index = i;
            
            // Get friendly name
            WCHAR* name = nullptr;
            UINT32 nameLen = 0;
            hr = devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
                                                 &name, &nameLen);
            if (SUCCEEDED(hr) && name) {
                info.device_name = name;
                CoTaskMemFree(name);
            }
            
            // Get symbolic link
            WCHAR* link = nullptr;
            UINT32 linkLen = 0;
            hr = devices[i]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
                &link, &linkLen);
            if (SUCCEEDED(hr) && link) {
                info.symbolic_link = link;
                CoTaskMemFree(link);
            }
            
            cameras.push_back(info);
            devices[i]->Release();
        }
        
        CoTaskMemFree(devices);
        
        return cameras;
    }
    
    // Open camera by index
    bool open(int cameraIndex = 0) {
        if (!g_mfInit.isInitialized()) {
            error_ = "Media Foundation not initialized";
            return false;
        }
        
        close();
        
        // Enumerate devices
        ComPtr<IMFAttributes> attributes;
        HRESULT hr = MFCreateAttributes(&attributes, 1);
        if (FAILED(hr)) {
            error_ = "Failed to create attributes";
            return false;
        }
        
        hr = attributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                  MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
        if (FAILED(hr)) return false;
        
        IMFActivate** devices = nullptr;
        UINT32 count = 0;
        
        hr = MFEnumDeviceSources(attributes.Get(), &devices, &count);
        if (FAILED(hr) || count == 0) {
            error_ = "No cameras found";
            return false;
        }
        
        if ((UINT32)cameraIndex >= count) {
            error_ = "Camera index out of range";
            for (UINT32 i = 0; i < count; i++) devices[i]->Release();
            CoTaskMemFree(devices);
            return false;
        }
        
        // Activate the camera source
        ComPtr<IMFMediaSource> source;
        hr = devices[cameraIndex]->ActivateObject(IID_PPV_ARGS(&source));
        
        for (UINT32 i = 0; i < count; i++) devices[i]->Release();
        CoTaskMemFree(devices);
        
        if (FAILED(hr)) {
            error_ = "Failed to activate camera";
            return false;
        }
        
        // Create source reader
        ComPtr<IMFAttributes> readerAttributes;
        hr = MFCreateAttributes(&readerAttributes, 2);
        if (FAILED(hr)) return false;
        
        hr = readerAttributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
        hr = readerAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
        
        hr = MFCreateSourceReaderFromMediaSource(source.Get(), readerAttributes.Get(),
                                                  &reader_);
        if (FAILED(hr)) {
            error_ = "Failed to create source reader";
            return false;
        }
        
        // Configure output format (prefer RGB32)
        configureOutputFormat();
        
        camera_index_ = cameraIndex;
        return true;
    }
    
    // Close camera
    void close() {
        stopCapture();
        reader_.Reset();
    }
    
    // Check if open
    bool isOpened() const { return reader_ != nullptr; }
    
    // Set resolution
    bool setResolution(int width, int height) {
        if (!isOpened()) return false;
        
        // Get current media type
        ComPtr<IMFMediaType> mediaType;
        HRESULT hr = reader_->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                   &mediaType);
        if (FAILED(hr)) return false;
        
        // Try to find a matching format
        DWORD index = 0;
        while (true) {
            ComPtr<IMFMediaType> type;
            hr = reader_->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                              index, &type);
            if (FAILED(hr)) break;
            
            UINT32 w, h;
            MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
            
            if ((int)w == width && (int)h == height) {
                hr = reader_->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                   nullptr, type.Get());
                if (SUCCEEDED(hr)) {
                    width_ = width;
                    height_ = height;
                    return true;
                }
            }
            
            index++;
        }
        
        return false;
    }
    
    void getResolution(int& width, int& height) const {
        width = width_;
        height = height_;
    }
    
    // Set frame rate
    bool setFrameRate(int fps) {
        // Frame rate is typically set through media type
        fps_ = fps;
        return true;
    }
    
    // Get supported resolutions
    struct Resolution {
        int width;
        int height;
        int fps;
    };
    
    std::vector<Resolution> getSupportedResolutions() {
        std::vector<Resolution> resolutions;
        
        if (!isOpened()) return resolutions;
        
        DWORD index = 0;
        while (true) {
            ComPtr<IMFMediaType> type;
            HRESULT hr = reader_->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                      index, &type);
            if (FAILED(hr)) break;
            
            Resolution res;
            UINT32 w, h;
            MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
            res.width = w;
            res.height = h;
            
            UINT32 num, denom;
            MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &num, &denom);
            res.fps = denom > 0 ? num / denom : 30;
            
            resolutions.push_back(res);
            index++;
        }
        
        return resolutions;
    }
    
    // Start capture
    bool startCapture() {
        if (!isOpened() || capturing_) return false;
        capturing_ = true;
        return true;
    }
    
    // Stop capture
    void stopCapture() {
        if (capture_thread_running_) {
            capture_thread_running_ = false;
            if (capture_thread_.joinable()) {
                capture_thread_.join();
            }
        }
        capturing_ = false;
    }
    
    // Read a single frame (synchronous)
    bool readFrame(std::vector<uint8_t>& data, int64_t& timestamp) {
        if (!isOpened() || !capturing_) return false;
        
        ComPtr<IMFSample> sample;
        DWORD streamIndex, flags;
        LONGLONG ts;
        
        HRESULT hr = reader_->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                          0, &streamIndex, &flags, &ts, &sample);
        
        if (FAILED(hr) || !sample) {
            if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
                error_ = "End of stream";
            }
            return false;
        }
        
        timestamp = ts / 10000;  // Convert to milliseconds
        
        // Get buffer from sample
        ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr)) return false;
        
        BYTE* rawData = nullptr;
        DWORD maxLen, curLen;
        hr = buffer->Lock(&rawData, &maxLen, &curLen);
        if (FAILED(hr)) return false;
        
        // Copy data
        data.resize(curLen);
        memcpy(data.data(), rawData, curLen);
        
        buffer->Unlock();
        
        return true;
    }
    
    // Read frame as RGB
    bool readFrameRGB(std::vector<uint8_t>& rgb_data, int64_t& timestamp) {
        std::vector<uint8_t> raw_data;
        if (!readFrame(raw_data, timestamp)) return false;
        
        // Convert to RGB based on current format
        convertToRGB(raw_data.data(), raw_data.size(), rgb_data);
        
        return true;
    }
    
    // Start asynchronous capture with callback
    bool startAsyncCapture(FrameCallback callback) {
        if (!capturing_ || capture_thread_running_) return false;
        
        frame_callback_ = callback;
        capture_thread_running_ = true;
        
        capture_thread_ = std::thread([this]() {
            std::vector<uint8_t> rgb_data;
            
            while (capture_thread_running_) {
                int64_t timestamp;
                if (readFrameRGB(rgb_data, timestamp)) {
                    if (frame_callback_) {
                        frame_callback_(rgb_data.data(), width_, height_, 3, timestamp);
                    }
                }
            }
        });
        
        return true;
    }
    
    // Camera controls
    bool setBrightness(int value) {
        return setCameraControl(VideoProcAmp_Brightness, value);
    }
    
    bool setContrast(int value) {
        return setCameraControl(VideoProcAmp_Contrast, value);
    }
    
    bool setSaturation(int value) {
        return setCameraControl(VideoProcAmp_Saturation, value);
    }
    
    bool setExposure(int value) {
        return setCameraControl(CameraControl_Exposure, value, true);
    }
    
    bool setAutoExposure(bool enabled) {
        ComPtr<IAMCameraControl> control;
        if (getControl(control)) {
            return SUCCEEDED(control->Set(CameraControl_Exposure, 0,
                enabled ? CameraControl_Flags_Auto : CameraControl_Flags_Manual));
        }
        return false;
    }
    
    bool setFocus(int value) {
        return setCameraControl(CameraControl_Focus, value, true);
    }
    
    bool setAutoFocus(bool enabled) {
        ComPtr<IAMCameraControl> control;
        if (getControl(control)) {
            return SUCCEEDED(control->Set(CameraControl_Focus, 0,
                enabled ? CameraControl_Flags_Auto : CameraControl_Flags_Manual));
        }
        return false;
    }
    
    std::string getLastError() const {
        // Convert wide string error if needed
        return error_;
    }

private:
    void configureOutputFormat() {
        // Try to set RGB32 output format for easy conversion
        ComPtr<IMFMediaType> outputType;
        MFCreateMediaType(&outputType);
        outputType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        outputType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
        
        HRESULT hr = reader_->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                   nullptr, outputType.Get());
        
        if (SUCCEEDED(hr)) {
            output_format_ = MFVideoFormat_RGB32;
        } else {
            // Fall back to native format
            ComPtr<IMFMediaType> nativeType;
            reader_->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &nativeType);
            nativeType->GetGUID(MF_MT_SUBTYPE, &output_format_);
        }
        
        // Get actual resolution
        ComPtr<IMFMediaType> currentType;
        hr = reader_->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType);
        if (SUCCEEDED(hr)) {
            UINT32 w, h;
            MFGetAttributeSize(currentType.Get(), MF_MT_FRAME_SIZE, &w, &h);
            width_ = w;
            height_ = h;
        }
    }
    
    void convertToRGB(const uint8_t* src, size_t src_size, std::vector<uint8_t>& dst) {
        int num_pixels = width_ * height_;
        dst.resize(num_pixels * 3);
        
        if (output_format_ == MFVideoFormat_RGB32) {
            // BGRA to RGB
            for (int i = 0; i < num_pixels; i++) {
                dst[i * 3 + 0] = src[i * 4 + 2];  // R
                dst[i * 3 + 1] = src[i * 4 + 1];  // G
                dst[i * 3 + 2] = src[i * 4 + 0];  // B
            }
        } else if (output_format_ == MFVideoFormat_YUY2) {
            // YUYV to RGB
            for (int i = 0; i < num_pixels / 2; i++) {
                int y0 = src[i * 4 + 0];
                int u  = src[i * 4 + 1];
                int y1 = src[i * 4 + 2];
                int v  = src[i * 4 + 3];
                
                int c0 = y0 - 16;
                int c1 = y1 - 16;
                int d = u - 128;
                int e = v - 128;
                
                auto clamp = [](int x) -> uint8_t {
                    return x < 0 ? 0 : (x > 255 ? 255 : (uint8_t)x);
                };
                
                dst[i * 6 + 0] = clamp((298 * c0 + 409 * e + 128) >> 8);
                dst[i * 6 + 1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
                dst[i * 6 + 2] = clamp((298 * c0 + 516 * d + 128) >> 8);
                
                dst[i * 6 + 3] = clamp((298 * c1 + 409 * e + 128) >> 8);
                dst[i * 6 + 4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
                dst[i * 6 + 5] = clamp((298 * c1 + 516 * d + 128) >> 8);
            }
        } else if (output_format_ == MFVideoFormat_NV12) {
            // NV12 to RGB
            const uint8_t* y_plane = src;
            const uint8_t* uv_plane = src + width_ * height_;
            
            for (int j = 0; j < height_; j++) {
                for (int i = 0; i < width_; i++) {
                    int y = y_plane[j * width_ + i];
                    int uv_idx = (j / 2) * width_ + (i / 2) * 2;
                    int u = uv_plane[uv_idx] - 128;
                    int v = uv_plane[uv_idx + 1] - 128;
                    
                    int c = y - 16;
                    
                    auto clamp = [](int x) -> uint8_t {
                        return x < 0 ? 0 : (x > 255 ? 255 : (uint8_t)x);
                    };
                    
                    int idx = (j * width_ + i) * 3;
                    dst[idx + 0] = clamp((298 * c + 409 * v + 128) >> 8);
                    dst[idx + 1] = clamp((298 * c - 100 * u - 208 * v + 128) >> 8);
                    dst[idx + 2] = clamp((298 * c + 516 * u + 128) >> 8);
                }
            }
        }
    }
    
    bool getControl(ComPtr<IAMCameraControl>& control) {
        if (!isOpened()) return false;
        
        ComPtr<IMFMediaSource> source;
        HRESULT hr = reader_->GetServiceForStream(MF_SOURCE_READER_MEDIASOURCE,
                                                   GUID_NULL, IID_PPV_ARGS(&source));
        if (FAILED(hr)) return false;
        
        hr = source->QueryInterface(IID_PPV_ARGS(&control));
        return SUCCEEDED(hr);
    }
    
    bool getVideoProcAmp(ComPtr<IAMVideoProcAmp>& amp) {
        if (!isOpened()) return false;
        
        ComPtr<IMFMediaSource> source;
        HRESULT hr = reader_->GetServiceForStream(MF_SOURCE_READER_MEDIASOURCE,
                                                   GUID_NULL, IID_PPV_ARGS(&source));
        if (FAILED(hr)) return false;
        
        hr = source->QueryInterface(IID_PPV_ARGS(&amp));
        return SUCCEEDED(hr);
    }
    
    bool setCameraControl(long property, int value, bool is_camera_control = false) {
        if (is_camera_control) {
            ComPtr<IAMCameraControl> control;
            if (getControl(control)) {
                return SUCCEEDED(control->Set(property, value, CameraControl_Flags_Manual));
            }
        } else {
            ComPtr<IAMVideoProcAmp> amp;
            if (getVideoProcAmp(amp)) {
                return SUCCEEDED(amp->Set(property, value, VideoProcAmp_Flags_Manual));
            }
        }
        return false;
    }

private:
    ComPtr<IMFSourceReader> reader_;
    int camera_index_ = -1;
    int width_;
    int height_;
    int fps_;
    GUID output_format_;
    std::string error_;
    
    std::atomic<bool> capturing_{false};
    std::thread capture_thread_;
    std::atomic<bool> capture_thread_running_{false};
    FrameCallback frame_callback_;
};

} // namespace mf
} // namespace videoio
} // namespace neurova

#endif // _WIN32
