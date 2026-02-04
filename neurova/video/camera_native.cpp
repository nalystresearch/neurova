/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/*
 * Neurova Native Camera Capture - macOS AVFoundation Implementation
 * High-performance camera interface using AVFoundation (like OpenCV)
 */

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace py = pybind11;

// Frame buffer delegate for AVFoundation
@interface FrameCaptureDelegate : NSObject <AVCaptureVideoDataOutputSampleBufferDelegate>
{
    std::mutex frame_mutex;
    std::condition_variable frame_cv;
    CVPixelBufferRef current_frame;
    bool has_new_frame;
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection;

- (CVPixelBufferRef)getFrame;
- (bool)hasNewFrame;
- (void)clearFrame;

@end

@implementation FrameCaptureDelegate

- (id)init {
    self = [super init];
    if (self) {
        current_frame = nullptr;
        has_new_frame = false;
    }
    return self;
}

- (void)dealloc {
    [self clearFrame];
}

- (void)captureOutput:(AVCaptureOutput *)output
didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
       fromConnection:(AVCaptureConnection *)connection {
    
    std::lock_guard<std::mutex> lock(frame_mutex);
    
    // Release old frame
    if (current_frame) {
        CVPixelBufferRelease(current_frame);
    }
    
    // Get new frame
    CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
    current_frame = (CVPixelBufferRef)CVPixelBufferRetain(imageBuffer);
    has_new_frame = true;
    frame_cv.notify_one();
}

- (CVPixelBufferRef)getFrame {
    std::lock_guard<std::mutex> lock(frame_mutex);
    if (current_frame) {
        CVPixelBufferRetain(current_frame);
    }
    return current_frame;
}

- (bool)hasNewFrame {
    std::lock_guard<std::mutex> lock(frame_mutex);
    return has_new_frame;
}

- (void)clearFrame {
    std::lock_guard<std::mutex> lock(frame_mutex);
    if (current_frame) {
        CVPixelBufferRelease(current_frame);
        current_frame = nullptr;
    }
    has_new_frame = false;
}

@end

// C++ Camera Capture Class
class CameraCapture {
private:
    int device_id;
    int width;
    int height;
    bool is_open;
    
#ifdef __APPLE__
    AVCaptureSession* capture_session;
    AVCaptureDeviceInput* device_input;
    AVCaptureVideoDataOutput* video_output;
    AVCaptureDevice* capture_device;
    FrameCaptureDelegate* delegate;
    dispatch_queue_t capture_queue;
#endif
    
public:
    CameraCapture(int device = 0, int w = 640, int h = 480) 
        : device_id(device), width(w), height(h), is_open(false) {
#ifdef __APPLE__
        capture_session = nil;
        device_input = nil;
        video_output = nil;
        capture_device = nil;
        delegate = nil;
        capture_queue = nil;
#endif
    }
    
    ~CameraCapture() {
        release();
    }
    
    bool open() {
#ifdef __APPLE__
        @autoreleasepool {
            // Create capture session
            capture_session = [[AVCaptureSession alloc] init];
            capture_session.sessionPreset = AVCaptureSessionPreset640x480;
            
            // Get camera device
            NSArray* devices;
            if (@available(macOS 10.15, *)) {
                AVCaptureDeviceDiscoverySession* discovery = 
                    [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                                                                           mediaType:AVMediaTypeVideo
                                                                            position:AVCaptureDevicePositionUnspecified];
                devices = discovery.devices;
            } else {
                devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
            }
            
            if (devices.count == 0) {
                return false;
            }
            
            capture_device = [devices objectAtIndex:(device_id < devices.count ? device_id : 0)];
            
            // Create device input
            NSError* error = nil;
            device_input = [[AVCaptureDeviceInput alloc] initWithDevice:capture_device error:&error];
            if (error) {
                return false;
            }
            
            // Add input to session
            if ([capture_session canAddInput:device_input]) {
                [capture_session addInput:device_input];
            } else {
                return false;
            }
            
            // Create video output
            video_output = [[AVCaptureVideoDataOutput alloc] init];
            
            // Set pixel format to 32BGRA (fastest on Mac)
            NSDictionary* settings = @{
                (id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA)
            };
            video_output.videoSettings = settings;
            
            // Create delegate
            delegate = [[FrameCaptureDelegate alloc] init];
            
            // Create dispatch queue
            capture_queue = dispatch_queue_create("neurova.camera.capture", DISPATCH_QUEUE_SERIAL);
            [video_output setSampleBufferDelegate:delegate queue:capture_queue];
            
            // Add output to session
            if ([capture_session canAddOutput:video_output]) {
                [capture_session addOutput:video_output];
            } else {
                return false;
            }
            
            // Start capture
            [capture_session startRunning];
            
            is_open = true;
            return true;
        }
#else
        return false;
#endif
    }
    
    py::array_t<uint8_t> read() {
        if (!is_open) {
            return py::array_t<uint8_t>();
        }
        
#ifdef __APPLE__
        CVPixelBufferRef pixelBuffer = [delegate getFrame];
        if (!pixelBuffer) {
            return py::array_t<uint8_t>();
        }
        
        // Lock pixel buffer
        CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        
        // Get buffer info
        size_t buf_width = CVPixelBufferGetWidth(pixelBuffer);
        size_t buf_height = CVPixelBufferGetHeight(pixelBuffer);
        uint8_t* base_address = (uint8_t*)CVPixelBufferGetBaseAddress(pixelBuffer);
        size_t bytes_per_row = CVPixelBufferGetBytesPerRow(pixelBuffer);
        
        // Create numpy array (height x width x 3 for RGB)
        py::array_t<uint8_t> frame({(int)buf_height, (int)buf_width, 3});
        auto buf = frame.request();
        uint8_t* ptr = static_cast<uint8_t*>(buf.ptr);
        
        // Convert BGRA to RGB
        for (size_t y = 0; y < buf_height; y++) {
            uint8_t* row = base_address + y * bytes_per_row;
            for (size_t x = 0; x < buf_width; x++) {
                // BGRA format: B G R A
                uint8_t b = row[x * 4 + 0];
                uint8_t g = row[x * 4 + 1];
                uint8_t r = row[x * 4 + 2];
                
                // Store as RGB
                size_t idx = (y * buf_width + x) * 3;
                ptr[idx + 0] = r;
                ptr[idx + 1] = g;
                ptr[idx + 2] = b;
            }
        }
        
        // Unlock and release
        CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
        CVPixelBufferRelease(pixelBuffer);
        
        return frame;
#else
        return py::array_t<uint8_t>();
#endif
    }
    
    void release() {
#ifdef __APPLE__
        if (capture_session) {
            [capture_session stopRunning];
            capture_session = nil;
        }
        device_input = nil;
        video_output = nil;
        delegate = nil;
        if (capture_queue) {
            capture_queue = nil;
        }
#endif
        is_open = false;
    }
    
    bool isOpened() const {
        return is_open;
    }
    
    int getWidth() const { return width; }
    int getHeight() const { return height; }
};

PYBIND11_MODULE(camera_native, m) {
    m.doc() = "Neurova native camera capture (C++ AVFoundation backend)";
    
    py::class_<CameraCapture>(m, "CameraCapture")
        .def(py::init<int, int, int>(), 
             py::arg("device") = 0,
             py::arg("width") = 640,
             py::arg("height") = 480)
        .def("open", &CameraCapture::open)
        .def("read", &CameraCapture::read)
        .def("release", &CameraCapture::release)
        .def("isOpened", &CameraCapture::isOpened)
        .def("get_width", &CameraCapture::getWidth)
        .def("get_height", &CameraCapture::getHeight);
}
