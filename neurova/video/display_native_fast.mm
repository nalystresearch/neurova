/*
 * Neurova Native Display - Optimized macOS Implementation
 * Based on OpenCV's window_cocoa.mm approach
 * Uses CALayer for maximum performance
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#include <string>

namespace py = pybind11;

// Native window class - OpenCV style with CALayer
@interface NativeWindow : NSWindow
@property (strong) NSView *imageView;
@property (assign) int width;
@property (assign) int height;
@end

@implementation NativeWindow
- (instancetype)initWithWidth:(int)w height:(int)h title:(NSString*)title {
    NSRect frame = NSMakeRect(100, 100, w, h);
    self = [super initWithContentRect:frame
                            styleMask:(NSWindowStyleMaskTitled |
                                     NSWindowStyleMaskClosable |
                                     NSWindowStyleMaskMiniaturizable |
                                     NSWindowStyleMaskResizable)
                              backing:NSBackingStoreBuffered
                                defer:NO];
    if (self) {
        self.width = w;
        self.height = h;
        [self setTitle:title];
        
        // Create CALayer-backed view (OpenCV approach for fast rendering)
        self.imageView = [[NSView alloc] initWithFrame:frame];
        [self.imageView setWantsLayer:YES];
        [self setContentView:self.imageView];
        
        // Make window visible
        [self makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
    }
    return self;
}

- (void)updateFrame:(const uint8_t*)data width:(int)w height:(int)h {
    @autoreleasepool {
        // OpenCV approach: Create bitmap from existing data (no copy)
        NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:(unsigned char**)&data
            pixelsWide:w
            pixelsHigh:h
            bitsPerSample:8
            samplesPerPixel:3
            hasAlpha:NO
            isPlanar:NO
            colorSpaceName:NSDeviceRGBColorSpace
            bitmapFormat:kCGImageAlphaNone
            bytesPerRow:w * 3
            bitsPerPixel:24];
        
        NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(w, h)];
        [image addRepresentation:bitmap];
        
        // CALayer rendering - FAST, no drawing overhead
        [[self.imageView layer] setContents:image];
    }
}
@end

// C++ wrapper
class NativeDisplay {
private:
    NativeWindow *window;
    int width, height;
    std::string window_title;
    bool is_open;

public:
    NativeDisplay(int w = 640, int h = 480, const std::string& title = "Neurova Display")
        : width(w), height(h), window_title(title), window(nil), is_open(false) {
        // Initialize NSApp
        @autoreleasepool {
            if (![NSApp isRunning]) {
                [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
                [NSApp finishLaunching];
            }
        }
    }
    
    ~NativeDisplay() {
        close();
    }
    
    bool open() {
        if (is_open) return true;
        
        @autoreleasepool {
            NSString *title = [NSString stringWithUTF8String:window_title.c_str()];
            window = [[NativeWindow alloc] initWithWidth:width height:height title:title];
            
            if (window) {
                is_open = true;
                // Brief event loop to show window
                [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
                return true;
            }
        }
        return false;
    }
    
    bool show(py::array_t<uint8_t> frame) {
        if (!is_open) {
            if (!open()) return false;
        }
        
        auto buf = frame.request();
        if (buf.ndim != 3 || buf.shape[2] != 3) return false;
        
        int h = buf.shape[0];
        int w = buf.shape[1];
        
        @autoreleasepool {
            uint8_t *data = static_cast<uint8_t*>(buf.ptr);
            [window updateFrame:data width:w height:h];
            
            // Minimal event processing - OpenCV style
            NSEvent *event;
            while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                              untilDate:nil
                                                 inMode:NSDefaultRunLoopMode
                                                dequeue:YES])) {
                [NSApp sendEvent:event];
            }
        }
        
        return true;
    }
    
    bool isOpened() const {
        if (!is_open || !window) return false;
        return [window isVisible];
    }
    
    void close() {
        if (window) {
            @autoreleasepool {
                [window close];
                window = nil;
            }
        }
        is_open = false;
    }
};

PYBIND11_MODULE(display_native, m) {
    m.doc() = "Neurova native display (OpenCV-style CALayer rendering)";
    
    py::class_<NativeDisplay>(m, "NativeDisplay")
        .def(py::init<int, int, const std::string&>(),
             py::arg("width") = 640,
             py::arg("height") = 480,
             py::arg("title") = "Neurova Display")
        .def("open", &NativeDisplay::open)
        .def("show", &NativeDisplay::show)
        .def("isOpened", &NativeDisplay::isOpened)
        .def("close", &NativeDisplay::close);
}
