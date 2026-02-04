/*
 * Neurova Native Display - macOS Implementation
 * Native window display using Cocoa/AppKit (like cv2.imshow)
 * Pure C++/Objective-C++ - no Python dependencies for display
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#include <string>
#include <memory>

namespace py = pybind11;

// Native window class for displaying frames
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
        
        // Create image view with CALayer (like OpenCV for fast rendering)
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
    // Use OpenCV's approach: Create NSBitmapImageRep directly from data
    // Use aligned bytesPerRow for better performance
    int bytesPerRow = ((w * 3 + 3) & -4);
    
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
        bytesPerRow:w * 3  // Our data is already packed
        bitsPerPixel:24];
    
    NSImage *image = [[NSImage alloc] initWithSize:NSMakeSize(w, h)];
    [image addRepresentation:bitmap];
    
    // Fast CALayer rendering (like OpenCV) - no drawing, just layer update
    [[self.imageView layer] setContents:image];
}
@end

// C++ wrapper class
class NativeDisplay {
private:
    NativeWindow *window;
    int width;
    int height;
    std::string title;
    bool is_open;
    
public:
    NativeDisplay(int w, int h, const std::string& t) 
        : window(nil), width(w), height(h), title(t), is_open(false) {
    }
    
    ~NativeDisplay() {
        close();
    }
    
    bool open() {
        if (is_open) return true;
        
        @autoreleasepool {
            // Ensure NSApp is initialized
            [NSApplication sharedApplication];
            [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
            
            // Create window
            NSString *nsTitle = [NSString stringWithUTF8String:title.c_str()];
            window = [[NativeWindow alloc] initWithWidth:width 
                                                  height:height 
                                                   title:nsTitle];
            
            if (window) {
                is_open = true;
                // Process events to show window
                [NSApp finishLaunching];
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
        if (buf.ndim != 3) {
            printf("ERROR: Frame must be 3D (HxWxC), got %ld dimensions\n", buf.ndim);
            return false;
        }
        
        int h = buf.shape[0];
        int w = buf.shape[1];
        int channels = buf.shape[2];
        
        if (channels != 3) return false;
        
        @autoreleasepool {
            uint8_t *data = static_cast<uint8_t*>(buf.ptr);
            
            // Direct update on main thread (we're already on it)
            [window updateFrame:data width:w height:h];
            
            // Minimal event processing - don't block
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
    m.doc() = "Neurova native window display module (C++ backend, like cv2.imshow)";
    
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
