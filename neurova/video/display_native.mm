/*
 * Neurova Native Display - macOS Implementation
 * Stable display path using NSBitmapImageRep with internal storage
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
#include <string>

namespace py = pybind11;

// Native window class for displaying frames
@interface NativeWindow : NSWindow
@property (strong) NSImageView *imageView;
@property (strong) NSBitmapImageRep *bitmapRep;
@property (strong) NSImage *image;
@property (assign) int width;
@property (assign) int height;
@property (assign) int channels;
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

        self.channels = 3;
        self.bitmapRep = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:NULL
            pixelsWide:w
            pixelsHigh:h
            bitsPerSample:8
            samplesPerPixel:3
            hasAlpha:NO
            isPlanar:NO
            colorSpaceName:NSDeviceRGBColorSpace
            bytesPerRow:w * 3
            bitsPerPixel:24];

        self.image = [[NSImage alloc] initWithSize:NSMakeSize(w, h)];
        [self.image addRepresentation:self.bitmapRep];

        self.imageView = [[NSImageView alloc] initWithFrame:frame];
        [self.imageView setImageScaling:NSImageScaleProportionallyUpOrDown];
        [self.imageView setImage:self.image];
        self.imageView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
        [self setContentView:self.imageView];

        [self makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];
    }
    return self;
}

- (void)updateFrame:(const uint8_t*)data width:(int)w height:(int)h channels:(int)c {
    if (w != self.width || h != self.height || c != self.channels || self.bitmapRep == nil) {
        self.width = w;
        self.height = h;
        self.channels = c;

        if (c == 4) {
            NSBitmapFormat format = NSBitmapFormatAlphaNonpremultiplied;
            self.bitmapRep = [[NSBitmapImageRep alloc]
                initWithBitmapDataPlanes:NULL
                pixelsWide:w
                pixelsHigh:h
                bitsPerSample:8
                samplesPerPixel:4
                hasAlpha:YES
                isPlanar:NO
                colorSpaceName:NSDeviceRGBColorSpace
                bitmapFormat:format
                bytesPerRow:w * 4
                bitsPerPixel:32];
        } else {
            self.bitmapRep = [[NSBitmapImageRep alloc]
                initWithBitmapDataPlanes:NULL
                pixelsWide:w
                pixelsHigh:h
                bitsPerSample:8
                samplesPerPixel:3
                hasAlpha:NO
                isPlanar:NO
                colorSpaceName:NSDeviceRGBColorSpace
                bytesPerRow:w * 3
                bitsPerPixel:24];
        }

        self.image = [[NSImage alloc] initWithSize:NSMakeSize(w, h)];
        [self.image addRepresentation:self.bitmapRep];
        [self.imageView setImage:self.image];
    }

    uint8_t *dst = [self.bitmapRep bitmapData];
    if (!dst) return;
    size_t rowBytes = (size_t)w * (size_t)c;
    if (c == 4) {
        for (int y = 0; y < h; ++y) {
            const uint8_t *src = data + (size_t)y * rowBytes;
            uint8_t *dstRow = dst + (size_t)y * rowBytes;
            for (int x = 0; x < w; ++x) {
                const size_t idx = (size_t)x * 4;
                dstRow[idx + 0] = src[idx + 2];
                dstRow[idx + 1] = src[idx + 1];
                dstRow[idx + 2] = src[idx + 0];
                dstRow[idx + 3] = src[idx + 3];
            }
        }
    } else {
        for (int y = 0; y < h; ++y) {
            memcpy(dst + (size_t)y * rowBytes, data + (size_t)y * rowBytes, rowBytes);
        }
    }
    [self.imageView setNeedsDisplay:YES];
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

        auto createWindow = [&]() {
            @autoreleasepool {
                [NSApplication sharedApplication];
                [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
                [NSApp finishLaunching];

                NSString *nsTitle = [NSString stringWithUTF8String:title.c_str()];
                window = [[NativeWindow alloc] initWithWidth:width
                                                      height:height
                                                       title:nsTitle];

                if (window) {
                    is_open = true;
                    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.01]];
                    return true;
                }
            }
            return false;
        };

        if ([NSThread isMainThread]) {
            return createWindow();
        }

        __block bool ok = false;
        dispatch_sync(dispatch_get_main_queue(), ^{
            ok = createWindow();
        });
        return ok;
    }
    
    bool show(py::array_t<uint8_t> frame) {
        if (!is_open) {
            if (!open()) return false;
        }
        
        auto buf = frame.request();
        if (buf.ndim != 3) {
            return false;
        }
        
        int h = buf.shape[0];
        int w = buf.shape[1];
        int channels = buf.shape[2];
        
        if (channels != 3 && channels != 4) return false;
        
        @autoreleasepool {
            uint8_t *data = static_cast<uint8_t*>(buf.ptr);

            if ([NSThread isMainThread]) {
                [window updateFrame:data width:w height:h channels:channels];
            } else {
                dispatch_sync(dispatch_get_main_queue(), ^{
                    [window updateFrame:data width:w height:h channels:channels];
                });
            }

            NSEvent *event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                               untilDate:[NSDate distantPast]
                                                  inMode:NSDefaultRunLoopMode
                                                 dequeue:YES];
            if (event) {
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
