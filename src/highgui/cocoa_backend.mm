/**
 * Neurova macOS Cocoa GUI Backend
 * Native macOS GUI using Cocoa/AppKit
 */

#import <Cocoa/Cocoa.h>
#import <QuartzCore/QuartzCore.h>

#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>

#ifdef __APPLE__

namespace neurova {
namespace gui {
namespace cocoa {

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type
using TrackbarCallback = std::function<void(int value)>;

} // namespace cocoa
} // namespace gui
} // namespace neurova

// Trackbar info
struct TrackbarInfo {
    NSSlider* slider;
    NSTextField* label;
    NSTextField* valueLabel;
    int* valuePtr;
    neurova::gui::cocoa::TrackbarCallback callback;
};

// Custom image view for mouse events
@interface NeurovaImageView : NSImageView {
    neurova::gui::cocoa::MouseCallback mouseCallback;
}
- (void)setMouseCallback:(neurova::gui::cocoa::MouseCallback)callback;
@end

@implementation NeurovaImageView

- (void)setMouseCallback:(neurova::gui::cocoa::MouseCallback)callback {
    mouseCallback = callback;
}

- (BOOL)acceptsFirstResponder {
    return YES;
}

- (void)mouseDown:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        int flags = 0;
        if ([event modifierFlags] & NSEventModifierFlagControl) flags |= 8;
        if ([event modifierFlags] & NSEventModifierFlagShift) flags |= 16;
        if ([event modifierFlags] & NSEventModifierFlagOption) flags |= 32;
        mouseCallback(1, (int)location.x, (int)(self.bounds.size.height - location.y), flags | 1);
    }
    [super mouseDown:event];
}

- (void)mouseUp:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        mouseCallback(4, (int)location.x, (int)(self.bounds.size.height - location.y), 0);
    }
    [super mouseUp:event];
}

- (void)mouseDragged:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        mouseCallback(0, (int)location.x, (int)(self.bounds.size.height - location.y), 1);
    }
    [super mouseDragged:event];
}

- (void)mouseMoved:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        mouseCallback(0, (int)location.x, (int)(self.bounds.size.height - location.y), 0);
    }
    [super mouseMoved:event];
}

- (void)rightMouseDown:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        mouseCallback(2, (int)location.x, (int)(self.bounds.size.height - location.y), 2);
    }
    [super rightMouseDown:event];
}

- (void)rightMouseUp:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        mouseCallback(5, (int)location.x, (int)(self.bounds.size.height - location.y), 0);
    }
    [super rightMouseUp:event];
}

- (void)scrollWheel:(NSEvent *)event {
    if (mouseCallback) {
        NSPoint location = [self convertPoint:[event locationInWindow] fromView:nil];
        int delta = (int)([event deltaY] * 120);  // Match Windows wheel delta
        // Pack wheel delta into flags
        mouseCallback(10, (int)location.x, (int)(self.bounds.size.height - location.y), delta);
    }
    [super scrollWheel:event];
}

@end

// Window delegate for handling close
@interface NeurovaWindowDelegate : NSObject <NSWindowDelegate> {
    BOOL windowClosed;
}
@property (nonatomic) BOOL windowClosed;
@end

@implementation NeurovaWindowDelegate
@synthesize windowClosed;

- (BOOL)windowShouldClose:(NSWindow *)sender {
    self.windowClosed = YES;
    return NO;  // Don't actually close, just mark as closed
}

@end

// Trackbar target for slider actions
@interface TrackbarTarget : NSObject {
    TrackbarInfo* info;
}
- (id)initWithInfo:(TrackbarInfo*)trackbarInfo;
- (void)sliderChanged:(id)sender;
@end

@implementation TrackbarTarget

- (id)initWithInfo:(TrackbarInfo*)trackbarInfo {
    self = [super init];
    if (self) {
        info = trackbarInfo;
    }
    return self;
}

- (void)sliderChanged:(id)sender {
    int value = (int)[info->slider integerValue];
    
    if (info->valuePtr) {
        *info->valuePtr = value;
    }
    
    if (info->valueLabel) {
        [info->valueLabel setStringValue:[NSString stringWithFormat:@"%d", value]];
    }
    
    if (info->callback) {
        info->callback(value);
    }
}

@end

namespace neurova {
namespace gui {
namespace cocoa {

// Window info structure
struct WindowInfo {
    NSWindow* window;
    NeurovaImageView* imageView;
    NSView* contentView;
    NeurovaWindowDelegate* delegate;
    std::unordered_map<std::string, TrackbarInfo*> trackbars;
    std::vector<id> trackbarTargets;
    int imageWidth;
    int imageHeight;
    int trackbarCount;
    int flags;
    MouseCallback mouseCallback;
};

// Global state
static std::unordered_map<std::string, WindowInfo*> g_windows;
static std::mutex g_mutex;
static bool g_initialized = false;
static int g_lastKey = -1;
static NSApplication* g_app = nullptr;

// Initialize Cocoa
bool init() {
    if (g_initialized) return true;
    
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
        [NSApp activateIgnoringOtherApps:YES];
    }
    
    g_app = [NSApplication sharedApplication];
    g_initialized = true;
    
    return true;
}

// Cleanup
void cleanup() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        for (auto& tpair : info->trackbars) {
            delete tpair.second;
        }
        
        [info->window close];
        delete info;
    }
    g_windows.clear();
}

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_windows.find(name) != g_windows.end()) {
        return;  // Window already exists
    }
    
    @autoreleasepool {
        WindowInfo* info = new WindowInfo();
        info->imageWidth = 640;
        info->imageHeight = 480;
        info->trackbarCount = 0;
        info->flags = flags;
        
        // Window style
        NSWindowStyleMask style = NSWindowStyleMaskTitled | 
                                  NSWindowStyleMaskClosable | 
                                  NSWindowStyleMaskMiniaturizable;
        
        if (!(flags & 1)) {  // Not WINDOW_AUTOSIZE
            style |= NSWindowStyleMaskResizable;
        }
        
        // Create window
        NSRect frame = NSMakeRect(100, 100, 640, 480);
        info->window = [[NSWindow alloc] initWithContentRect:frame
                                        styleMask:style
                                        backing:NSBackingStoreBuffered
                                        defer:NO];
        
        [info->window setTitle:[NSString stringWithUTF8String:name.c_str()]];
        
        // Set delegate
        info->delegate = [[NeurovaWindowDelegate alloc] init];
        [info->window setDelegate:info->delegate];
        
        // Create content view
        info->contentView = [[NSView alloc] initWithFrame:frame];
        [info->window setContentView:info->contentView];
        
        // Create image view
        info->imageView = [[NeurovaImageView alloc] initWithFrame:frame];
        [info->imageView setImageScaling:NSImageScaleProportionallyUpOrDown];
        [info->contentView addSubview:info->imageView];
        
        // Enable mouse tracking
        NSTrackingArea* trackingArea = [[NSTrackingArea alloc] 
            initWithRect:frame
            options:(NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited | 
                    NSTrackingActiveAlways | NSTrackingInVisibleRect)
            owner:info->imageView
            userInfo:nil];
        [info->imageView addTrackingArea:trackingArea];
        
        [info->window makeKeyAndOrderFront:nil];
        
        g_windows[name] = info;
    }
}

// Display image in window
void imshow(const std::string& name, const unsigned char* data,
            int width, int height, int channels) {
    if (!init()) return;
    
    // Create window if it doesn't exist
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_windows.find(name) == g_windows.end()) {
            g_mutex.unlock();
            namedWindow(name);
            g_mutex.lock();
        }
    }
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it == g_windows.end()) return;
    
    WindowInfo* info = it->second;
    
    @autoreleasepool {
        // Create bitmap representation
        NSBitmapImageRep* bitmap = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:NULL
            pixelsWide:width
            pixelsHigh:height
            bitsPerSample:8
            samplesPerPixel:channels == 1 ? 1 : (channels == 4 ? 4 : 3)
            hasAlpha:(channels == 4)
            isPlanar:NO
            colorSpaceName:channels == 1 ? NSDeviceWhiteColorSpace : NSDeviceRGBColorSpace
            bytesPerRow:width * (channels == 1 ? 1 : (channels == 4 ? 4 : 3))
            bitsPerPixel:0];
        
        unsigned char* bitmapData = [bitmap bitmapData];
        
        // Copy data
        if (channels == 1) {
            memcpy(bitmapData, data, width * height);
        } else if (channels == 3 || channels == 4) {
            memcpy(bitmapData, data, width * height * (channels == 4 ? 4 : 3));
        }
        
        // Create image
        NSImage* image = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
        [image addRepresentation:bitmap];
        
        // Set image
        [info->imageView setImage:image];
        
        info->imageWidth = width;
        info->imageHeight = height;
        
        // Resize window if AUTOSIZE
        if (info->flags & 1) {
            NSRect frame = [info->window frame];
            NSRect contentRect = [info->window contentRectForFrameRect:frame];
            contentRect.size.width = width;
            contentRect.size.height = height + info->trackbarCount * 30;
            frame = [info->window frameRectForContentRect:contentRect];
            [info->window setFrame:frame display:YES animate:NO];
            
            // Update image view frame
            NSRect imageFrame = NSMakeRect(0, info->trackbarCount * 30, width, height);
            [info->imageView setFrame:imageFrame];
        }
    }
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_initialized) return -1;
    
    g_lastKey = -1;
    
    @autoreleasepool {
        NSDate* endDate = delay > 0 ? 
            [NSDate dateWithTimeIntervalSinceNow:delay / 1000.0] : 
            [NSDate distantFuture];
        
        while (g_lastKey == -1) {
            NSEvent* event = [g_app nextEventMatchingMask:NSEventMaskAny
                                    untilDate:endDate
                                    inMode:NSDefaultRunLoopMode
                                    dequeue:YES];
            
            if (event == nil) {
                break;  // Timeout
            }
            
            if ([event type] == NSEventTypeKeyDown) {
                g_lastKey = [event keyCode];
                // Convert to ASCII if possible
                NSString* chars = [event characters];
                if ([chars length] > 0) {
                    g_lastKey = [chars characterAtIndex:0];
                }
            }
            
            [g_app sendEvent:event];
            [g_app updateWindows];
        }
    }
    
    return g_lastKey;
}

// Destroy window
void destroyWindow(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        @autoreleasepool {
            WindowInfo* info = it->second;
            
            for (auto& tpair : info->trackbars) {
                delete tpair.second;
            }
            
            [info->window close];
            delete info;
        }
        g_windows.erase(it);
    }
}

// Destroy all windows
void destroyAllWindows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    @autoreleasepool {
        for (auto& pair : g_windows) {
            WindowInfo* info = pair.second;
            
            for (auto& tpair : info->trackbars) {
                delete tpair.second;
            }
            
            [info->window close];
            delete info;
        }
    }
    g_windows.clear();
}

// Create trackbar
void createTrackbar(const std::string& trackbarName,
                    const std::string& windowName,
                    int* value, int maxValue,
                    TrackbarCallback callback = nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(windowName);
    if (it == g_windows.end()) return;
    
    WindowInfo* winfo = it->second;
    
    @autoreleasepool {
        int y = winfo->trackbarCount * 30;
        
        // Create trackbar info
        TrackbarInfo* tinfo = new TrackbarInfo();
        tinfo->valuePtr = value;
        tinfo->callback = callback;
        
        // Create label
        tinfo->label = [[NSTextField alloc] initWithFrame:NSMakeRect(5, y, 100, 25)];
        [tinfo->label setStringValue:[NSString stringWithUTF8String:trackbarName.c_str()]];
        [tinfo->label setBezeled:NO];
        [tinfo->label setDrawsBackground:NO];
        [tinfo->label setEditable:NO];
        [tinfo->label setSelectable:NO];
        [winfo->contentView addSubview:tinfo->label];
        
        // Create slider
        NSRect contentFrame = [winfo->contentView frame];
        tinfo->slider = [[NSSlider alloc] initWithFrame:NSMakeRect(110, y, contentFrame.size.width - 170, 25)];
        [tinfo->slider setMinValue:0];
        [tinfo->slider setMaxValue:maxValue];
        [tinfo->slider setIntegerValue:*value];
        [tinfo->slider setContinuous:YES];
        
        // Create target for slider
        TrackbarTarget* target = [[TrackbarTarget alloc] initWithInfo:tinfo];
        [tinfo->slider setTarget:target];
        [tinfo->slider setAction:@selector(sliderChanged:)];
        winfo->trackbarTargets.push_back(target);
        
        [winfo->contentView addSubview:tinfo->slider];
        
        // Create value label
        tinfo->valueLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(contentFrame.size.width - 55, y, 50, 25)];
        [tinfo->valueLabel setStringValue:[NSString stringWithFormat:@"%d", *value]];
        [tinfo->valueLabel setBezeled:NO];
        [tinfo->valueLabel setDrawsBackground:NO];
        [tinfo->valueLabel setEditable:NO];
        [tinfo->valueLabel setSelectable:NO];
        [tinfo->valueLabel setAlignment:NSTextAlignmentRight];
        [winfo->contentView addSubview:tinfo->valueLabel];
        
        winfo->trackbars[trackbarName] = tinfo;
        winfo->trackbarCount++;
        
        // Resize window
        if (winfo->flags & 1) {
            NSRect frame = [winfo->window frame];
            NSRect contentRect = [winfo->window contentRectForFrameRect:frame];
            contentRect.size.height = winfo->imageHeight + winfo->trackbarCount * 30;
            frame = [winfo->window frameRectForContentRect:contentRect];
            [winfo->window setFrame:frame display:YES animate:NO];
        }
    }
}

// Set mouse callback
void setMouseCallback(const std::string& windowName, MouseCallback callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(windowName);
    if (it != g_windows.end()) {
        it->second->mouseCallback = callback;
        [it->second->imageView setMouseCallback:callback];
    }
}

// Move window
void moveWindow(const std::string& name, int x, int y) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        @autoreleasepool {
            NSPoint point = NSMakePoint(x, [[NSScreen mainScreen] frame].size.height - y);
            [it->second->window setFrameTopLeftPoint:point];
        }
    }
}

// Resize window
void resizeWindow(const std::string& name, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        @autoreleasepool {
            NSRect frame = [it->second->window frame];
            frame.size.width = width;
            frame.size.height = height;
            [it->second->window setFrame:frame display:YES animate:NO];
        }
    }
}

// Set window title
void setWindowTitle(const std::string& name, const std::string& title) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        @autoreleasepool {
            [it->second->window setTitle:[NSString stringWithUTF8String:title.c_str()]];
        }
    }
}

// Open file dialog
std::string openFileDialog(const std::string& title,
                           const std::string& filter = "") {
    @autoreleasepool {
        NSOpenPanel* panel = [NSOpenPanel openPanel];
        [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];
        
        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [[panel URLs] objectAtIndex:0];
            return [[url path] UTF8String];
        }
    }
    return "";
}

// Save file dialog
std::string saveFileDialog(const std::string& title,
                           const std::string& filter = "") {
    @autoreleasepool {
        NSSavePanel* panel = [NSSavePanel savePanel];
        [panel setTitle:[NSString stringWithUTF8String:title.c_str()]];
        
        if ([panel runModal] == NSModalResponseOK) {
            NSURL* url = [panel URL];
            return [[url path] UTF8String];
        }
    }
    return "";
}

// Message box
void showMessage(const std::string& title, const std::string& message) {
    @autoreleasepool {
        NSAlert* alert = [[NSAlert alloc] init];
        [alert setMessageText:[NSString stringWithUTF8String:title.c_str()]];
        [alert setInformativeText:[NSString stringWithUTF8String:message.c_str()]];
        [alert runModal];
    }
}

// Clipboard operations
bool copyToClipboard(const unsigned char* data, int width, int height, int channels) {
    @autoreleasepool {
        // Create bitmap
        NSBitmapImageRep* bitmap = [[NSBitmapImageRep alloc]
            initWithBitmapDataPlanes:NULL
            pixelsWide:width
            pixelsHigh:height
            bitsPerSample:8
            samplesPerPixel:channels == 1 ? 1 : (channels == 4 ? 4 : 3)
            hasAlpha:(channels == 4)
            isPlanar:NO
            colorSpaceName:channels == 1 ? NSDeviceWhiteColorSpace : NSDeviceRGBColorSpace
            bytesPerRow:width * (channels == 1 ? 1 : (channels == 4 ? 4 : 3))
            bitsPerPixel:0];
        
        memcpy([bitmap bitmapData], data, width * height * (channels == 4 ? 4 : (channels == 1 ? 1 : 3)));
        
        // Create image
        NSImage* image = [[NSImage alloc] initWithSize:NSMakeSize(width, height)];
        [image addRepresentation:bitmap];
        
        // Copy to clipboard
        NSPasteboard* pasteboard = [NSPasteboard generalPasteboard];
        [pasteboard clearContents];
        return [pasteboard writeObjects:@[image]];
    }
}

} // namespace cocoa
} // namespace gui
} // namespace neurova

#endif // __APPLE__
