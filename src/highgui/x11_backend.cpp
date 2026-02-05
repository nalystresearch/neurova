/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova X11 Window Backend
 * Linux/Unix X11 window display implementation
 */

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <cstring>

#if defined(__linux__) || defined(__unix__)
#ifndef NEUROVA_USE_WAYLAND

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/extensions/XShm.h>
#include <sys/shm.h>

namespace neurova {
namespace gui {
namespace x11 {

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type  
using TrackbarCallback = std::function<void(int value)>;

// Trackbar info
struct TrackbarInfo {
    std::string name;
    int* value_ptr;
    int min_value;
    int max_value;
    int x, y, width, height;
    TrackbarCallback callback;
};

// Window info
struct WindowInfo {
    Window window;
    GC gc;
    XImage* image;
    XShmSegmentInfo shm_info;
    bool use_shm;
    unsigned char* image_data;
    int width;
    int height;
    int channels;
    int trackbar_height;
    std::vector<TrackbarInfo> trackbars;
    MouseCallback mouse_callback;
    bool closed;
};

// Global state
static Display* g_display = nullptr;
static std::unordered_map<std::string, WindowInfo*> g_windows;
static std::unordered_map<Window, std::string> g_window_names;
static std::mutex g_mutex;
static int g_last_key = -1;
static Atom g_wm_delete_message;

// Initialize X11
bool init() {
    if (g_display) return true;
    
    g_display = XOpenDisplay(nullptr);
    if (!g_display) {
        return false;
    }
    
    // Get WM_DELETE_WINDOW atom for close button handling
    g_wm_delete_message = XInternAtom(g_display, "WM_DELETE_WINDOW", False);
    
    return true;
}

// Cleanup
void cleanup() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        if (info->image) {
            if (info->use_shm) {
                XShmDetach(g_display, &info->shm_info);
                shmdt(info->shm_info.shmaddr);
                shmctl(info->shm_info.shmid, IPC_RMID, nullptr);
            }
            XDestroyImage(info->image);
        }
        
        XFreeGC(g_display, info->gc);
        XDestroyWindow(g_display, info->window);
        delete info;
    }
    
    g_windows.clear();
    g_window_names.clear();
    
    if (g_display) {
        XCloseDisplay(g_display);
        g_display = nullptr;
    }
}

// Check for XShm extension
bool checkShmExtension() {
    int major, minor;
    Bool pixmaps;
    return XShmQueryVersion(g_display, &major, &minor, &pixmaps) == True;
}

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_windows.find(name) != g_windows.end()) {
        return;  // Window already exists
    }
    
    WindowInfo* info = new WindowInfo();
    info->image = nullptr;
    info->image_data = nullptr;
    info->width = 640;
    info->height = 480;
    info->channels = 0;
    info->trackbar_height = 0;
    info->use_shm = checkShmExtension();
    info->closed = false;
    
    // Get screen info
    int screen = DefaultScreen(g_display);
    
    // Create window
    info->window = XCreateSimpleWindow(
        g_display, RootWindow(g_display, screen),
        100, 100, 640, 480, 1,
        BlackPixel(g_display, screen),
        WhitePixel(g_display, screen));
    
    // Set window title
    XStoreName(g_display, info->window, name.c_str());
    
    // Select events
    XSelectInput(g_display, info->window,
                 ExposureMask | KeyPressMask | KeyReleaseMask |
                 ButtonPressMask | ButtonReleaseMask |
                 PointerMotionMask | StructureNotifyMask);
    
    // Handle window close button
    XSetWMProtocols(g_display, info->window, &g_wm_delete_message, 1);
    
    // Create graphics context
    info->gc = XCreateGC(g_display, info->window, 0, nullptr);
    
    // Map window
    XMapWindow(g_display, info->window);
    XFlush(g_display);
    
    g_windows[name] = info;
    g_window_names[info->window] = name;
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
    
    int screen = DefaultScreen(g_display);
    int depth = DefaultDepth(g_display, screen);
    Visual* visual = DefaultVisual(g_display, screen);
    
    // Recreate image if size changed
    if (info->width != width || info->height != height || !info->image) {
        if (info->image) {
            if (info->use_shm) {
                XShmDetach(g_display, &info->shm_info);
                shmdt(info->shm_info.shmaddr);
                shmctl(info->shm_info.shmid, IPC_RMID, nullptr);
            }
            XDestroyImage(info->image);
            info->image = nullptr;
        }
        
        info->width = width;
        info->height = height;
        
        // Try to use shared memory for better performance
        if (info->use_shm) {
            info->image = XShmCreateImage(g_display, visual, depth, ZPixmap,
                                          nullptr, &info->shm_info, width, height);
            
            if (info->image) {
                info->shm_info.shmid = shmget(IPC_PRIVATE,
                    info->image->bytes_per_line * height, IPC_CREAT | 0777);
                
                if (info->shm_info.shmid >= 0) {
                    info->shm_info.shmaddr = info->image->data = 
                        (char*)shmat(info->shm_info.shmid, nullptr, 0);
                    info->shm_info.readOnly = False;
                    
                    if (!XShmAttach(g_display, &info->shm_info)) {
                        info->use_shm = false;
                        shmdt(info->shm_info.shmaddr);
                        shmctl(info->shm_info.shmid, IPC_RMID, nullptr);
                        XDestroyImage(info->image);
                        info->image = nullptr;
                    }
                } else {
                    info->use_shm = false;
                    XDestroyImage(info->image);
                    info->image = nullptr;
                }
            } else {
                info->use_shm = false;
            }
        }
        
        // Fallback to regular image
        if (!info->image) {
            info->image_data = (unsigned char*)malloc(width * height * 4);
            info->image = XCreateImage(g_display, visual, depth, ZPixmap, 0,
                                       (char*)info->image_data, width, height, 32, 0);
        }
        
        // Resize window
        XResizeWindow(g_display, info->window, width, 
                     height + info->trackbar_height);
    }
    
    info->channels = channels;
    
    // Convert to BGRA for X11
    unsigned char* dst = (unsigned char*)info->image->data;
    int bytes_per_line = info->image->bytes_per_line;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * channels;
            int dst_idx = y * bytes_per_line + x * 4;
            
            if (channels == 1) {
                // Grayscale
                dst[dst_idx + 0] = data[src_idx];  // B
                dst[dst_idx + 1] = data[src_idx];  // G
                dst[dst_idx + 2] = data[src_idx];  // R
                dst[dst_idx + 3] = 255;            // A
            } else if (channels == 3) {
                // RGB to BGRA
                dst[dst_idx + 0] = data[src_idx + 2];  // B
                dst[dst_idx + 1] = data[src_idx + 1];  // G
                dst[dst_idx + 2] = data[src_idx + 0];  // R
                dst[dst_idx + 3] = 255;                 // A
            } else if (channels == 4) {
                // RGBA to BGRA
                dst[dst_idx + 0] = data[src_idx + 2];  // B
                dst[dst_idx + 1] = data[src_idx + 1];  // G
                dst[dst_idx + 2] = data[src_idx + 0];  // R
                dst[dst_idx + 3] = data[src_idx + 3];  // A
            }
        }
    }
    
    // Display image
    if (info->use_shm) {
        XShmPutImage(g_display, info->window, info->gc, info->image,
                     0, 0, 0, 0, width, height, False);
    } else {
        XPutImage(g_display, info->window, info->gc, info->image,
                  0, 0, 0, 0, width, height);
    }
    
    // Draw trackbars
    drawTrackbars(info);
    
    XFlush(g_display);
}

// Draw trackbars
void drawTrackbars(WindowInfo* info) {
    if (info->trackbars.empty()) return;
    
    int screen = DefaultScreen(g_display);
    
    for (const auto& tb : info->trackbars) {
        // Draw background
        XSetForeground(g_display, info->gc, WhitePixel(g_display, screen));
        XFillRectangle(g_display, info->window, info->gc,
                      0, info->height + tb.y, info->width, 30);
        
        // Draw label
        XSetForeground(g_display, info->gc, BlackPixel(g_display, screen));
        XDrawString(g_display, info->window, info->gc,
                   5, info->height + tb.y + 20,
                   tb.name.c_str(), tb.name.length());
        
        // Draw track
        int track_x = 100;
        int track_width = info->width - 160;
        int track_y = info->height + tb.y + 12;
        
        XSetForeground(g_display, info->gc, 0x808080);
        XFillRectangle(g_display, info->window, info->gc,
                      track_x, track_y, track_width, 6);
        
        // Draw slider position
        int value = tb.value_ptr ? *tb.value_ptr : 0;
        float ratio = (float)(value - tb.min_value) / (tb.max_value - tb.min_value);
        int slider_x = track_x + (int)(ratio * track_width) - 5;
        
        XSetForeground(g_display, info->gc, 0x0000FF);
        XFillRectangle(g_display, info->window, info->gc,
                      slider_x, track_y - 4, 10, 14);
        
        // Draw value
        char value_str[32];
        snprintf(value_str, sizeof(value_str), "%d", value);
        XSetForeground(g_display, info->gc, BlackPixel(g_display, screen));
        XDrawString(g_display, info->window, info->gc,
                   info->width - 50, info->height + tb.y + 20,
                   value_str, strlen(value_str));
    }
}

// Handle trackbar mouse event
bool handleTrackbarClick(WindowInfo* info, int x, int y) {
    for (auto& tb : info->trackbars) {
        int track_x = 100;
        int track_width = info->width - 160;
        int track_y = info->height + tb.y + 12;
        
        if (y >= track_y - 10 && y <= track_y + 20 &&
            x >= track_x && x <= track_x + track_width) {
            
            float ratio = (float)(x - track_x) / track_width;
            int value = tb.min_value + (int)(ratio * (tb.max_value - tb.min_value));
            value = std::max(tb.min_value, std::min(tb.max_value, value));
            
            if (tb.value_ptr) {
                *tb.value_ptr = value;
            }
            
            if (tb.callback) {
                tb.callback(value);
            }
            
            return true;
        }
    }
    return false;
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_display) return -1;
    
    g_last_key = -1;
    
    struct timeval tv;
    fd_set fds;
    int x11_fd = ConnectionNumber(g_display);
    
    auto start = std::chrono::steady_clock::now();
    
    while (g_last_key == -1) {
        // Process pending events
        while (XPending(g_display)) {
            XEvent event;
            XNextEvent(g_display, &event);
            
            switch (event.type) {
                case KeyPress: {
                    KeySym keysym = XLookupKeysym(&event.xkey, 0);
                    g_last_key = keysym & 0xFF;  // Get ASCII value
                    break;
                }
                
                case ButtonPress: {
                    auto name_it = g_window_names.find(event.xbutton.window);
                    if (name_it != g_window_names.end()) {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        auto win_it = g_windows.find(name_it->second);
                        if (win_it != g_windows.end()) {
                            WindowInfo* info = win_it->second;
                            
                            // Check trackbar
                            if (handleTrackbarClick(info, event.xbutton.x, event.xbutton.y)) {
                                drawTrackbars(info);
                                XFlush(g_display);
                            } else if (info->mouse_callback) {
                                int btn = 0;
                                switch (event.xbutton.button) {
                                    case Button1: btn = 1; break;
                                    case Button2: btn = 3; break;
                                    case Button3: btn = 2; break;
                                }
                                info->mouse_callback(btn, event.xbutton.x, 
                                                    event.xbutton.y, btn);
                            }
                        }
                    }
                    break;
                }
                
                case ButtonRelease: {
                    auto name_it = g_window_names.find(event.xbutton.window);
                    if (name_it != g_window_names.end()) {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        auto win_it = g_windows.find(name_it->second);
                        if (win_it != g_windows.end() && win_it->second->mouse_callback) {
                            win_it->second->mouse_callback(4, event.xbutton.x,
                                                          event.xbutton.y, 0);
                        }
                    }
                    break;
                }
                
                case MotionNotify: {
                    auto name_it = g_window_names.find(event.xmotion.window);
                    if (name_it != g_window_names.end()) {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        auto win_it = g_windows.find(name_it->second);
                        if (win_it != g_windows.end()) {
                            WindowInfo* info = win_it->second;
                            
                            // Handle trackbar drag
                            if (event.xmotion.state & Button1Mask) {
                                if (handleTrackbarClick(info, event.xmotion.x, event.xmotion.y)) {
                                    drawTrackbars(info);
                                    XFlush(g_display);
                                }
                            }
                            
                            if (info->mouse_callback) {
                                int flags = 0;
                                if (event.xmotion.state & Button1Mask) flags |= 1;
                                if (event.xmotion.state & Button3Mask) flags |= 2;
                                if (event.xmotion.state & Button2Mask) flags |= 4;
                                info->mouse_callback(0, event.xmotion.x,
                                                    event.xmotion.y, flags);
                            }
                        }
                    }
                    break;
                }
                
                case ClientMessage: {
                    if ((Atom)event.xclient.data.l[0] == g_wm_delete_message) {
                        auto name_it = g_window_names.find(event.xclient.window);
                        if (name_it != g_window_names.end()) {
                            std::lock_guard<std::mutex> lock(g_mutex);
                            auto win_it = g_windows.find(name_it->second);
                            if (win_it != g_windows.end()) {
                                win_it->second->closed = true;
                            }
                        }
                    }
                    break;
                }
                
                case Expose: {
                    // Redraw on expose
                    auto name_it = g_window_names.find(event.xexpose.window);
                    if (name_it != g_window_names.end()) {
                        std::lock_guard<std::mutex> lock(g_mutex);
                        auto win_it = g_windows.find(name_it->second);
                        if (win_it != g_windows.end()) {
                            WindowInfo* info = win_it->second;
                            if (info->image) {
                                if (info->use_shm) {
                                    XShmPutImage(g_display, info->window, info->gc, 
                                                info->image, 0, 0, 0, 0,
                                                info->width, info->height, False);
                                } else {
                                    XPutImage(g_display, info->window, info->gc,
                                             info->image, 0, 0, 0, 0,
                                             info->width, info->height);
                                }
                                drawTrackbars(info);
                            }
                        }
                    }
                    break;
                }
            }
        }
        
        if (delay > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed >= delay) break;
            
            // Wait with select
            FD_ZERO(&fds);
            FD_SET(x11_fd, &fds);
            
            int remaining = delay - elapsed;
            tv.tv_sec = remaining / 1000;
            tv.tv_usec = (remaining % 1000) * 1000;
            
            select(x11_fd + 1, &fds, nullptr, nullptr, &tv);
        } else if (delay == 0) {
            break;  // Non-blocking
        }
    }
    
    return g_last_key;
}

// Destroy window
void destroyWindow(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        WindowInfo* info = it->second;
        
        g_window_names.erase(info->window);
        
        if (info->image) {
            if (info->use_shm) {
                XShmDetach(g_display, &info->shm_info);
                shmdt(info->shm_info.shmaddr);
                shmctl(info->shm_info.shmid, IPC_RMID, nullptr);
            }
            XDestroyImage(info->image);
        }
        
        XFreeGC(g_display, info->gc);
        XDestroyWindow(g_display, info->window);
        delete info;
        
        g_windows.erase(it);
        XFlush(g_display);
    }
}

// Destroy all windows
void destroyAllWindows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        if (info->image) {
            if (info->use_shm) {
                XShmDetach(g_display, &info->shm_info);
                shmdt(info->shm_info.shmaddr);
                shmctl(info->shm_info.shmid, IPC_RMID, nullptr);
            }
            XDestroyImage(info->image);
        }
        
        XFreeGC(g_display, info->gc);
        XDestroyWindow(g_display, info->window);
        delete info;
    }
    
    g_windows.clear();
    g_window_names.clear();
    XFlush(g_display);
}

// Create trackbar
void createTrackbar(const std::string& trackbar_name,
                    const std::string& window_name,
                    int* value, int max_value,
                    TrackbarCallback callback = nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it == g_windows.end()) return;
    
    WindowInfo* info = it->second;
    
    TrackbarInfo tb;
    tb.name = trackbar_name;
    tb.value_ptr = value;
    tb.min_value = 0;
    tb.max_value = max_value;
    tb.y = info->trackbar_height;
    tb.callback = callback;
    
    info->trackbars.push_back(tb);
    info->trackbar_height += 30;
    
    // Resize window
    XResizeWindow(g_display, info->window, info->width,
                 info->height + info->trackbar_height);
    XFlush(g_display);
}

// Set mouse callback
void setMouseCallback(const std::string& window_name, MouseCallback callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it != g_windows.end()) {
        it->second->mouse_callback = callback;
    }
}

// Move window
void moveWindow(const std::string& name, int x, int y) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        XMoveWindow(g_display, it->second->window, x, y);
        XFlush(g_display);
    }
}

// Resize window
void resizeWindow(const std::string& name, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        XResizeWindow(g_display, it->second->window, width, height);
        XFlush(g_display);
    }
}

// Set window title
void setWindowTitle(const std::string& name, const std::string& title) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        XStoreName(g_display, it->second->window, title.c_str());
        XFlush(g_display);
    }
}

} // namespace x11
} // namespace gui
} // namespace neurova

#endif // NEUROVA_USE_WAYLAND
#endif // __linux__ || __unix__
