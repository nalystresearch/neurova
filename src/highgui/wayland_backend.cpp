/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/**
 * Neurova Wayland Window Backend
 * Linux Wayland window display implementation
 */

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#if defined(__linux__) && defined(NEUROVA_USE_WAYLAND)

#include <wayland-client.h>
#include <wayland-client-protocol.h>
#include <xdg-shell-client-protocol.h>
#include <linux/input-event-codes.h>

namespace neurova {
namespace gui {
namespace wayland {

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type
using TrackbarCallback = std::function<void(int value)>;

// Shared memory buffer
struct ShmBuffer {
    wl_buffer* buffer;
    void* data;
    int width;
    int height;
    int stride;
    size_t size;
    bool busy;
};

// Window info
struct WindowInfo {
    std::string name;
    wl_surface* surface;
    xdg_surface* xdg_surface;
    xdg_toplevel* xdg_toplevel;
    ShmBuffer* current_buffer;
    ShmBuffer buffers[2];  // Double buffering
    int width;
    int height;
    int channels;
    MouseCallback mouse_callback;
    int mouse_x;
    int mouse_y;
    bool closed;
    bool configured;
};

// Global Wayland state
static wl_display* g_display = nullptr;
static wl_registry* g_registry = nullptr;
static wl_compositor* g_compositor = nullptr;
static wl_shm* g_shm = nullptr;
static wl_seat* g_seat = nullptr;
static wl_pointer* g_pointer = nullptr;
static wl_keyboard* g_keyboard = nullptr;
static xdg_wm_base* g_xdg_wm_base = nullptr;

static std::unordered_map<std::string, WindowInfo*> g_windows;
static std::unordered_map<wl_surface*, std::string> g_surface_names;
static std::mutex g_mutex;
static int g_last_key = -1;
static WindowInfo* g_pointer_focus = nullptr;

// Forward declarations
static void registryHandler(void* data, wl_registry* registry, uint32_t id,
                           const char* interface, uint32_t version);
static void registryRemover(void* data, wl_registry* registry, uint32_t id);

static const wl_registry_listener registry_listener = {
    registryHandler,
    registryRemover
};

// XDG WM base listener
static void xdgWmBasePing(void* data, xdg_wm_base* wm_base, uint32_t serial) {
    xdg_wm_base_pong(wm_base, serial);
}

static const xdg_wm_base_listener xdg_wm_base_listener = {
    xdgWmBasePing
};

// XDG surface listener
static void xdgSurfaceConfigure(void* data, xdg_surface* surface, uint32_t serial) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    xdg_surface_ack_configure(surface, serial);
    info->configured = true;
}

static const xdg_surface_listener xdg_surface_listener = {
    xdgSurfaceConfigure
};

// XDG toplevel listener
static void xdgToplevelConfigure(void* data, xdg_toplevel* toplevel,
                                  int32_t width, int32_t height,
                                  wl_array* states) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    
    if (width > 0 && height > 0) {
        info->width = width;
        info->height = height;
    }
}

static void xdgToplevelClose(void* data, xdg_toplevel* toplevel) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    info->closed = true;
}

static const xdg_toplevel_listener xdg_toplevel_listener = {
    xdgToplevelConfigure,
    xdgToplevelClose
};

// Pointer listener
static void pointerEnter(void* data, wl_pointer* pointer, uint32_t serial,
                         wl_surface* surface, wl_fixed_t sx, wl_fixed_t sy) {
    auto it = g_surface_names.find(surface);
    if (it != g_surface_names.end()) {
        auto win_it = g_windows.find(it->second);
        if (win_it != g_windows.end()) {
            g_pointer_focus = win_it->second;
        }
    }
}

static void pointerLeave(void* data, wl_pointer* pointer, uint32_t serial,
                         wl_surface* surface) {
    g_pointer_focus = nullptr;
}

static void pointerMotion(void* data, wl_pointer* pointer, uint32_t time,
                          wl_fixed_t sx, wl_fixed_t sy) {
    if (g_pointer_focus) {
        int x = wl_fixed_to_int(sx);
        int y = wl_fixed_to_int(sy);
        g_pointer_focus->mouse_x = x;
        g_pointer_focus->mouse_y = y;
        
        if (g_pointer_focus->mouse_callback) {
            g_pointer_focus->mouse_callback(0, x, y, 0);
        }
    }
}

static void pointerButton(void* data, wl_pointer* pointer, uint32_t serial,
                          uint32_t time, uint32_t button, uint32_t state) {
    if (g_pointer_focus && g_pointer_focus->mouse_callback) {
        int event = 0;
        int flags = 0;
        
        if (state == WL_POINTER_BUTTON_STATE_PRESSED) {
            switch (button) {
                case BTN_LEFT: event = 1; flags = 1; break;
                case BTN_RIGHT: event = 2; flags = 2; break;
                case BTN_MIDDLE: event = 3; flags = 4; break;
            }
        } else {
            event = 4;  // Release
        }
        
        g_pointer_focus->mouse_callback(event, g_pointer_focus->mouse_x,
                                        g_pointer_focus->mouse_y, flags);
    }
}

static void pointerAxis(void* data, wl_pointer* pointer, uint32_t time,
                        uint32_t axis, wl_fixed_t value) {
    if (g_pointer_focus && g_pointer_focus->mouse_callback) {
        int delta = wl_fixed_to_int(value);
        g_pointer_focus->mouse_callback(10, g_pointer_focus->mouse_x,
                                        g_pointer_focus->mouse_y, delta);
    }
}

static const wl_pointer_listener pointer_listener = {
    pointerEnter,
    pointerLeave,
    pointerMotion,
    pointerButton,
    pointerAxis
};

// Keyboard listener
static void keyboardKeymap(void* data, wl_keyboard* keyboard, uint32_t format,
                           int32_t fd, uint32_t size) {
    close(fd);
}

static void keyboardEnter(void* data, wl_keyboard* keyboard, uint32_t serial,
                          wl_surface* surface, wl_array* keys) {}

static void keyboardLeave(void* data, wl_keyboard* keyboard, uint32_t serial,
                          wl_surface* surface) {}

static void keyboardKey(void* data, wl_keyboard* keyboard, uint32_t serial,
                        uint32_t time, uint32_t key, uint32_t state) {
    if (state == WL_KEYBOARD_KEY_STATE_PRESSED) {
        g_last_key = key;
    }
}

static void keyboardModifiers(void* data, wl_keyboard* keyboard, uint32_t serial,
                              uint32_t mods_depressed, uint32_t mods_latched,
                              uint32_t mods_locked, uint32_t group) {}

static const wl_keyboard_listener keyboard_listener = {
    keyboardKeymap,
    keyboardEnter,
    keyboardLeave,
    keyboardKey,
    keyboardModifiers
};

// Seat listener
static void seatCapabilities(void* data, wl_seat* seat, uint32_t caps) {
    if (caps & WL_SEAT_CAPABILITY_POINTER) {
        g_pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(g_pointer, &pointer_listener, nullptr);
    }
    
    if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
        g_keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(g_keyboard, &keyboard_listener, nullptr);
    }
}

static void seatName(void* data, wl_seat* seat, const char* name) {}

static const wl_seat_listener seat_listener = {
    seatCapabilities,
    seatName
};

// Shm format listener
static void shmFormat(void* data, wl_shm* shm, uint32_t format) {
    // We use ARGB8888
}

static const wl_shm_listener shm_listener = {
    shmFormat
};

// Registry handler
static void registryHandler(void* data, wl_registry* registry, uint32_t id,
                           const char* interface, uint32_t version) {
    if (strcmp(interface, wl_compositor_interface.name) == 0) {
        g_compositor = static_cast<wl_compositor*>(
            wl_registry_bind(registry, id, &wl_compositor_interface, 4));
    } else if (strcmp(interface, wl_shm_interface.name) == 0) {
        g_shm = static_cast<wl_shm*>(
            wl_registry_bind(registry, id, &wl_shm_interface, 1));
        wl_shm_add_listener(g_shm, &shm_listener, nullptr);
    } else if (strcmp(interface, wl_seat_interface.name) == 0) {
        g_seat = static_cast<wl_seat*>(
            wl_registry_bind(registry, id, &wl_seat_interface, 5));
        wl_seat_add_listener(g_seat, &seat_listener, nullptr);
    } else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
        g_xdg_wm_base = static_cast<xdg_wm_base*>(
            wl_registry_bind(registry, id, &xdg_wm_base_interface, 1));
        xdg_wm_base_add_listener(g_xdg_wm_base, &xdg_wm_base_listener, nullptr);
    }
}

static void registryRemover(void* data, wl_registry* registry, uint32_t id) {}

// Buffer release callback
static void bufferRelease(void* data, wl_buffer* buffer) {
    ShmBuffer* shm_buf = static_cast<ShmBuffer*>(data);
    shm_buf->busy = false;
}

static const wl_buffer_listener buffer_listener = {
    bufferRelease
};

// Create shared memory file
static int createShmFile(size_t size) {
    char name[32];
    snprintf(name, sizeof(name), "/neurova-shm-%d", getpid());
    
    int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    shm_unlink(name);
    
    if (fd < 0) return -1;
    
    if (ftruncate(fd, size) < 0) {
        close(fd);
        return -1;
    }
    
    return fd;
}

// Create buffer
static bool createBuffer(ShmBuffer* buffer, int width, int height) {
    buffer->width = width;
    buffer->height = height;
    buffer->stride = width * 4;
    buffer->size = buffer->stride * height;
    
    int fd = createShmFile(buffer->size);
    if (fd < 0) return false;
    
    buffer->data = mmap(nullptr, buffer->size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    
    if (buffer->data == MAP_FAILED) {
        close(fd);
        return false;
    }
    
    wl_shm_pool* pool = wl_shm_create_pool(g_shm, fd, buffer->size);
    buffer->buffer = wl_shm_pool_create_buffer(pool, 0, width, height,
                                                buffer->stride,
                                                WL_SHM_FORMAT_ARGB8888);
    wl_shm_pool_destroy(pool);
    close(fd);
    
    wl_buffer_add_listener(buffer->buffer, &buffer_listener, buffer);
    buffer->busy = false;
    
    return true;
}

// Destroy buffer
static void destroyBuffer(ShmBuffer* buffer) {
    if (buffer->buffer) {
        wl_buffer_destroy(buffer->buffer);
        buffer->buffer = nullptr;
    }
    if (buffer->data) {
        munmap(buffer->data, buffer->size);
        buffer->data = nullptr;
    }
}

// Initialize Wayland
bool init() {
    if (g_display) return true;
    
    g_display = wl_display_connect(nullptr);
    if (!g_display) {
        return false;
    }
    
    g_registry = wl_display_get_registry(g_display);
    wl_registry_add_listener(g_registry, &registry_listener, nullptr);
    
    wl_display_roundtrip(g_display);
    
    if (!g_compositor || !g_shm || !g_xdg_wm_base) {
        cleanup();
        return false;
    }
    
    return true;
}

// Cleanup
void cleanup() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        destroyBuffer(&info->buffers[0]);
        destroyBuffer(&info->buffers[1]);
        
        if (info->xdg_toplevel) xdg_toplevel_destroy(info->xdg_toplevel);
        if (info->xdg_surface) xdg_surface_destroy(info->xdg_surface);
        if (info->surface) wl_surface_destroy(info->surface);
        
        delete info;
    }
    
    g_windows.clear();
    g_surface_names.clear();
    
    if (g_pointer) wl_pointer_destroy(g_pointer);
    if (g_keyboard) wl_keyboard_destroy(g_keyboard);
    if (g_seat) wl_seat_destroy(g_seat);
    if (g_shm) wl_shm_destroy(g_shm);
    if (g_xdg_wm_base) xdg_wm_base_destroy(g_xdg_wm_base);
    if (g_compositor) wl_compositor_destroy(g_compositor);
    if (g_registry) wl_registry_destroy(g_registry);
    if (g_display) wl_display_disconnect(g_display);
    
    g_display = nullptr;
    g_registry = nullptr;
    g_compositor = nullptr;
    g_shm = nullptr;
    g_seat = nullptr;
    g_pointer = nullptr;
    g_keyboard = nullptr;
    g_xdg_wm_base = nullptr;
}

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_windows.find(name) != g_windows.end()) {
        return;
    }
    
    WindowInfo* info = new WindowInfo();
    info->name = name;
    info->width = 640;
    info->height = 480;
    info->channels = 0;
    info->closed = false;
    info->configured = false;
    info->current_buffer = nullptr;
    
    // Create surface
    info->surface = wl_compositor_create_surface(g_compositor);
    
    // Create xdg surface
    info->xdg_surface = xdg_wm_base_get_xdg_surface(g_xdg_wm_base, info->surface);
    xdg_surface_add_listener(info->xdg_surface, &xdg_surface_listener, info);
    
    // Create toplevel
    info->xdg_toplevel = xdg_surface_get_toplevel(info->xdg_surface);
    xdg_toplevel_add_listener(info->xdg_toplevel, &xdg_toplevel_listener, info);
    xdg_toplevel_set_title(info->xdg_toplevel, name.c_str());
    
    wl_surface_commit(info->surface);
    
    // Create initial buffers
    createBuffer(&info->buffers[0], info->width, info->height);
    createBuffer(&info->buffers[1], info->width, info->height);
    
    g_windows[name] = info;
    g_surface_names[info->surface] = name;
    
    // Wait for configure
    while (!info->configured) {
        wl_display_roundtrip(g_display);
    }
}

// Get available buffer
static ShmBuffer* getBuffer(WindowInfo* info) {
    if (!info->buffers[0].busy) return &info->buffers[0];
    if (!info->buffers[1].busy) return &info->buffers[1];
    return nullptr;
}

// Display image in window
void imshow(const std::string& name, const unsigned char* data,
            int width, int height, int channels) {
    if (!init()) return;
    
    // Create window if needed
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
    
    // Resize buffers if needed
    if (info->width != width || info->height != height) {
        destroyBuffer(&info->buffers[0]);
        destroyBuffer(&info->buffers[1]);
        info->width = width;
        info->height = height;
        createBuffer(&info->buffers[0], width, height);
        createBuffer(&info->buffers[1], width, height);
    }
    
    info->channels = channels;
    
    // Get buffer
    ShmBuffer* buffer = getBuffer(info);
    if (!buffer) {
        wl_display_roundtrip(g_display);
        buffer = getBuffer(info);
        if (!buffer) return;
    }
    
    // Convert to ARGB
    uint32_t* dst = static_cast<uint32_t*>(buffer->data);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * channels;
            uint32_t pixel;
            
            if (channels == 1) {
                uint8_t v = data[src_idx];
                pixel = 0xFF000000 | (v << 16) | (v << 8) | v;
            } else if (channels == 3) {
                pixel = 0xFF000000 | (data[src_idx] << 16) |
                        (data[src_idx + 1] << 8) | data[src_idx + 2];
            } else if (channels == 4) {
                pixel = (data[src_idx + 3] << 24) | (data[src_idx] << 16) |
                        (data[src_idx + 1] << 8) | data[src_idx + 2];
            } else {
                pixel = 0xFF000000;
            }
            
            dst[y * width + x] = pixel;
        }
    }
    
    // Attach and commit
    buffer->busy = true;
    wl_surface_attach(info->surface, buffer->buffer, 0, 0);
    wl_surface_damage(info->surface, 0, 0, width, height);
    wl_surface_commit(info->surface);
    
    info->current_buffer = buffer;
    
    wl_display_flush(g_display);
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_display) return -1;
    
    g_last_key = -1;
    
    auto start = std::chrono::steady_clock::now();
    
    while (g_last_key == -1) {
        wl_display_dispatch_pending(g_display);
        
        if (delay > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
            if (elapsed >= delay) break;
            
            // Poll with timeout
            struct pollfd pfd = { wl_display_get_fd(g_display), POLLIN, 0 };
            int remaining = delay - elapsed;
            poll(&pfd, 1, remaining);
        } else if (delay == 0) {
            break;
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
        
        g_surface_names.erase(info->surface);
        
        destroyBuffer(&info->buffers[0]);
        destroyBuffer(&info->buffers[1]);
        
        if (info->xdg_toplevel) xdg_toplevel_destroy(info->xdg_toplevel);
        if (info->xdg_surface) xdg_surface_destroy(info->xdg_surface);
        if (info->surface) wl_surface_destroy(info->surface);
        
        delete info;
        g_windows.erase(it);
    }
}

// Destroy all windows
void destroyAllWindows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        destroyBuffer(&info->buffers[0]);
        destroyBuffer(&info->buffers[1]);
        
        if (info->xdg_toplevel) xdg_toplevel_destroy(info->xdg_toplevel);
        if (info->xdg_surface) xdg_surface_destroy(info->xdg_surface);
        if (info->surface) wl_surface_destroy(info->surface);
        
        delete info;
    }
    
    g_windows.clear();
    g_surface_names.clear();
}

// Set mouse callback
void setMouseCallback(const std::string& window_name, MouseCallback callback) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it != g_windows.end()) {
        it->second->mouse_callback = callback;
    }
}

// Set window title
void setWindowTitle(const std::string& name, const std::string& title) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        xdg_toplevel_set_title(it->second->xdg_toplevel, title.c_str());
    }
}

} // namespace wayland
} // namespace gui
} // namespace neurova

#endif // __linux__ && NEUROVA_USE_WAYLAND
