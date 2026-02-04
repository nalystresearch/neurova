/* copyright (c) 2025 @analytics withharry
 * all rights reserved.
 * licensed under the mit license.
 */

/**
 * Neurova GTK GUI Backend
 * Linux/Unix GUI using GTK3/GTK4
 */

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>

#ifdef NEUROVA_HAVE_GTK

#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include <cairo.h>

namespace neurova {
namespace gui {
namespace gtk {

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type
using TrackbarCallback = std::function<void(int value)>;

// Trackbar info
struct TrackbarInfo {
    GtkWidget* widget;
    GtkWidget* scale;
    GtkWidget* label;
    int* value_ptr;
    TrackbarCallback callback;
};

// Window info structure
struct WindowInfo {
    GtkWidget* window;
    GtkWidget* image;
    GtkWidget* vbox;
    cairo_surface_t* surface;
    unsigned char* image_data;
    int width;
    int height;
    int channels;
    MouseCallback mouse_callback;
    std::unordered_map<std::string, TrackbarInfo> trackbars;
    bool closed;
};

// Global state
static std::unordered_map<std::string, WindowInfo*> g_windows;
static std::mutex g_mutex;
static bool g_initialized = false;
static int g_last_key = -1;

// Initialize GTK
bool init() {
    if (g_initialized) return true;
    
    if (!gtk_init_check(nullptr, nullptr)) {
        return false;
    }
    
    g_initialized = true;
    return true;
}

// Cleanup
void cleanup() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        if (pair.second->surface) {
            cairo_surface_destroy(pair.second->surface);
        }
        if (pair.second->image_data) {
            delete[] pair.second->image_data;
        }
        gtk_widget_destroy(pair.second->window);
        delete pair.second;
    }
    g_windows.clear();
}

// Draw callback
static gboolean on_draw(GtkWidget* widget, cairo_t* cr, gpointer data) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    
    if (info->surface) {
        cairo_set_source_surface(cr, info->surface, 0, 0);
        cairo_paint(cr);
    }
    
    return FALSE;
}

// Window close callback
static gboolean on_delete(GtkWidget* widget, GdkEvent* event, gpointer data) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    info->closed = true;
    return TRUE;  // Don't destroy, just mark as closed
}

// Key press callback
static gboolean on_key_press(GtkWidget* widget, GdkEventKey* event, gpointer data) {
    g_last_key = event->keyval;
    return FALSE;
}

// Mouse motion callback
static gboolean on_motion(GtkWidget* widget, GdkEventMotion* event, gpointer data) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    
    if (info->mouse_callback) {
        int flags = 0;
        if (event->state & GDK_BUTTON1_MASK) flags |= 1;
        if (event->state & GDK_BUTTON2_MASK) flags |= 4;
        if (event->state & GDK_BUTTON3_MASK) flags |= 2;
        info->mouse_callback(0, (int)event->x, (int)event->y, flags);
    }
    
    return FALSE;
}

// Mouse button press callback
static gboolean on_button_press(GtkWidget* widget, GdkEventButton* event, gpointer data) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    
    if (info->mouse_callback) {
        int ev = 0;
        int flags = 0;
        
        switch (event->button) {
            case 1: ev = 1; flags = 1; break;  // Left button
            case 2: ev = 3; flags = 4; break;  // Middle button
            case 3: ev = 2; flags = 2; break;  // Right button
        }
        
        info->mouse_callback(ev, (int)event->x, (int)event->y, flags);
    }
    
    return FALSE;
}

// Mouse button release callback
static gboolean on_button_release(GtkWidget* widget, GdkEventButton* event, gpointer data) {
    WindowInfo* info = static_cast<WindowInfo*>(data);
    
    if (info->mouse_callback) {
        info->mouse_callback(4, (int)event->x, (int)event->y, 0);
    }
    
    return FALSE;
}

// Trackbar value changed callback
static void on_scale_changed(GtkRange* range, gpointer data) {
    TrackbarInfo* info = static_cast<TrackbarInfo*>(data);
    
    int value = (int)gtk_range_get_value(range);
    
    if (info->value_ptr) {
        *info->value_ptr = value;
    }
    
    if (info->label) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d", value);
        gtk_label_set_text(GTK_LABEL(info->label), buf);
    }
    
    if (info->callback) {
        info->callback(value);
    }
}

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_windows.find(name) != g_windows.end()) {
        return;  // Window already exists
    }
    
    WindowInfo* info = new WindowInfo();
    info->surface = nullptr;
    info->image_data = nullptr;
    info->width = 0;
    info->height = 0;
    info->channels = 0;
    info->closed = false;
    
    // Create window
    info->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(info->window), name.c_str());
    gtk_window_set_default_size(GTK_WINDOW(info->window), 640, 480);
    
    // Create vertical box for layout
    info->vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(info->window), info->vbox);
    
    // Create drawing area for image
    info->image = gtk_drawing_area_new();
    gtk_widget_set_size_request(info->image, 640, 480);
    gtk_box_pack_start(GTK_BOX(info->vbox), info->image, TRUE, TRUE, 0);
    
    // Enable events
    gtk_widget_add_events(info->image, 
        GDK_BUTTON_PRESS_MASK | 
        GDK_BUTTON_RELEASE_MASK |
        GDK_POINTER_MOTION_MASK |
        GDK_KEY_PRESS_MASK);
    
    gtk_widget_set_can_focus(info->window, TRUE);
    
    // Connect signals
    g_signal_connect(info->image, "draw", G_CALLBACK(on_draw), info);
    g_signal_connect(info->window, "delete-event", G_CALLBACK(on_delete), info);
    g_signal_connect(info->window, "key-press-event", G_CALLBACK(on_key_press), info);
    g_signal_connect(info->image, "motion-notify-event", G_CALLBACK(on_motion), info);
    g_signal_connect(info->image, "button-press-event", G_CALLBACK(on_button_press), info);
    g_signal_connect(info->image, "button-release-event", G_CALLBACK(on_button_release), info);
    
    gtk_widget_show_all(info->window);
    
    g_windows[name] = info;
}

// Display image in window
void imshow(const std::string& name, const unsigned char* data,
            int width, int height, int channels) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it == g_windows.end()) {
        // Create window if it doesn't exist
        g_mutex.unlock();
        namedWindow(name);
        g_mutex.lock();
        it = g_windows.find(name);
    }
    
    WindowInfo* info = it->second;
    
    // Allocate or reallocate image data
    int size = width * height * 4;  // Always use RGBA for cairo
    
    if (info->image_data == nullptr || 
        info->width != width || 
        info->height != height) {
        delete[] info->image_data;
        info->image_data = new unsigned char[size];
        info->width = width;
        info->height = height;
        
        gtk_widget_set_size_request(info->image, width, height);
    }
    
    // Convert to RGBA for cairo
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * channels;
            int dst_idx = (y * width + x) * 4;
            
            if (channels == 1) {
                // Grayscale to RGBA
                info->image_data[dst_idx + 0] = data[src_idx];
                info->image_data[dst_idx + 1] = data[src_idx];
                info->image_data[dst_idx + 2] = data[src_idx];
                info->image_data[dst_idx + 3] = 255;
            } else if (channels == 3) {
                // RGB to BGRA (cairo uses BGRA)
                info->image_data[dst_idx + 0] = data[src_idx + 2];  // B
                info->image_data[dst_idx + 1] = data[src_idx + 1];  // G
                info->image_data[dst_idx + 2] = data[src_idx + 0];  // R
                info->image_data[dst_idx + 3] = 255;                 // A
            } else if (channels == 4) {
                // RGBA to BGRA
                info->image_data[dst_idx + 0] = data[src_idx + 2];  // B
                info->image_data[dst_idx + 1] = data[src_idx + 1];  // G
                info->image_data[dst_idx + 2] = data[src_idx + 0];  // R
                info->image_data[dst_idx + 3] = data[src_idx + 3];  // A
            }
        }
    }
    
    // Create cairo surface
    if (info->surface) {
        cairo_surface_destroy(info->surface);
    }
    
    info->surface = cairo_image_surface_create_for_data(
        info->image_data, CAIRO_FORMAT_ARGB32,
        width, height, width * 4);
    
    info->channels = channels;
    
    // Request redraw
    gtk_widget_queue_draw(info->image);
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_initialized) return -1;
    
    g_last_key = -1;
    
    if (delay <= 0) {
        // Block until key press
        while (g_last_key == -1) {
            gtk_main_iteration_do(TRUE);
        }
    } else {
        // Wait for specified time or key press
        guint64 start = g_get_monotonic_time();
        guint64 end = start + delay * 1000;
        
        while (g_last_key == -1 && g_get_monotonic_time() < end) {
            gtk_main_iteration_do(FALSE);
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
        
        if (info->surface) {
            cairo_surface_destroy(info->surface);
        }
        if (info->image_data) {
            delete[] info->image_data;
        }
        
        gtk_widget_destroy(info->window);
        delete info;
        g_windows.erase(it);
    }
}

// Destroy all windows
void destroyAllWindows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        if (info->surface) {
            cairo_surface_destroy(info->surface);
        }
        if (info->image_data) {
            delete[] info->image_data;
        }
        
        gtk_widget_destroy(info->window);
        delete info;
    }
    g_windows.clear();
}

// Create trackbar
void createTrackbar(const std::string& trackbar_name,
                    const std::string& window_name,
                    int* value, int max_value,
                    TrackbarCallback callback = nullptr) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it == g_windows.end()) return;
    
    WindowInfo* winfo = it->second;
    
    // Create trackbar container
    GtkWidget* hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    
    // Label for trackbar name
    GtkWidget* name_label = gtk_label_new(trackbar_name.c_str());
    gtk_box_pack_start(GTK_BOX(hbox), name_label, FALSE, FALSE, 5);
    
    // Scale widget
    GtkWidget* scale = gtk_scale_new_with_range(
        GTK_ORIENTATION_HORIZONTAL, 0, max_value, 1);
    gtk_range_set_value(GTK_RANGE(scale), *value);
    gtk_scale_set_draw_value(GTK_SCALE(scale), FALSE);
    gtk_widget_set_hexpand(scale, TRUE);
    gtk_box_pack_start(GTK_BOX(hbox), scale, TRUE, TRUE, 0);
    
    // Value label
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", *value);
    GtkWidget* value_label = gtk_label_new(buf);
    gtk_box_pack_start(GTK_BOX(hbox), value_label, FALSE, FALSE, 5);
    
    // Store trackbar info
    TrackbarInfo& tinfo = winfo->trackbars[trackbar_name];
    tinfo.widget = hbox;
    tinfo.scale = scale;
    tinfo.label = value_label;
    tinfo.value_ptr = value;
    tinfo.callback = callback;
    
    // Connect signal
    g_signal_connect(scale, "value-changed", G_CALLBACK(on_scale_changed), &tinfo);
    
    // Add to window
    gtk_box_pack_start(GTK_BOX(winfo->vbox), hbox, FALSE, FALSE, 5);
    gtk_widget_show_all(hbox);
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
        gtk_window_move(GTK_WINDOW(it->second->window), x, y);
    }
}

// Resize window
void resizeWindow(const std::string& name, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        gtk_window_resize(GTK_WINDOW(it->second->window), width, height);
    }
}

// Get window image rect
void getWindowImageRect(const std::string& name, int& x, int& y, int& width, int& height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        GtkAllocation allocation;
        gtk_widget_get_allocation(it->second->image, &allocation);
        x = allocation.x;
        y = allocation.y;
        width = allocation.width;
        height = allocation.height;
    }
}

} // namespace gtk
} // namespace gui
} // namespace neurova

#endif // NEUROVA_HAVE_GTK
