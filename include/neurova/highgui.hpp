/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova HighGUI - Unified GUI Interface
 * Cross-platform image display and user interaction
 */

#ifndef NEUROVA_HIGHGUI_HPP
#define NEUROVA_HIGHGUI_HPP

#include <string>
#include <functional>
#include <vector>
#include <memory>

namespace neurova {
namespace highgui {

// Window flags
enum WindowFlags {
    WINDOW_NORMAL     = 0x00000000,  // User can resize
    WINDOW_AUTOSIZE   = 0x00000001,  // Auto-size to image
    WINDOW_OPENGL     = 0x00001000,  // OpenGL support
    WINDOW_FULLSCREEN = 0x00000002,  // Fullscreen
    WINDOW_FREERATIO  = 0x00000100,  // No aspect ratio constraint
    WINDOW_KEEPRATIO  = 0x00000000,  // Keep aspect ratio
    WINDOW_GUI_EXPANDED = 0x00000000,
    WINDOW_GUI_NORMAL   = 0x00000010
};

// Window properties
enum WindowPropertyFlags {
    WND_PROP_FULLSCREEN   = 0,
    WND_PROP_AUTOSIZE     = 1,
    WND_PROP_ASPECT_RATIO = 2,
    WND_PROP_OPENGL       = 3,
    WND_PROP_VISIBLE      = 4,
    WND_PROP_TOPMOST      = 5,
    WND_PROP_VSYNC        = 6
};

// Mouse events
enum MouseEventTypes {
    EVENT_MOUSEMOVE     = 0,
    EVENT_LBUTTONDOWN   = 1,
    EVENT_RBUTTONDOWN   = 2,
    EVENT_MBUTTONDOWN   = 3,
    EVENT_LBUTTONUP     = 4,
    EVENT_RBUTTONUP     = 5,
    EVENT_MBUTTONUP     = 6,
    EVENT_LBUTTONDBLCLK = 7,
    EVENT_RBUTTONDBLCLK = 8,
    EVENT_MBUTTONDBLCLK = 9,
    EVENT_MOUSEWHEEL    = 10,
    EVENT_MOUSEHWHEEL   = 11
};

// Mouse event flags
enum MouseEventFlags {
    EVENT_FLAG_LBUTTON  = 1,
    EVENT_FLAG_RBUTTON  = 2,
    EVENT_FLAG_MBUTTON  = 4,
    EVENT_FLAG_CTRLKEY  = 8,
    EVENT_FLAG_SHIFTKEY = 16,
    EVENT_FLAG_ALTKEY   = 32
};

// Qt button types
enum QtButtonTypes {
    QT_PUSH_BUTTON   = 0x00000000,
    QT_CHECKBOX      = 0x00000001,
    QT_RADIOBOX      = 0x00000002,
    QT_NEW_BUTTONBAR = 0x00000004
};

// Font styles
enum QtFontStyles {
    QT_STYLE_NORMAL  = 0,
    QT_STYLE_ITALIC  = 1,
    QT_STYLE_OBLIQUE = 2
};

enum QtFontWeight {
    QT_FONT_LIGHT  = 25,
    QT_FONT_NORMAL = 50,
    QT_FONT_DEMIBOLD = 63,
    QT_FONT_BOLD   = 75,
    QT_FONT_BLACK  = 87
};

// Callback types
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;
using TrackbarCallback = std::function<void(int value)>;
using ButtonCallback = std::function<void(int state)>;
using OpenGLCallback = std::function<void()>;

// Backend type
enum class Backend {
    AUTO,
    QT,
    GTK,
    WIN32,
    COCOA,
    WAYLAND
};

// Get/set active backend
Backend getBackend();
void setBackend(Backend backend);

// Window management
void namedWindow(const std::string& winname, int flags = WINDOW_AUTOSIZE);
void destroyWindow(const std::string& winname);
void destroyAllWindows();

// Image display
void imshow(const std::string& winname, const unsigned char* data,
            int width, int height, int channels);

// Wait for input
int waitKey(int delay = 0);
int waitKeyEx(int delay = 0);
int pollKey();

// Window properties
void moveWindow(const std::string& winname, int x, int y);
void resizeWindow(const std::string& winname, int width, int height);
void setWindowTitle(const std::string& winname, const std::string& title);
double getWindowProperty(const std::string& winname, int prop_id);
void setWindowProperty(const std::string& winname, int prop_id, double prop_value);

// Get window image dimensions
struct Rect {
    int x, y, width, height;
};
Rect getWindowImageRect(const std::string& winname);

// Trackbars
int createTrackbar(const std::string& trackbarname, const std::string& winname,
                   int* value, int count, TrackbarCallback onChange = nullptr);
int getTrackbarPos(const std::string& trackbarname, const std::string& winname);
void setTrackbarPos(const std::string& trackbarname, const std::string& winname, int pos);
void setTrackbarMax(const std::string& trackbarname, const std::string& winname, int maxval);
void setTrackbarMin(const std::string& trackbarname, const std::string& winname, int minval);

// Mouse callbacks
void setMouseCallback(const std::string& winname, MouseCallback onMouse);

// Buttons (Qt-specific)
int createButton(const std::string& bar_name, ButtonCallback on_change = nullptr,
                 int button_type = QT_PUSH_BUTTON, bool initial_button_state = false);

// OpenGL support
void setOpenGlDrawCallback(const std::string& winname, OpenGLCallback onOpenGlDraw);
void setOpenGlContext(const std::string& winname);
void updateWindow(const std::string& winname);

// Display information
int getScreenCount();
struct ScreenInfo {
    int x, y;
    int width, height;
    bool is_primary;
};
ScreenInfo getScreenInfo(int screen = 0);

// Clipboard operations
void setClipboardImage(const unsigned char* data, int width, int height, int channels);
bool getClipboardImage(std::vector<unsigned char>& data, int& width, int& height, int& channels);

// File dialogs
std::string selectFile(const std::string& title = "Select File",
                       const std::string& filter = "",
                       const std::string& default_path = "");
std::string saveFile(const std::string& title = "Save File",
                     const std::string& filter = "",
                     const std::string& default_path = "");
std::string selectFolder(const std::string& title = "Select Folder",
                         const std::string& default_path = "");
std::vector<std::string> selectFiles(const std::string& title = "Select Files",
                                     const std::string& filter = "");

// Status bar (Qt-specific)
void displayOverlay(const std::string& winname, const std::string& text, int delayms = 0);
void displayStatusBar(const std::string& winname, const std::string& text, int delayms = 0);

// Font loading
int loadFont(const std::string& filepath, int font_size = 12, int style = QT_STYLE_NORMAL,
             int weight = QT_FONT_NORMAL);

// Message boxes
enum MessageBoxFlags {
    MB_OK              = 0x00000000,
    MB_OKCANCEL        = 0x00000001,
    MB_YESNO           = 0x00000004,
    MB_YESNOCANCEL     = 0x00000003,
    MB_ICON_INFO       = 0x00000040,
    MB_ICON_WARNING    = 0x00000030,
    MB_ICON_ERROR      = 0x00000010,
    MB_ICON_QUESTION   = 0x00000020
};

int messageBox(const std::string& title, const std::string& message, int flags = MB_OK);

// Progress indicator
void showProgress(const std::string& title, int progress, int max_progress = 100);
void hideProgress();

// Utility functions
void startWindowThread();
void stopWindowThread();

// Image annotations (draw on display)
void addText(const std::string& winname, const std::string& text,
             int x, int y, const std::string& font = "",
             int font_size = 12, int color_r = 255, int color_g = 255, int color_b = 255);

void addRect(const std::string& winname, int x, int y, int width, int height,
             int color_r = 255, int color_g = 0, int color_b = 0, int thickness = 1);

void addCircle(const std::string& winname, int cx, int cy, int radius,
               int color_r = 255, int color_g = 0, int color_b = 0, int thickness = 1);

void addLine(const std::string& winname, int x1, int y1, int x2, int y2,
             int color_r = 255, int color_g = 0, int color_b = 0, int thickness = 1);

void clearOverlay(const std::string& winname);

// Screenshot
bool saveScreenshot(const std::string& winname, const std::string& filename);

// Video recording
bool startRecording(const std::string& winname, const std::string& filename,
                    int fps = 30, const std::string& codec = "mp4v");
void stopRecording(const std::string& winname);
bool isRecording(const std::string& winname);

} // namespace highgui
} // namespace neurova

#endif // NEUROVA_HIGHGUI_HPP
