/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Win32 GUI Backend
 * Windows-native GUI using Win32 API
 */

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <thread>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>

#pragma comment(lib, "comctl32.lib")

namespace neurova {
namespace gui {
namespace win32 {

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type
using TrackbarCallback = std::function<void(int value)>;

// Trackbar info
struct TrackbarInfo {
    HWND trackbar;
    HWND label;
    HWND value_label;
    int* value_ptr;
    int max_value;
    TrackbarCallback callback;
};

// Window info structure
struct WindowInfo {
    HWND hwnd;
    HWND image_area;
    HDC hdc;
    HDC memory_dc;
    HBITMAP bitmap;
    unsigned char* image_data;
    BITMAPINFO bmi;
    int width;
    int height;
    int channels;
    int window_flags;
    MouseCallback mouse_callback;
    std::unordered_map<std::string, TrackbarInfo> trackbars;
    int trackbar_count;
    bool closed;
};

// Global state
static std::unordered_map<std::string, WindowInfo*> g_windows;
static std::mutex g_mutex;
static bool g_initialized = false;
static int g_last_key = -1;
static HINSTANCE g_instance = nullptr;
static const wchar_t* WINDOW_CLASS = L"NeurovaWindow";

// Forward declarations
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Initialize Win32
bool init() {
    if (g_initialized) return true;
    
    g_instance = GetModuleHandle(nullptr);
    
    // Initialize common controls
    INITCOMMONCONTROLSEX icc;
    icc.dwSize = sizeof(icc);
    icc.dwICC = ICC_BAR_CLASSES | ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icc);
    
    // Register window class
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = g_instance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = WINDOW_CLASS;
    
    if (!RegisterClassExW(&wc)) {
        return false;
    }
    
    g_initialized = true;
    return true;
}

// Get window info from HWND
WindowInfo* getWindowInfo(HWND hwnd) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        if (pair.second->hwnd == hwnd) {
            return pair.second;
        }
    }
    return nullptr;
}

// Window procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    WindowInfo* info = getWindowInfo(hwnd);
    
    switch (msg) {
        case WM_DESTROY:
            if (info) info->closed = true;
            PostQuitMessage(0);
            return 0;
            
        case WM_CLOSE:
            if (info) info->closed = true;
            return 0;
            
        case WM_KEYDOWN:
            g_last_key = (int)wParam;
            return 0;
            
        case WM_PAINT: {
            if (info && info->bitmap) {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                
                BitBlt(hdc, 0, 0, info->width, info->height,
                       info->memory_dc, 0, 0, SRCCOPY);
                
                EndPaint(hwnd, &ps);
            }
            return 0;
        }
        
        case WM_MOUSEMOVE: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                int flags = 0;
                if (wParam & MK_LBUTTON) flags |= 1;
                if (wParam & MK_RBUTTON) flags |= 2;
                if (wParam & MK_MBUTTON) flags |= 4;
                info->mouse_callback(0, x, y, flags);
            }
            return 0;
        }
        
        case WM_LBUTTONDOWN: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                info->mouse_callback(1, x, y, 1);
            }
            return 0;
        }
        
        case WM_RBUTTONDOWN: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                info->mouse_callback(2, x, y, 2);
            }
            return 0;
        }
        
        case WM_MBUTTONDOWN: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                info->mouse_callback(3, x, y, 4);
            }
            return 0;
        }
        
        case WM_LBUTTONUP:
        case WM_RBUTTONUP:
        case WM_MBUTTONUP: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                info->mouse_callback(4, x, y, 0);
            }
            return 0;
        }
        
        case WM_LBUTTONDBLCLK: {
            if (info && info->mouse_callback) {
                int x = GET_X_LPARAM(lParam);
                int y = GET_Y_LPARAM(lParam);
                info->mouse_callback(7, x, y, 1);
            }
            return 0;
        }
        
        case WM_HSCROLL: {
            if (info) {
                HWND trackbar = (HWND)lParam;
                for (auto& pair : info->trackbars) {
                    if (pair.second.trackbar == trackbar) {
                        int value = (int)SendMessage(trackbar, TBM_GETPOS, 0, 0);
                        
                        if (pair.second.value_ptr) {
                            *pair.second.value_ptr = value;
                        }
                        
                        // Update value label
                        if (pair.second.value_label) {
                            wchar_t buf[32];
                            swprintf(buf, 32, L"%d", value);
                            SetWindowTextW(pair.second.value_label, buf);
                        }
                        
                        if (pair.second.callback) {
                            pair.second.callback(value);
                        }
                        break;
                    }
                }
            }
            return 0;
        }
    }
    
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

// Convert UTF-8 to wide string
std::wstring utf8ToWide(const std::string& str) {
    if (str.empty()) return L"";
    
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring result(size - 1, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &result[0], size);
    
    return result;
}

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    if (!init()) return;
    
    std::lock_guard<std::mutex> lock(g_mutex);
    
    if (g_windows.find(name) != g_windows.end()) {
        return;  // Window already exists
    }
    
    WindowInfo* info = new WindowInfo();
    info->image_data = nullptr;
    info->bitmap = nullptr;
    info->memory_dc = nullptr;
    info->width = 640;
    info->height = 480;
    info->channels = 0;
    info->window_flags = flags;
    info->trackbar_count = 0;
    info->closed = false;
    
    // Window style
    DWORD style = WS_OVERLAPPEDWINDOW;
    if (flags & 1) {  // WINDOW_AUTOSIZE
        style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX;
    }
    
    // Create window
    std::wstring wname = utf8ToWide(name);
    info->hwnd = CreateWindowExW(
        0, WINDOW_CLASS, wname.c_str(), style,
        CW_USEDEFAULT, CW_USEDEFAULT, 640, 480,
        nullptr, nullptr, g_instance, nullptr);
    
    if (!info->hwnd) {
        delete info;
        return;
    }
    
    // Get DC
    info->hdc = GetDC(info->hwnd);
    info->memory_dc = CreateCompatibleDC(info->hdc);
    
    ShowWindow(info->hwnd, SW_SHOW);
    UpdateWindow(info->hwnd);
    
    g_windows[name] = info;
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
    
    // Recreate bitmap if size changed
    if (info->width != width || info->height != height || !info->bitmap) {
        if (info->bitmap) {
            DeleteObject(info->bitmap);
        }
        
        // Create DIB
        memset(&info->bmi, 0, sizeof(info->bmi));
        info->bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        info->bmi.bmiHeader.biWidth = width;
        info->bmi.bmiHeader.biHeight = -height;  // Top-down
        info->bmi.bmiHeader.biPlanes = 1;
        info->bmi.bmiHeader.biBitCount = 32;
        info->bmi.bmiHeader.biCompression = BI_RGB;
        
        info->bitmap = CreateDIBSection(info->memory_dc, &info->bmi,
                                        DIB_RGB_COLORS, (void**)&info->image_data,
                                        nullptr, 0);
        
        SelectObject(info->memory_dc, info->bitmap);
        
        info->width = width;
        info->height = height;
        
        // Resize window if AUTOSIZE
        if (info->window_flags & 1) {
            RECT rect = { 0, 0, width, height + info->trackbar_count * 30 };
            AdjustWindowRect(&rect, GetWindowLong(info->hwnd, GWL_STYLE), FALSE);
            SetWindowPos(info->hwnd, nullptr, 0, 0,
                        rect.right - rect.left, rect.bottom - rect.top,
                        SWP_NOMOVE | SWP_NOZORDER);
        }
    }
    
    info->channels = channels;
    
    // Convert image to BGRA
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_idx = (y * width + x) * channels;
            int dst_idx = (y * width + x) * 4;
            
            if (channels == 1) {
                // Grayscale to BGRA
                info->image_data[dst_idx + 0] = data[src_idx];
                info->image_data[dst_idx + 1] = data[src_idx];
                info->image_data[dst_idx + 2] = data[src_idx];
                info->image_data[dst_idx + 3] = 255;
            } else if (channels == 3) {
                // RGB to BGRA
                info->image_data[dst_idx + 0] = data[src_idx + 2];  // B
                info->image_data[dst_idx + 1] = data[src_idx + 1];  // G
                info->image_data[dst_idx + 2] = data[src_idx + 0];  // R
                info->image_data[dst_idx + 3] = 255;
            } else if (channels == 4) {
                // RGBA to BGRA
                info->image_data[dst_idx + 0] = data[src_idx + 2];  // B
                info->image_data[dst_idx + 1] = data[src_idx + 1];  // G
                info->image_data[dst_idx + 2] = data[src_idx + 0];  // R
                info->image_data[dst_idx + 3] = data[src_idx + 3];  // A
            }
        }
    }
    
    // Trigger repaint
    InvalidateRect(info->hwnd, nullptr, FALSE);
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_initialized) return -1;
    
    g_last_key = -1;
    
    MSG msg;
    
    if (delay <= 0) {
        // Block until key press
        while (g_last_key == -1) {
            if (GetMessage(&msg, nullptr, 0, 0) > 0) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                break;
            }
        }
    } else {
        // Wait for specified time or key press
        DWORD end_time = GetTickCount() + delay;
        
        while (g_last_key == -1 && GetTickCount() < end_time) {
            if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                Sleep(1);
            }
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
        
        if (info->bitmap) {
            DeleteObject(info->bitmap);
        }
        if (info->memory_dc) {
            DeleteDC(info->memory_dc);
        }
        if (info->hdc) {
            ReleaseDC(info->hwnd, info->hdc);
        }
        
        DestroyWindow(info->hwnd);
        delete info;
        g_windows.erase(it);
    }
}

// Destroy all windows
void destroyAllWindows() {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    for (auto& pair : g_windows) {
        WindowInfo* info = pair.second;
        
        if (info->bitmap) {
            DeleteObject(info->bitmap);
        }
        if (info->memory_dc) {
            DeleteDC(info->memory_dc);
        }
        if (info->hdc) {
            ReleaseDC(info->hwnd, info->hdc);
        }
        
        DestroyWindow(info->hwnd);
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
    
    // Calculate position
    RECT rect;
    GetClientRect(winfo->hwnd, &rect);
    int y = rect.bottom + winfo->trackbar_count * 30;
    
    std::wstring wname = utf8ToWide(trackbar_name);
    
    // Create label
    HWND label = CreateWindowExW(
        0, L"STATIC", wname.c_str(),
        WS_CHILD | WS_VISIBLE,
        5, y, 100, 25,
        winfo->hwnd, nullptr, g_instance, nullptr);
    
    // Create trackbar
    HWND trackbar = CreateWindowExW(
        0, TRACKBAR_CLASS, L"",
        WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS,
        110, y, rect.right - 170, 25,
        winfo->hwnd, nullptr, g_instance, nullptr);
    
    SendMessage(trackbar, TBM_SETRANGE, TRUE, MAKELONG(0, max_value));
    SendMessage(trackbar, TBM_SETPOS, TRUE, *value);
    
    // Create value label
    wchar_t buf[32];
    swprintf(buf, 32, L"%d", *value);
    HWND value_label = CreateWindowExW(
        0, L"STATIC", buf,
        WS_CHILD | WS_VISIBLE | SS_RIGHT,
        rect.right - 55, y, 50, 25,
        winfo->hwnd, nullptr, g_instance, nullptr);
    
    // Store trackbar info
    TrackbarInfo& tinfo = winfo->trackbars[trackbar_name];
    tinfo.trackbar = trackbar;
    tinfo.label = label;
    tinfo.value_label = value_label;
    tinfo.value_ptr = value;
    tinfo.max_value = max_value;
    tinfo.callback = callback;
    
    winfo->trackbar_count++;
    
    // Resize window to accommodate trackbar
    rect.bottom += 30;
    AdjustWindowRect(&rect, GetWindowLong(winfo->hwnd, GWL_STYLE), FALSE);
    SetWindowPos(winfo->hwnd, nullptr, 0, 0,
                rect.right - rect.left, rect.bottom - rect.top,
                SWP_NOMOVE | SWP_NOZORDER);
}

// Get trackbar position
int getTrackbarPos(const std::string& trackbar_name,
                   const std::string& window_name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it == g_windows.end()) return 0;
    
    auto tit = it->second->trackbars.find(trackbar_name);
    if (tit == it->second->trackbars.end()) return 0;
    
    return (int)SendMessage(tit->second.trackbar, TBM_GETPOS, 0, 0);
}

// Set trackbar position
void setTrackbarPos(const std::string& trackbar_name,
                    const std::string& window_name,
                    int pos) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(window_name);
    if (it == g_windows.end()) return;
    
    auto tit = it->second->trackbars.find(trackbar_name);
    if (tit == it->second->trackbars.end()) return;
    
    SendMessage(tit->second.trackbar, TBM_SETPOS, TRUE, pos);
    
    // Update value label
    wchar_t buf[32];
    swprintf(buf, 32, L"%d", pos);
    SetWindowTextW(tit->second.value_label, buf);
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
        SetWindowPos(it->second->hwnd, nullptr, x, y, 0, 0,
                    SWP_NOSIZE | SWP_NOZORDER);
    }
}

// Resize window
void resizeWindow(const std::string& name, int width, int height) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        RECT rect = { 0, 0, width, height };
        AdjustWindowRect(&rect, GetWindowLong(it->second->hwnd, GWL_STYLE), FALSE);
        SetWindowPos(it->second->hwnd, nullptr, 0, 0,
                    rect.right - rect.left, rect.bottom - rect.top,
                    SWP_NOMOVE | SWP_NOZORDER);
    }
}

// Set window title
void setWindowTitle(const std::string& name, const std::string& title) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        std::wstring wtitle = utf8ToWide(title);
        SetWindowTextW(it->second->hwnd, wtitle.c_str());
    }
}

// Get window handle
HWND getWindowHandle(const std::string& name) {
    std::lock_guard<std::mutex> lock(g_mutex);
    
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        return it->second->hwnd;
    }
    return nullptr;
}

// Open file dialog
std::string openFileDialog(const std::string& title,
                           const std::string& filter = "") {
    wchar_t filename[MAX_PATH] = {0};
    
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = utf8ToWide(title).c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
    
    if (!filter.empty()) {
        std::wstring wfilter = utf8ToWide(filter);
        ofn.lpstrFilter = wfilter.c_str();
    }
    
    if (GetOpenFileNameW(&ofn)) {
        char result[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, result, MAX_PATH, nullptr, nullptr);
        return std::string(result);
    }
    
    return "";
}

// Save file dialog
std::string saveFileDialog(const std::string& title,
                           const std::string& filter = "") {
    wchar_t filename[MAX_PATH] = {0};
    
    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = nullptr;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = utf8ToWide(title).c_str();
    ofn.Flags = OFN_OVERWRITEPROMPT;
    
    if (!filter.empty()) {
        std::wstring wfilter = utf8ToWide(filter);
        ofn.lpstrFilter = wfilter.c_str();
    }
    
    if (GetSaveFileNameW(&ofn)) {
        char result[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, filename, -1, result, MAX_PATH, nullptr, nullptr);
        return std::string(result);
    }
    
    return "";
}

// Message box
int showMessageBox(const std::string& title, const std::string& message, int flags = 0) {
    std::wstring wtitle = utf8ToWide(title);
    std::wstring wmessage = utf8ToWide(message);
    
    return MessageBoxW(nullptr, wmessage.c_str(), wtitle.c_str(), flags);
}

// Clipboard operations
bool copyToClipboard(const unsigned char* data, int width, int height, int channels) {
    if (!OpenClipboard(nullptr)) return false;
    
    EmptyClipboard();
    
    // Create DIB
    int row_bytes = ((width * 3 + 3) / 4) * 4;  // DWORD-aligned
    int data_size = sizeof(BITMAPINFOHEADER) + row_bytes * height;
    
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, data_size);
    if (!hMem) {
        CloseClipboard();
        return false;
    }
    
    unsigned char* mem = (unsigned char*)GlobalLock(hMem);
    
    BITMAPINFOHEADER* header = (BITMAPINFOHEADER*)mem;
    header->biSize = sizeof(BITMAPINFOHEADER);
    header->biWidth = width;
    header->biHeight = height;  // Bottom-up
    header->biPlanes = 1;
    header->biBitCount = 24;
    header->biCompression = BI_RGB;
    header->biSizeImage = row_bytes * height;
    header->biXPelsPerMeter = 0;
    header->biYPelsPerMeter = 0;
    header->biClrUsed = 0;
    header->biClrImportant = 0;
    
    unsigned char* pixels = mem + sizeof(BITMAPINFOHEADER);
    
    // Convert and copy pixels (bottom-up, BGR)
    for (int y = 0; y < height; y++) {
        int src_y = height - 1 - y;
        for (int x = 0; x < width; x++) {
            int src_idx = (src_y * width + x) * channels;
            int dst_idx = y * row_bytes + x * 3;
            
            if (channels == 1) {
                pixels[dst_idx + 0] = data[src_idx];
                pixels[dst_idx + 1] = data[src_idx];
                pixels[dst_idx + 2] = data[src_idx];
            } else {
                pixels[dst_idx + 0] = data[src_idx + 2];  // B
                pixels[dst_idx + 1] = data[src_idx + 1];  // G
                pixels[dst_idx + 2] = data[src_idx + 0];  // R
            }
        }
    }
    
    GlobalUnlock(hMem);
    SetClipboardData(CF_DIB, hMem);
    CloseClipboard();
    
    return true;
}

} // namespace win32
} // namespace gui
} // namespace neurova

#endif // _WIN32
