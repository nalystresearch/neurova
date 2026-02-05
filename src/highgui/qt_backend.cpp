/* Copyright (c) 2026 @squid consultancy group (scg)
 * all rights reserved.
 * licensed under the apache license 2.0.
 */

/**
 * Neurova Qt GUI Backend
 * Cross-platform GUI using Qt5/Qt6
 */

#include <vector>
#include <string>
#include <functional>
#include <unordered_map>

#ifdef NEUROVA_HAVE_QT

#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QSlider>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QTimer>
#include <QMessageBox>
#include <QFileDialog>

namespace neurova {
namespace gui {
namespace qt {

// Forward declarations
class ImageWindow;

// Global window registry
static std::unordered_map<std::string, ImageWindow*> g_windows;
static QApplication* g_app = nullptr;
static bool g_initialized = false;

// Initialize Qt application
bool init(int argc = 0, char** argv = nullptr) {
    if (g_initialized) return true;
    
    static int dummy_argc = 1;
    static char* dummy_argv[] = {(char*)"neurova", nullptr};
    
    if (argc == 0) {
        argc = dummy_argc;
        argv = dummy_argv;
    }
    
    g_app = new QApplication(argc, argv);
    g_initialized = true;
    
    return true;
}

// Cleanup
void cleanup() {
    for (auto& pair : g_windows) {
        delete pair.second;
    }
    g_windows.clear();
    
    if (g_app) {
        delete g_app;
        g_app = nullptr;
    }
    g_initialized = false;
}

// Mouse callback type
using MouseCallback = std::function<void(int event, int x, int y, int flags)>;

// Trackbar callback type
using TrackbarCallback = std::function<void(int value)>;

// Image display window
class ImageWindow : public QMainWindow {
    Q_OBJECT

public:
    ImageWindow(const std::string& name, int flags = 0)
        : name_(name), flags_(flags) {
        
        setWindowTitle(QString::fromStdString(name));
        
        central_widget_ = new QWidget(this);
        setCentralWidget(central_widget_);
        
        layout_ = new QVBoxLayout(central_widget_);
        
        image_label_ = new QLabel(this);
        image_label_->setAlignment(Qt::AlignCenter);
        image_label_->setMouseTracking(true);
        layout_->addWidget(image_label_);
        
        trackbar_layout_ = new QVBoxLayout();
        layout_->addLayout(trackbar_layout_);
        
        setMouseTracking(true);
    }
    
    void showImage(const unsigned char* data, int width, int height, int channels) {
        QImage::Format format;
        switch (channels) {
            case 1:
                format = QImage::Format_Grayscale8;
                break;
            case 3:
                format = QImage::Format_RGB888;
                break;
            case 4:
                format = QImage::Format_RGBA8888;
                break;
            default:
                return;
        }
        
        QImage image(data, width, height, width * channels, format);
        image_label_->setPixmap(QPixmap::fromImage(image));
        
        if (flags_ & 1) {  // WINDOW_AUTOSIZE
            resize(width, height + trackbar_layout_->count() * 40);
        }
    }
    
    void addTrackbar(const std::string& name, int* value, int max_value,
                     TrackbarCallback callback) {
        QWidget* trackbar_widget = new QWidget(this);
        QHBoxLayout* hbox = new QHBoxLayout(trackbar_widget);
        
        QLabel* label = new QLabel(QString::fromStdString(name), this);
        QSlider* slider = new QSlider(Qt::Horizontal, this);
        QLabel* value_label = new QLabel(QString::number(*value), this);
        
        slider->setRange(0, max_value);
        slider->setValue(*value);
        
        connect(slider, &QSlider::valueChanged, [=](int val) {
            *value = val;
            value_label->setText(QString::number(val));
            if (callback) {
                callback(val);
            }
        });
        
        hbox->addWidget(label);
        hbox->addWidget(slider);
        hbox->addWidget(value_label);
        
        trackbar_layout_->addWidget(trackbar_widget);
        trackbars_[name] = slider;
    }
    
    void setTrackbarValue(const std::string& name, int value) {
        auto it = trackbars_.find(name);
        if (it != trackbars_.end()) {
            it->second->setValue(value);
        }
    }
    
    int getTrackbarValue(const std::string& name) {
        auto it = trackbars_.find(name);
        if (it != trackbars_.end()) {
            return it->second->value();
        }
        return 0;
    }
    
    void setMouseCallback(MouseCallback callback) {
        mouse_callback_ = callback;
    }
    
    std::string getName() const { return name_; }

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (mouse_callback_) {
            int flags = 0;
            if (event->buttons() & Qt::LeftButton) flags |= 1;
            if (event->buttons() & Qt::RightButton) flags |= 2;
            if (event->buttons() & Qt::MiddleButton) flags |= 4;
            mouse_callback_(1, event->x(), event->y(), flags);
        }
    }
    
    void mouseReleaseEvent(QMouseEvent* event) override {
        if (mouse_callback_) {
            mouse_callback_(4, event->x(), event->y(), 0);
        }
    }
    
    void mouseMoveEvent(QMouseEvent* event) override {
        if (mouse_callback_) {
            int flags = 0;
            if (event->buttons() & Qt::LeftButton) flags |= 1;
            if (event->buttons() & Qt::RightButton) flags |= 2;
            if (event->buttons() & Qt::MiddleButton) flags |= 4;
            mouse_callback_(0, event->x(), event->y(), flags);
        }
    }
    
    void keyPressEvent(QKeyEvent* event) override {
        last_key_ = event->key();
    }

private:
    std::string name_;
    int flags_;
    QWidget* central_widget_;
    QVBoxLayout* layout_;
    QVBoxLayout* trackbar_layout_;
    QLabel* image_label_;
    std::unordered_map<std::string, QSlider*> trackbars_;
    MouseCallback mouse_callback_;
    int last_key_ = -1;
};

// Create named window
void namedWindow(const std::string& name, int flags = 0) {
    init();
    
    if (g_windows.find(name) == g_windows.end()) {
        ImageWindow* window = new ImageWindow(name, flags);
        g_windows[name] = window;
        window->show();
    }
}

// Display image in window
void imshow(const std::string& name, const unsigned char* data,
            int width, int height, int channels) {
    init();
    
    auto it = g_windows.find(name);
    if (it == g_windows.end()) {
        namedWindow(name);
        it = g_windows.find(name);
    }
    
    it->second->showImage(data, width, height, channels);
}

// Wait for key press
int waitKey(int delay = 0) {
    if (!g_initialized) return -1;
    
    if (delay <= 0) {
        g_app->exec();
        return -1;
    }
    
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(&timer, &QTimer::timeout, g_app, &QApplication::quit);
    timer.start(delay);
    g_app->exec();
    
    return -1;  // TODO: Return actual key
}

// Destroy window
void destroyWindow(const std::string& name) {
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        delete it->second;
        g_windows.erase(it);
    }
}

// Destroy all windows
void destroyAllWindows() {
    for (auto& pair : g_windows) {
        delete pair.second;
    }
    g_windows.clear();
}

// Create trackbar
void createTrackbar(const std::string& trackbar_name,
                    const std::string& window_name,
                    int* value, int max_value,
                    TrackbarCallback callback = nullptr) {
    auto it = g_windows.find(window_name);
    if (it != g_windows.end()) {
        it->second->addTrackbar(trackbar_name, value, max_value, callback);
    }
}

// Set mouse callback
void setMouseCallback(const std::string& window_name, MouseCallback callback) {
    auto it = g_windows.find(window_name);
    if (it != g_windows.end()) {
        it->second->setMouseCallback(callback);
    }
}

// Move window
void moveWindow(const std::string& name, int x, int y) {
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        it->second->move(x, y);
    }
}

// Resize window
void resizeWindow(const std::string& name, int width, int height) {
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        it->second->resize(width, height);
    }
}

// Get window property
double getWindowProperty(const std::string& name, int prop_id) {
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        switch (prop_id) {
            case 0:  // WND_PROP_FULLSCREEN
                return it->second->isFullScreen() ? 1.0 : 0.0;
            case 1:  // WND_PROP_AUTOSIZE
                return 0.0;
            case 2:  // WND_PROP_ASPECT_RATIO
                return (double)it->second->width() / it->second->height();
        }
    }
    return -1.0;
}

// Set window property
void setWindowProperty(const std::string& name, int prop_id, double value) {
    auto it = g_windows.find(name);
    if (it != g_windows.end()) {
        switch (prop_id) {
            case 0:  // WND_PROP_FULLSCREEN
                if (value > 0) {
                    it->second->showFullScreen();
                } else {
                    it->second->showNormal();
                }
                break;
        }
    }
}

// File dialog
std::string openFileDialog(const std::string& title,
                           const std::string& filter = "") {
    init();
    QString filename = QFileDialog::getOpenFileName(
        nullptr, QString::fromStdString(title), "",
        QString::fromStdString(filter));
    return filename.toStdString();
}

std::string saveFileDialog(const std::string& title,
                           const std::string& filter = "") {
    init();
    QString filename = QFileDialog::getSaveFileName(
        nullptr, QString::fromStdString(title), "",
        QString::fromStdString(filter));
    return filename.toStdString();
}

// Message box
void showMessage(const std::string& title, const std::string& message) {
    init();
    QMessageBox::information(nullptr, QString::fromStdString(title),
                            QString::fromStdString(message));
}

} // namespace qt
} // namespace gui
} // namespace neurova

#endif // NEUROVA_HAVE_QT
