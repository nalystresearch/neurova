// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file dnn.hpp
 * @brief Deep Neural Network module for model loading and inference
 * 
 * Provides DNN functionality compatible with common model formats.
 */

#ifndef NEUROVA_DNN_HPP
#define NEUROVA_DNN_HPP

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace neurova {
namespace dnn {

// ============================================================================
// Backend and Target Constants
// ============================================================================

enum Backend {
    DNN_BACKEND_DEFAULT = 0,
    DNN_BACKEND_HALIDE = 1,
    DNN_BACKEND_INFERENCE_ENGINE = 2,
    DNN_BACKEND_NEUROVA = 3,
    DNN_BACKEND_VKCOM = 4,
    DNN_BACKEND_CUDA = 5
};

enum Target {
    DNN_TARGET_CPU = 0,
    DNN_TARGET_OPENCL = 1,
    DNN_TARGET_OPENCL_FP16 = 2,
    DNN_TARGET_MYRIAD = 3,
    DNN_TARGET_VULKAN = 4,
    DNN_TARGET_FPGA = 5,
    DNN_TARGET_CUDA = 6,
    DNN_TARGET_CUDA_FP16 = 7
};

// ============================================================================
// Layer
// ============================================================================

/**
 * @brief Represents a network layer
 */
struct Layer {
    std::string name;
    std::string type;
    std::vector<std::vector<float>> blobs;
    
    Layer(const std::string& n = "", const std::string& t = "")
        : name(n), type(t) {}
    
    int outputNameToIndex(const std::string& output_name) const {
        return 0;
    }
};

// ============================================================================
// Blob Operations
// ============================================================================

/**
 * @brief 4D Blob for network input/output (NCHW format)
 */
class Blob {
public:
    Blob() : n_(0), c_(0), h_(0), w_(0) {}
    
    Blob(int n, int c, int h, int w)
        : n_(n), c_(c), h_(h), w_(w), data_(n * c * h * w, 0.0f) {}
    
    Blob(int n, int c, int h, int w, const float* data)
        : n_(n), c_(c), h_(h), w_(w), data_(data, data + n * c * h * w) {}
    
    float& at(int n, int c, int h, int w) {
        return data_[((n * c_ + c) * h_ + h) * w_ + w];
    }
    
    float at(int n, int c, int h, int w) const {
        return data_[((n * c_ + c) * h_ + h) * w_ + w];
    }
    
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    
    size_t total() const { return data_.size(); }
    int batch() const { return n_; }
    int channels() const { return c_; }
    int height() const { return h_; }
    int width() const { return w_; }
    
    std::vector<int> shape() const { return {n_, c_, h_, w_}; }
    
private:
    int n_, c_, h_, w_;
    std::vector<float> data_;
};

/**
 * @brief Create a 4D blob from image data
 * 
 * @param image Image data (HWC format, float)
 * @param height Image height
 * @param width Image width
 * @param channels Number of channels
 * @param scalefactor Scale factor for pixel values
 * @param size Output size (width, height). (0,0) means no resize
 * @param mean Mean values to subtract (per channel)
 * @param swapRB Swap R and B channels
 * @param crop Crop after resize to maintain aspect ratio
 * @return Blob in NCHW format
 */
inline Blob blobFromImage(
    const float* image, int height, int width, int channels,
    float scalefactor = 1.0f,
    std::pair<int, int> size = {0, 0},
    const float mean[3] = nullptr,
    bool swapRB = false,
    bool crop = false
) {
    int target_w = size.first > 0 ? size.first : width;
    int target_h = size.second > 0 ? size.second : height;
    
    Blob blob(1, channels, target_h, target_w);
    
    // Simple nearest-neighbor resize + reorder to CHW
    for (int c = 0; c < channels; ++c) {
        int src_c = c;
        if (swapRB && channels >= 3) {
            if (c == 0) src_c = 2;
            else if (c == 2) src_c = 0;
        }
        
        float channel_mean = (mean && c < 3) ? mean[c] : 0.0f;
        
        for (int h = 0; h < target_h; ++h) {
            int src_h = h * height / target_h;
            for (int w = 0; w < target_w; ++w) {
                int src_w = w * width / target_w;
                float val = image[(src_h * width + src_w) * channels + src_c];
                blob.at(0, c, h, w) = (val - channel_mean) * scalefactor;
            }
        }
    }
    
    return blob;
}

/**
 * @brief Create blob from multiple images
 */
inline Blob blobFromImages(
    const std::vector<const float*>& images,
    const std::vector<std::tuple<int, int, int>>& shapes, // (h, w, c) for each
    float scalefactor = 1.0f,
    std::pair<int, int> size = {0, 0},
    const float mean[3] = nullptr,
    bool swapRB = false
) {
    if (images.empty()) return Blob();
    
    auto [h0, w0, c0] = shapes[0];
    int target_w = size.first > 0 ? size.first : w0;
    int target_h = size.second > 0 ? size.second : h0;
    
    Blob blob(static_cast<int>(images.size()), c0, target_h, target_w);
    
    for (size_t n = 0; n < images.size(); ++n) {
        auto [h, w, c] = shapes[n];
        for (int ch = 0; ch < c; ++ch) {
            int src_c = ch;
            if (swapRB && c >= 3) {
                if (ch == 0) src_c = 2;
                else if (ch == 2) src_c = 0;
            }
            
            float channel_mean = (mean && ch < 3) ? mean[ch] : 0.0f;
            
            for (int y = 0; y < target_h; ++y) {
                int src_y = y * h / target_h;
                for (int x = 0; x < target_w; ++x) {
                    int src_x = x * w / target_w;
                    float val = images[n][(src_y * w + src_x) * c + src_c];
                    blob.at(static_cast<int>(n), ch, y, x) = (val - channel_mean) * scalefactor;
                }
            }
        }
    }
    
    return blob;
}

// ============================================================================
// Net (Neural Network)
// ============================================================================

/**
 * @brief Deep Neural Network class for model inference
 */
class Net {
public:
    Net() : backend_(DNN_BACKEND_DEFAULT), target_(DNN_TARGET_CPU), loaded_(false) {}
    
    /**
     * @brief Check if network is empty
     */
    bool empty() const { return !loaded_; }
    
    /**
     * @brief Set preferred backend
     */
    void setPreferableBackend(int backend_id) {
        backend_ = backend_id;
    }
    
    /**
     * @brief Set preferred target device
     */
    void setPreferableTarget(int target_id) {
        target_ = target_id;
    }
    
    /**
     * @brief Get names of all layers
     */
    std::vector<std::string> getLayerNames() const {
        return layer_names_;
    }
    
    /**
     * @brief Get indices of unconnected output layers
     */
    std::vector<int> getUnconnectedOutLayers() const {
        if (output_names_.empty()) {
            return {static_cast<int>(layer_names_.size())};
        }
        
        std::vector<int> indices;
        for (const auto& name : output_names_) {
            auto it = std::find(layer_names_.begin(), layer_names_.end(), name);
            if (it != layer_names_.end()) {
                indices.push_back(static_cast<int>(it - layer_names_.begin()) + 1);
            }
        }
        return indices.empty() ? std::vector<int>{static_cast<int>(layer_names_.size())} : indices;
    }
    
    /**
     * @brief Get names of output layers
     */
    std::vector<std::string> getUnconnectedOutLayersNames() const {
        if (!output_names_.empty()) return output_names_;
        if (!layer_names_.empty()) return {layer_names_.back()};
        return {};
    }
    
    /**
     * @brief Get layer by index or name
     */
    Layer getLayer(int index) const {
        if (index >= 0 && index < static_cast<int>(layer_names_.size())) {
            const auto& name = layer_names_[index];
            auto it = layers_.find(name);
            if (it != layers_.end()) return it->second;
        }
        return Layer("unknown", "unknown");
    }
    
    Layer getLayer(const std::string& name) const {
        auto it = layers_.find(name);
        return it != layers_.end() ? it->second : Layer(name, "unknown");
    }
    
    /**
     * @brief Set input blob
     */
    void setInput(const Blob& blob, const std::string& name = "") {
        input_blob_ = blob;
    }
    
    /**
     * @brief Run forward pass
     */
    Blob forward(const std::string& output_name = "") {
        if (!loaded_) {
            // Return dummy output
            return Blob(1, 1000, 1, 1);
        }
        
        // Placeholder: return input or dummy output
        if (input_blob_.total() > 0) {
            // For detection models
            if (framework_ == "weights") {
                return Blob(1, 100, 85, 1);
            }
            return input_blob_;
        }
        
        return Blob(1, 1000, 1, 1);
    }
    
    /**
     * @brief Estimate FLOPS
     */
    long long getFLOPS(const std::vector<int>& input_shape) const {
        long long total = 1;
        for (int s : input_shape) total *= s;
        return total * 1000;
    }
    
    /**
     * @brief Get memory consumption estimate
     */
    std::pair<size_t, size_t> getMemoryConsumption(const std::vector<int>& input_shape) const {
        size_t weights_mem = 0;
        for (const auto& [name, layer] : layers_) {
            for (const auto& blob : layer.blobs) {
                weights_mem += blob.size() * sizeof(float);
            }
        }
        
        size_t input_size = 1;
        for (int s : input_shape) input_size *= s;
        size_t blobs_mem = input_size * sizeof(float) * 2;
        
        return {weights_mem, blobs_mem};
    }
    
    // Internal: mark as loaded
    void _setLoaded(bool loaded, const std::string& framework = "") {
        loaded_ = loaded;
        framework_ = framework;
    }
    
    void _addLayer(const std::string& name, const std::string& type) {
        layer_names_.push_back(name);
        layers_[name] = Layer(name, type);
    }
    
    void _setInputNames(const std::vector<std::string>& names) {
        input_names_ = names;
    }
    
    void _setOutputNames(const std::vector<std::string>& names) {
        output_names_ = names;
    }
    
private:
    std::map<std::string, Layer> layers_;
    std::vector<std::string> layer_names_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int backend_;
    int target_;
    bool loaded_;
    std::string framework_;
    Blob input_blob_;
};

// ============================================================================
// Model Loading Functions
// ============================================================================

/**
 * @brief Load network from model file
 */
inline Net readNet(const std::string& model, const std::string& config = "", 
                   const std::string& framework = "") {
    Net net;
    
    // Detect framework from extension
    std::string ext;
    size_t dot_pos = model.rfind('.');
    if (dot_pos != std::string::npos) {
        ext = model.substr(dot_pos);
    }
    
    std::string detected_framework = framework;
    if (detected_framework.empty()) {
        if (ext == ".prototext" || ext == ".prototxt") detected_framework = "prototext";
        else if (ext == ".pb" || ext == ".pbtxt") detected_framework = "pb";
        else if (ext == ".onnx") detected_framework = "onnx";
        else if (ext == ".weights") detected_framework = "weights";
        else if (ext == ".pt" || ext == ".pth") detected_framework = "torch";
        else detected_framework = "unknown";
    }
    
    // Check if file exists
    std::ifstream file(model);
    if (file.good()) {
        net._setLoaded(true, detected_framework);
        
        // Add placeholder layers
        net._addLayer("input", "Input");
        net._addLayer("conv1", "Convolution");
        net._addLayer("relu1", "ReLU");
        net._addLayer("pool1", "Pooling");
        net._addLayer("fc1", "InnerProduct");
        net._addLayer("output", "Output");
        
        net._setInputNames({"input"});
        net._setOutputNames({"output"});
    }
    
    return net;
}

inline Net readNetFromPrototext(const std::string& prototxt, const std::string& prototextModel = "") {
    return readNet(prototextModel.empty() ? prototxt : prototextModel, prototxt, "prototext");
}

inline Net readNetFromONNX(const std::string& onnxFile) {
    return readNet(onnxFile, "", "onnx");
}

inline Net readNetFromWeights(const std::string& cfgFile, const std::string& weightsModel = "") {
    return readNet(weightsModel.empty() ? cfgFile : weightsModel, cfgFile, "weights");
}

inline Net readNetFromPB(const std::string& model, const std::string& config = "") {
    return readNet(model, config, "pb");
}

// ============================================================================
// NMS Functions
// ============================================================================

/**
 * @brief Non-Maximum Suppression on bounding boxes
 * 
 * @param bboxes Bounding boxes [x, y, width, height]
 * @param scores Confidence scores
 * @param score_threshold Minimum score to keep
 * @param nms_threshold IoU threshold for suppression
 * @param top_k Maximum boxes to keep (0 = all)
 * @return Indices of kept boxes
 */
inline std::vector<int> NMSBoxes(
    const std::vector<std::tuple<int, int, int, int>>& bboxes,
    const std::vector<float>& scores,
    float score_threshold,
    float nms_threshold,
    int top_k = 0
) {
    if (bboxes.empty() || scores.empty()) return {};
    
    // Filter by score and sort
    std::vector<std::pair<float, int>> score_indices;
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] >= score_threshold) {
            score_indices.emplace_back(scores[i], static_cast<int>(i));
        }
    }
    
    std::sort(score_indices.begin(), score_indices.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    if (top_k > 0 && static_cast<int>(score_indices.size()) > top_k) {
        score_indices.resize(top_k);
    }
    
    std::vector<int> keep;
    std::vector<bool> suppressed(bboxes.size(), false);
    
    for (const auto& [score, i] : score_indices) {
        if (suppressed[i]) continue;
        
        keep.push_back(i);
        
        auto [x1, y1, w1, h1] = bboxes[i];
        float area1 = static_cast<float>(w1 * h1);
        
        for (const auto& [s2, j] : score_indices) {
            if (suppressed[j] || i == j) continue;
            
            auto [x2, y2, w2, h2] = bboxes[j];
            
            // Compute IoU
            int xi1 = std::max(x1, x2);
            int yi1 = std::max(y1, y2);
            int xi2 = std::min(x1 + w1, x2 + w2);
            int yi2 = std::min(y1 + h1, y2 + h2);
            
            if (xi2 > xi1 && yi2 > yi1) {
                float intersection = static_cast<float>((xi2 - xi1) * (yi2 - yi1));
                float area2 = static_cast<float>(w2 * h2);
                float union_area = area1 + area2 - intersection;
                float iou = intersection / std::max(union_area, 1e-6f);
                
                if (iou > nms_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }
    
    return keep;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Get available backend-target pairs
 */
inline std::vector<std::pair<int, int>> getAvailableBackends() {
    return {
        {DNN_BACKEND_DEFAULT, DNN_TARGET_CPU},
        {DNN_BACKEND_NEUROVA, DNN_TARGET_CPU}
    };
}

/**
 * @brief Get available targets for a backend
 */
inline std::vector<int> getAvailableTargets(int backend) {
    if (backend == DNN_BACKEND_DEFAULT || backend == DNN_BACKEND_NEUROVA) {
        return {DNN_TARGET_CPU};
    }
    if (backend == DNN_BACKEND_CUDA) {
        return {DNN_TARGET_CUDA, DNN_TARGET_CUDA_FP16};
    }
    return {DNN_TARGET_CPU};
}

} // namespace dnn
} // namespace neurova

#endif // NEUROVA_DNN_HPP
