// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file face_dataset.hpp
 * @brief Face dataset management for training and evaluation
 */

#ifndef NEUROVA_FACE_DATASET_HPP
#define NEUROVA_FACE_DATASET_HPP

#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

namespace neurova {
namespace face {

namespace fs = std::filesystem;

/**
 * @brief Sample structure containing face data
 */
struct FaceSample {
    std::vector<float> image;
    int width, height, channels;
    int label;
    std::string filename;
};

/**
 * @brief Face dataset for training and evaluation
 * 
 * Organizes face images into train/test/validation splits.
 * 
 * Directory structure:
 *   root/
 *     train/
 *       person1/
 *         img1.jpg, img2.jpg
 *       person2/
 *         img1.jpg
 *     test/
 *       person1/
 *         img1.jpg
 */
class FaceDataset {
public:
    /**
     * @brief Construct dataset from directory
     */
    FaceDataset(
        const std::string& root_dir = "",
        int image_width = 100,
        int image_height = 100,
        bool grayscale = true
    ) : image_width_(image_width),
        image_height_(image_height),
        grayscale_(grayscale) {
        
        if (!root_dir.empty()) {
            root_dir_ = root_dir;
            train_dir_ = root_dir + "/train";
            test_dir_ = root_dir + "/test";
            val_dir_ = root_dir + "/validation";
            buildClassMapping();
        }
    }

    /**
     * @brief Set train directory
     */
    void setTrainDir(const std::string& dir) {
        train_dir_ = dir;
        buildClassMapping();
    }

    /**
     * @brief Set test directory
     */
    void setTestDir(const std::string& dir) {
        test_dir_ = dir;
    }

    /**
     * @brief Set validation directory
     */
    void setValDir(const std::string& dir) {
        val_dir_ = dir;
    }

    /**
     * @brief Get class names
     */
    const std::vector<std::string>& classes() const { return classes_; }

    /**
     * @brief Get number of classes
     */
    size_t numClasses() const { return classes_.size(); }

    /**
     * @brief Get class index by name
     */
    int classIndex(const std::string& name) const {
        auto it = class_to_idx_.find(name);
        return (it != class_to_idx_.end()) ? it->second : -1;
    }

    /**
     * @brief Get class name by index
     */
    const std::string& className(int idx) const {
        static const std::string empty;
        return (idx >= 0 && idx < static_cast<int>(classes_.size())) ? 
               classes_[idx] : empty;
    }

    /**
     * @brief Load training samples
     */
    std::vector<FaceSample> loadTrain() {
        return loadFromDir(train_dir_);
    }

    /**
     * @brief Load test samples
     */
    std::vector<FaceSample> loadTest() {
        return loadFromDir(test_dir_);
    }

    /**
     * @brief Load validation samples
     */
    std::vector<FaceSample> loadValidation() {
        return loadFromDir(val_dir_);
    }

    /**
     * @brief Get training images and labels as flat arrays
     */
    std::pair<std::vector<std::vector<float>>, std::vector<int>> getTrainData() {
        auto samples = loadTrain();
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        
        for (const auto& s : samples) {
            images.push_back(s.image);
            labels.push_back(s.label);
        }
        
        return {images, labels};
    }

    /**
     * @brief Get test images and labels as flat arrays
     */
    std::pair<std::vector<std::vector<float>>, std::vector<int>> getTestData() {
        auto samples = loadTest();
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        
        for (const auto& s : samples) {
            images.push_back(s.image);
            labels.push_back(s.label);
        }
        
        return {images, labels};
    }

    /**
     * @brief Split data into train/test sets
     */
    void splitData(
        const std::string& input_dir,
        const std::string& output_dir,
        float train_ratio = 0.8f,
        bool shuffle = true
    ) {
        // Create output directories
        std::string train_out = output_dir + "/train";
        std::string test_out = output_dir + "/test";
        
        createDirectory(train_out);
        createDirectory(test_out);
        
        // Process each class
        for (const auto& entry : fs::directory_iterator(input_dir)) {
            if (!entry.is_directory()) continue;
            
            std::string class_name = entry.path().filename().string();
            if (class_name[0] == '.') continue;
            
            std::string train_class = train_out + "/" + class_name;
            std::string test_class = test_out + "/" + class_name;
            
            createDirectory(train_class);
            createDirectory(test_class);
            
            // Collect images
            std::vector<std::string> images;
            for (const auto& img : fs::directory_iterator(entry.path())) {
                std::string ext = img.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    images.push_back(img.path().string());
                }
            }
            
            // Shuffle if requested
            if (shuffle) {
                // Simple shuffle using modulo
                for (size_t i = images.size() - 1; i > 0; --i) {
                    size_t j = (i * 31337 + 12345) % (i + 1);
                    std::swap(images[i], images[j]);
                }
            }
            
            // Split
            size_t train_count = static_cast<size_t>(images.size() * train_ratio);
            
            for (size_t i = 0; i < images.size(); ++i) {
                std::string dest = (i < train_count) ? train_class : test_class;
                std::string filename = fs::path(images[i]).filename().string();
                fs::copy_file(images[i], dest + "/" + filename, 
                             fs::copy_options::skip_existing);
            }
        }
    }

private:
    std::string root_dir_;
    std::string train_dir_;
    std::string test_dir_;
    std::string val_dir_;
    int image_width_;
    int image_height_;
    bool grayscale_;
    std::vector<std::string> classes_;
    std::map<std::string, int> class_to_idx_;

    void buildClassMapping() {
        classes_.clear();
        class_to_idx_.clear();
        
        if (train_dir_.empty() || !fs::exists(train_dir_)) return;
        
        for (const auto& entry : fs::directory_iterator(train_dir_)) {
            if (entry.is_directory()) {
                std::string name = entry.path().filename().string();
                if (name[0] != '.') {
                    classes_.push_back(name);
                }
            }
        }
        
        std::sort(classes_.begin(), classes_.end());
        
        for (size_t i = 0; i < classes_.size(); ++i) {
            class_to_idx_[classes_[i]] = static_cast<int>(i);
        }
    }

    std::vector<FaceSample> loadFromDir(const std::string& dir) {
        std::vector<FaceSample> samples;
        
        if (dir.empty() || !fs::exists(dir)) return samples;
        
        for (const auto& class_entry : fs::directory_iterator(dir)) {
            if (!class_entry.is_directory()) continue;
            
            std::string class_name = class_entry.path().filename().string();
            if (class_name[0] == '.') continue;
            
            // Add class if new
            if (class_to_idx_.find(class_name) == class_to_idx_.end()) {
                int idx = static_cast<int>(classes_.size());
                classes_.push_back(class_name);
                class_to_idx_[class_name] = idx;
            }
            
            int label = class_to_idx_[class_name];
            
            for (const auto& img_entry : fs::directory_iterator(class_entry.path())) {
                std::string ext = img_entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    FaceSample sample;
                    sample.filename = img_entry.path().string();
                    sample.label = label;
                    sample.width = image_width_;
                    sample.height = image_height_;
                    sample.channels = grayscale_ ? 1 : 3;
                    
                    // Note: Actual image loading would require external library
                    // This provides the structure for integration
                    sample.image.resize(image_width_ * image_height_ * sample.channels);
                    
                    samples.push_back(sample);
                }
            }
        }
        
        return samples;
    }

    void createDirectory(const std::string& path) {
        fs::create_directories(path);
    }
};

// ============================================================================
// Data Iterator
// ============================================================================

/**
 * @brief Iterator for batching face samples
 */
class FaceDataIterator {
public:
    FaceDataIterator(
        std::vector<FaceSample>& samples,
        size_t batch_size = 32,
        bool shuffle = true
    ) : samples_(samples),
        batch_size_(batch_size),
        current_(0) {
        
        indices_.resize(samples.size());
        for (size_t i = 0; i < indices_.size(); ++i) {
            indices_[i] = i;
        }
        
        if (shuffle) {
            shuffleIndices();
        }
    }

    /**
     * @brief Reset iterator
     */
    void reset(bool shuffle = true) {
        current_ = 0;
        if (shuffle) {
            shuffleIndices();
        }
    }

    /**
     * @brief Check if more batches available
     */
    bool hasNext() const {
        return current_ < indices_.size();
    }

    /**
     * @brief Get next batch
     */
    std::vector<FaceSample> next() {
        std::vector<FaceSample> batch;
        
        size_t end = std::min(current_ + batch_size_, indices_.size());
        for (size_t i = current_; i < end; ++i) {
            batch.push_back(samples_[indices_[i]]);
        }
        
        current_ = end;
        return batch;
    }

    /**
     * @brief Get number of batches
     */
    size_t numBatches() const {
        return (indices_.size() + batch_size_ - 1) / batch_size_;
    }

private:
    std::vector<FaceSample>& samples_;
    std::vector<size_t> indices_;
    size_t batch_size_;
    size_t current_;

    void shuffleIndices() {
        for (size_t i = indices_.size() - 1; i > 0; --i) {
            size_t j = (i * 31337 + 12345) % (i + 1);
            std::swap(indices_[i], indices_[j]);
        }
    }
};

} // namespace face
} // namespace neurova

#endif // NEUROVA_FACE_DATASET_HPP
