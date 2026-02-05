// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file data.hpp
 * @brief Main include for Neurova Data module
 * 
 * This header provides data loading and dataset utilities:
 * - Dataset base classes (Dataset, TensorDataset, ImageDataset)
 * - Samplers (Sequential, Random, Weighted, Distributed)
 * - DataLoader for batch iteration
 * - Utility functions for data paths
 * 
 * @example Basic DataLoader Usage
 * @code
 * #include <neurova/data/data.hpp>
 * 
 * using namespace neurova::data;
 * 
 * // Create dataset
 * auto dataset = std::make_shared<TensorDataset<float>>(features, labels);
 * 
 * // Create DataLoader with batching and shuffling
 * DataLoader<float> train_loader(dataset, 32, true, false);
 * 
 * // Iterate over batches
 * for (auto& batch : train_loader) {
 *     auto& X = batch.features;  // vector<vector<float>>
 *     auto& y = batch.labels;    // vector<int>
 *     // ... train step
 * }
 * @endcode
 * 
 * @example Train/Test Split
 * @code
 * auto dataset = std::make_shared<TensorDataset<float>>(X, y);
 * auto splits = random_split_frac(dataset, {0.8f, 0.2f}, 42);
 * auto train_set = splits[0];
 * auto test_set = splits[1];
 * @endcode
 */

#ifndef NEUROVA_DATA_HPP
#define NEUROVA_DATA_HPP

#include "dataset.hpp"
#include "sampler.hpp"
#include "dataloader.hpp"

#include <string>
#include <filesystem>

namespace neurova {
namespace data {

/**
 * @brief Version information for the data module
 */
constexpr const char* DATA_VERSION = "1.0.0";

/**
 * @brief Get path to cascade classifiers directory
 */
inline std::string get_cascade_path(const std::string& type = "haarcascades") {
    // Try common locations
    std::vector<std::string> candidates = {
        "data/" + type,
        "../data/" + type,
        "../../data/" + type,
    };
    
    for (const auto& path : candidates) {
        if (std::filesystem::exists(path)) {
            return path + "/";
        }
    }
    
    return "";
}

/**
 * @brief Get path to Haar cascade file
 * @param name Cascade name (e.g., "frontalface_default")
 * @return Full path to cascade XML file
 */
inline std::string get_haarcascade(const std::string& name) {
    std::string filename = name;
    
    // Add prefix if missing
    if (filename.find("haarcascade_") != 0) {
        filename = "haarcascade_" + filename;
    }
    
    // Add extension if missing
    if (filename.find(".xml") == std::string::npos) {
        filename += ".xml";
    }
    
    std::string base_path = get_cascade_path("haarcascades");
    return base_path + filename;
}

/**
 * @brief Get path to LBP cascade file
 */
inline std::string get_lbpcascade(const std::string& name) {
    std::string filename = name;
    
    if (filename.find("lbpcascade_") != 0) {
        filename = "lbpcascade_" + filename;
    }
    
    if (filename.find(".xml") == std::string::npos) {
        filename += ".xml";
    }
    
    std::string base_path = get_cascade_path("lbpcascades");
    return base_path + filename;
}

/**
 * @brief Get path to HOG cascade file
 */
inline std::string get_hogcascade(const std::string& name = "pedestrians") {
    std::string filename = name;
    
    if (filename.find("hogcascade_") != 0) {
        filename = "hogcascade_" + filename;
    }
    
    if (filename.find(".xml") == std::string::npos) {
        filename += ".xml";
    }
    
    std::string base_path = get_cascade_path("hogcascades");
    return base_path + filename;
}

/**
 * @brief CSV file reader for tabular data
 */
class CSVReader {
public:
    /**
     * @brief Read CSV file
     * @param filepath Path to CSV file
     * @param has_header Whether first row is header
     * @param delimiter Field delimiter
     * @return Tuple of (headers, data)
     */
    static std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>>
    read(const std::string& filepath, bool has_header = true, char delimiter = ',') {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filepath);
        }
        
        std::vector<std::string> headers;
        std::vector<std::vector<std::string>> data;
        std::string line;
        bool first_line = true;
        
        while (std::getline(file, line)) {
            std::vector<std::string> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, delimiter)) {
                // Trim whitespace
                cell.erase(0, cell.find_first_not_of(" \t\r\n"));
                cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
                row.push_back(cell);
            }
            
            if (first_line && has_header) {
                headers = std::move(row);
                first_line = false;
            } else {
                data.push_back(std::move(row));
            }
        }
        
        return {headers, data};
    }
    
    /**
     * @brief Read CSV as numeric data
     * @return Tuple of (headers, features, labels)
     */
    static std::tuple<std::vector<std::string>, 
                      std::vector<std::vector<float>>,
                      std::vector<float>>
    read_numeric(const std::string& filepath, 
                 int label_col = -1,
                 bool has_header = true) {
        auto [headers, str_data] = read(filepath, has_header);
        
        if (str_data.empty()) {
            return {headers, {}, {}};
        }
        
        int num_cols = static_cast<int>(str_data[0].size());
        if (label_col < 0) label_col = num_cols + label_col;
        
        std::vector<std::vector<float>> features;
        std::vector<float> labels;
        
        for (const auto& row : str_data) {
            std::vector<float> feat;
            for (int i = 0; i < num_cols; ++i) {
                float val = 0.0f;
                try {
                    val = std::stof(row[i]);
                } catch (...) {
                    // Keep as 0 if not numeric
                }
                
                if (i == label_col) {
                    labels.push_back(val);
                } else {
                    feat.push_back(val);
                }
            }
            features.push_back(std::move(feat));
        }
        
        return {headers, features, labels};
    }
};

/**
 * @brief Load tabular data from CSV into TensorDataset
 */
inline std::shared_ptr<TensorDataset<float>> load_csv(
    const std::string& filepath,
    int label_col = -1,
    bool has_header = true
) {
    auto [headers, features, labels] = CSVReader::read_numeric(filepath, label_col, has_header);
    return std::make_shared<TensorDataset<float>>(features, labels);
}

/**
 * @brief Image folder dataset for loading images from directory structure
 * 
 * Expected structure:
 *   root/class1/image1.jpg
 *   root/class1/image2.jpg
 *   root/class2/image1.jpg
 *   ...
 */
template<typename T = float>
class ImageFolderDataset : public Dataset<std::tuple<std::vector<T>, int>> {
public:
    using SampleType = std::tuple<std::vector<T>, int>;
    
    /**
     * @brief Construct from root directory
     * @param root Root directory path
     * @param extensions Allowed file extensions
     */
    ImageFolderDataset(const std::string& root,
                       const std::vector<std::string>& extensions = {".jpg", ".jpeg", ".png", ".bmp"})
        : root_(root), extensions_(extensions) {
        scan_directory();
    }
    
    SampleType get(size_t index) const override {
        if (index >= samples_.size()) {
            throw std::out_of_range("Index out of range");
        }
        
        const auto& [path, label] = samples_[index];
        
        // Load image (placeholder - actual implementation would use image loading library)
        std::vector<T> image_data;
        // ... load image from path ...
        
        return std::make_tuple(image_data, label);
    }
    
    size_t size() const override {
        return samples_.size();
    }
    
    const std::vector<std::string>& classes() const {
        return classes_;
    }
    
    int class_to_idx(const std::string& class_name) const {
        auto it = class_to_idx_.find(class_name);
        if (it != class_to_idx_.end()) {
            return it->second;
        }
        return -1;
    }
    
private:
    std::string root_;
    std::vector<std::string> extensions_;
    std::vector<std::string> classes_;
    std::map<std::string, int> class_to_idx_;
    std::vector<std::pair<std::string, int>> samples_;
    
    void scan_directory() {
        namespace fs = std::filesystem;
        
        if (!fs::exists(root_)) {
            throw std::runtime_error("Directory does not exist: " + root_);
        }
        
        // Find all class directories
        for (const auto& entry : fs::directory_iterator(root_)) {
            if (entry.is_directory()) {
                classes_.push_back(entry.path().filename().string());
            }
        }
        
        std::sort(classes_.begin(), classes_.end());
        
        // Build class to index map
        for (size_t i = 0; i < classes_.size(); ++i) {
            class_to_idx_[classes_[i]] = static_cast<int>(i);
        }
        
        // Find all images
        for (const auto& class_name : classes_) {
            int label = class_to_idx_[class_name];
            fs::path class_path = fs::path(root_) / class_name;
            
            for (const auto& entry : fs::directory_iterator(class_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    
                    if (std::find(extensions_.begin(), extensions_.end(), ext) != extensions_.end()) {
                        samples_.emplace_back(entry.path().string(), label);
                    }
                }
            }
        }
    }
};

} // namespace data
} // namespace neurova

// Additional includes needed by CSVReader
#include <fstream>
#include <sstream>
#include <map>

#endif // NEUROVA_DATA_HPP
