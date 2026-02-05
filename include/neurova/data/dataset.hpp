// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file dataset.hpp
 * @brief Dataset base classes for data loading
 * 
 * Provides batch-style Dataset abstractions for C++.
 */

#ifndef NEUROVA_DATA_DATASET_HPP
#define NEUROVA_DATA_DATASET_HPP

#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <functional>

namespace neurova {
namespace data {

/**
 * @brief Abstract base class for map-style datasets
 * 
 * Subclasses must implement:
 *   - get(index): Returns sample at given index
 *   - size(): Returns the size of the dataset
 * 
 * @tparam T The type of data samples
 */
template<typename T>
class Dataset {
public:
    using SampleType = T;
    
    virtual ~Dataset() = default;
    
    /**
     * @brief Get sample at index
     */
    virtual T get(size_t index) const = 0;
    
    /**
     * @brief Get dataset size
     */
    virtual size_t size() const = 0;
    
    /**
     * @brief Operator[] for convenience
     */
    T operator[](size_t index) const {
        return get(index);
    }
};

/**
 * @brief Dataset wrapping arrays/vectors
 * 
 * Each sample is a tuple of values from the same index.
 * 
 * @tparam T Data type (e.g., float)
 */
template<typename T>
class TensorDataset : public Dataset<std::tuple<std::vector<T>, T>> {
public:
    using SampleType = std::tuple<std::vector<T>, T>;
    
    /**
     * @brief Construct from features and labels
     */
    TensorDataset(const std::vector<std::vector<T>>& features,
                  const std::vector<T>& labels)
        : features_(features), labels_(labels) {
        if (features_.size() != labels_.size()) {
            throw std::invalid_argument("Features and labels must have same length");
        }
    }
    
    SampleType get(size_t index) const override {
        if (index >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return std::make_tuple(features_[index], labels_[index]);
    }
    
    size_t size() const override {
        return features_.size();
    }
    
    /**
     * @brief Get all features
     */
    const std::vector<std::vector<T>>& features() const { return features_; }
    
    /**
     * @brief Get all labels
     */
    const std::vector<T>& labels() const { return labels_; }
    
private:
    std::vector<std::vector<T>> features_;
    std::vector<T> labels_;
};

/**
 * @brief Image dataset with image data and labels
 */
template<typename T = float>
class ImageDataset : public Dataset<std::tuple<std::vector<T>, int>> {
public:
    using SampleType = std::tuple<std::vector<T>, int>;
    
    /**
     * @brief Construct from images and labels
     * @param images Vector of flattened images (each is H*W*C elements)
     * @param labels Vector of integer labels
     * @param height Image height
     * @param width Image width
     * @param channels Number of channels
     */
    ImageDataset(const std::vector<std::vector<T>>& images,
                 const std::vector<int>& labels,
                 int height, int width, int channels = 1)
        : images_(images), labels_(labels),
          height_(height), width_(width), channels_(channels) {
        if (images_.size() != labels_.size()) {
            throw std::invalid_argument("Images and labels must have same length");
        }
    }
    
    SampleType get(size_t index) const override {
        if (index >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return std::make_tuple(images_[index], labels_[index]);
    }
    
    size_t size() const override {
        return images_.size();
    }
    
    int height() const { return height_; }
    int width() const { return width_; }
    int channels() const { return channels_; }
    
private:
    std::vector<std::vector<T>> images_;
    std::vector<int> labels_;
    int height_, width_, channels_;
};

/**
 * @brief Subset of a dataset at specified indices
 */
template<typename T>
class Subset : public Dataset<T> {
public:
    /**
     * @brief Construct subset from dataset and indices
     */
    Subset(std::shared_ptr<Dataset<T>> dataset, std::vector<size_t> indices)
        : dataset_(dataset), indices_(std::move(indices)) {}
    
    T get(size_t index) const override {
        if (index >= size()) {
            throw std::out_of_range("Index out of range");
        }
        return dataset_->get(indices_[index]);
    }
    
    size_t size() const override {
        return indices_.size();
    }
    
private:
    std::shared_ptr<Dataset<T>> dataset_;
    std::vector<size_t> indices_;
};

/**
 * @brief Concatenate multiple datasets
 */
template<typename T>
class ConcatDataset : public Dataset<T> {
public:
    ConcatDataset(std::vector<std::shared_ptr<Dataset<T>>> datasets)
        : datasets_(std::move(datasets)) {
        cumulative_sizes_.reserve(datasets_.size());
        size_t total = 0;
        for (const auto& ds : datasets_) {
            total += ds->size();
            cumulative_sizes_.push_back(total);
        }
    }
    
    T get(size_t index) const override {
        if (index >= size()) {
            throw std::out_of_range("Index out of range");
        }
        
        // Binary search for dataset
        auto it = std::upper_bound(cumulative_sizes_.begin(), 
                                   cumulative_sizes_.end(), index);
        size_t dataset_idx = std::distance(cumulative_sizes_.begin(), it);
        
        size_t local_idx = index;
        if (dataset_idx > 0) {
            local_idx -= cumulative_sizes_[dataset_idx - 1];
        }
        
        return datasets_[dataset_idx]->get(local_idx);
    }
    
    size_t size() const override {
        return cumulative_sizes_.empty() ? 0 : cumulative_sizes_.back();
    }
    
private:
    std::vector<std::shared_ptr<Dataset<T>>> datasets_;
    std::vector<size_t> cumulative_sizes_;
};

/**
 * @brief Randomly split a dataset into non-overlapping subsets
 * 
 * @param dataset Dataset to split
 * @param lengths Lengths of each split
 * @param seed Random seed (optional)
 * @return Vector of Subset objects
 */
template<typename T>
std::vector<std::shared_ptr<Subset<T>>> random_split(
    std::shared_ptr<Dataset<T>> dataset,
    const std::vector<size_t>& lengths,
    unsigned int seed = 42
) {
    size_t total = 0;
    for (size_t len : lengths) total += len;
    
    if (total != dataset->size()) {
        throw std::invalid_argument("Sum of lengths must equal dataset size");
    }
    
    // Generate shuffled indices
    std::vector<size_t> indices(dataset->size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);
    
    // Create subsets
    std::vector<std::shared_ptr<Subset<T>>> subsets;
    size_t offset = 0;
    
    for (size_t length : lengths) {
        std::vector<size_t> subset_indices(indices.begin() + offset,
                                           indices.begin() + offset + length);
        subsets.push_back(std::make_shared<Subset<T>>(dataset, std::move(subset_indices)));
        offset += length;
    }
    
    return subsets;
}

/**
 * @brief Split by fractions instead of absolute lengths
 */
template<typename T>
std::vector<std::shared_ptr<Subset<T>>> random_split_frac(
    std::shared_ptr<Dataset<T>> dataset,
    const std::vector<float>& fractions,
    unsigned int seed = 42
) {
    size_t total = dataset->size();
    std::vector<size_t> lengths;
    size_t assigned = 0;
    
    for (size_t i = 0; i < fractions.size() - 1; ++i) {
        size_t len = static_cast<size_t>(fractions[i] * total);
        lengths.push_back(len);
        assigned += len;
    }
    lengths.push_back(total - assigned);
    
    return random_split(dataset, lengths, seed);
}

} // namespace data
} // namespace neurova

#endif // NEUROVA_DATA_DATASET_HPP
