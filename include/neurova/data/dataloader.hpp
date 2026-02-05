// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file dataloader.hpp
 * @brief batch-style DataLoader for batch iteration
 */

#ifndef NEUROVA_DATA_DATALOADER_HPP
#define NEUROVA_DATA_DATALOADER_HPP

#include "dataset.hpp"
#include "sampler.hpp"

#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>

namespace neurova {
namespace data {

/**
 * @brief Batch of samples with features and labels
 */
template<typename T, typename L = int>
struct Batch {
    std::vector<std::vector<T>> features;
    std::vector<L> labels;
    size_t size() const { return features.size(); }
    bool empty() const { return features.empty(); }
};

/**
 * @brief Default collate function - stacks samples into a batch
 */
template<typename T, typename L>
Batch<T, L> default_collate(const std::vector<std::tuple<std::vector<T>, L>>& samples) {
    Batch<T, L> batch;
    batch.features.reserve(samples.size());
    batch.labels.reserve(samples.size());
    
    for (const auto& sample : samples) {
        batch.features.push_back(std::get<0>(sample));
        batch.labels.push_back(std::get<1>(sample));
    }
    
    return batch;
}

/**
 * @brief Iterator for DataLoader
 */
template<typename T, typename L = int>
class DataLoaderIterator {
public:
    using BatchType = Batch<T, L>;
    using DatasetType = Dataset<std::tuple<std::vector<T>, L>>;
    
    DataLoaderIterator() : dataset_(nullptr), batch_idx_(0) {}
    
    DataLoaderIterator(std::shared_ptr<DatasetType> dataset,
                       const std::vector<std::vector<size_t>>& batches,
                       size_t batch_idx)
        : dataset_(dataset), batches_(batches), batch_idx_(batch_idx) {}
    
    BatchType operator*() const {
        if (batch_idx_ >= batches_.size()) {
            return BatchType{};
        }
        
        const auto& indices = batches_[batch_idx_];
        std::vector<std::tuple<std::vector<T>, L>> samples;
        samples.reserve(indices.size());
        
        for (size_t idx : indices) {
            samples.push_back(dataset_->get(idx));
        }
        
        return default_collate<T, L>(samples);
    }
    
    DataLoaderIterator& operator++() {
        ++batch_idx_;
        return *this;
    }
    
    bool operator!=(const DataLoaderIterator& other) const {
        return batch_idx_ != other.batch_idx_;
    }
    
    bool operator==(const DataLoaderIterator& other) const {
        return batch_idx_ == other.batch_idx_;
    }
    
private:
    std::shared_ptr<DatasetType> dataset_;
    std::vector<std::vector<size_t>> batches_;
    size_t batch_idx_;
};

/**
 * @brief DataLoader for batched iteration over datasets
 * 
 * Provides an iterable over batches of data with optional shuffling.
 * 
 * @tparam T Feature data type
 * @tparam L Label type
 * 
 * @example
 * @code
 * auto dataset = std::make_shared<TensorDataset<float>>(features, labels);
 * DataLoader<float> loader(dataset, 32, true);
 * 
 * for (auto& batch : loader) {
 *     // batch.features: vector of feature vectors
 *     // batch.labels: vector of labels
 *     train_step(batch);
 * }
 * @endcode
 */
template<typename T, typename L = int>
class DataLoader {
public:
    using DatasetType = Dataset<std::tuple<std::vector<T>, L>>;
    using BatchType = Batch<T, L>;
    using Iterator = DataLoaderIterator<T, L>;
    
    /**
     * @brief Construct DataLoader
     * @param dataset Dataset to load from
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle data each epoch
     * @param drop_last Whether to drop incomplete last batch
     * @param num_workers Number of worker threads (0 = main thread)
     * @param seed Random seed for shuffling
     */
    DataLoader(std::shared_ptr<DatasetType> dataset,
               size_t batch_size = 1,
               bool shuffle = false,
               bool drop_last = false,
               size_t num_workers = 0,
               unsigned int seed = std::random_device{}())
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          drop_last_(drop_last),
          num_workers_(num_workers),
          seed_(seed),
          epoch_(0) {
        prepare_batches();
    }
    
    /**
     * @brief Construct with custom sampler
     */
    DataLoader(std::shared_ptr<DatasetType> dataset,
               std::shared_ptr<Sampler> sampler,
               size_t batch_size = 1,
               bool drop_last = false)
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(false),
          drop_last_(drop_last),
          num_workers_(0),
          seed_(0),
          epoch_(0),
          sampler_(sampler) {
        prepare_batches();
    }
    
    Iterator begin() {
        if (shuffle_ && !sampler_) {
            prepare_batches();
        }
        return Iterator(dataset_, batches_, 0);
    }
    
    Iterator end() {
        return Iterator(dataset_, batches_, batches_.size());
    }
    
    /**
     * @brief Get batch at specific index
     */
    BatchType get_batch(size_t batch_idx) const {
        if (batch_idx >= batches_.size()) {
            throw std::out_of_range("Batch index out of range");
        }
        
        const auto& indices = batches_[batch_idx];
        std::vector<std::tuple<std::vector<T>, L>> samples;
        samples.reserve(indices.size());
        
        for (size_t idx : indices) {
            samples.push_back(dataset_->get(idx));
        }
        
        return default_collate<T, L>(samples);
    }
    
    /**
     * @brief Get number of batches
     */
    size_t num_batches() const { return batches_.size(); }
    
    /**
     * @brief Get total number of samples
     */
    size_t size() const { return dataset_->size(); }
    
    /**
     * @brief Get batch size
     */
    size_t batch_size() const { return batch_size_; }
    
    /**
     * @brief Set epoch (affects shuffling)
     */
    void set_epoch(int epoch) {
        epoch_ = epoch;
        if (shuffle_) {
            prepare_batches();
        }
    }
    
private:
    std::shared_ptr<DatasetType> dataset_;
    std::shared_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    size_t num_workers_;
    unsigned int seed_;
    int epoch_;
    std::vector<std::vector<size_t>> batches_;
    
    void prepare_batches() {
        batches_.clear();
        
        std::vector<size_t> indices;
        
        if (sampler_) {
            indices = sampler_->indices();
        } else {
            indices.resize(dataset_->size());
            std::iota(indices.begin(), indices.end(), 0);
            
            if (shuffle_) {
                std::mt19937 gen(seed_ + epoch_);
                std::shuffle(indices.begin(), indices.end(), gen);
            }
        }
        
        // Create batches
        for (size_t i = 0; i < indices.size(); i += batch_size_) {
            size_t end = std::min(i + batch_size_, indices.size());
            
            if (drop_last_ && end - i < batch_size_) {
                break;
            }
            
            batches_.emplace_back(indices.begin() + i, indices.begin() + end);
        }
    }
};

/**
 * @brief Multi-threaded prefetching DataLoader
 * 
 * Prefetches batches in background threads for better performance.
 */
template<typename T, typename L = int>
class PrefetchDataLoader {
public:
    using DatasetType = Dataset<std::tuple<std::vector<T>, L>>;
    using BatchType = Batch<T, L>;
    
    PrefetchDataLoader(std::shared_ptr<DatasetType> dataset,
                       size_t batch_size,
                       bool shuffle = false,
                       bool drop_last = false,
                       size_t num_workers = 2,
                       size_t prefetch_factor = 2,
                       unsigned int seed = std::random_device{}())
        : dataset_(dataset),
          batch_size_(batch_size),
          shuffle_(shuffle),
          drop_last_(drop_last),
          num_workers_(std::max(size_t(1), num_workers)),
          prefetch_factor_(prefetch_factor),
          seed_(seed),
          epoch_(0),
          running_(false) {}
    
    ~PrefetchDataLoader() {
        stop();
    }
    
    /**
     * @brief Start prefetching
     */
    void start() {
        if (running_) return;
        
        prepare_batches();
        running_ = true;
        current_batch_ = 0;
        
        // Clear queue
        std::queue<std::future<BatchType>> empty;
        std::swap(prefetch_queue_, empty);
        
        // Start prefetching
        prefetch_next_batches();
    }
    
    /**
     * @brief Stop prefetching
     */
    void stop() {
        running_ = false;
    }
    
    /**
     * @brief Get next batch
     */
    BatchType next() {
        if (!running_ || current_batch_ >= batches_.size()) {
            return BatchType{};
        }
        
        // Get from prefetch queue if available
        if (!prefetch_queue_.empty()) {
            auto batch = prefetch_queue_.front().get();
            prefetch_queue_.pop();
            current_batch_++;
            prefetch_next_batches();
            return batch;
        }
        
        // Otherwise load directly
        return load_batch(current_batch_++);
    }
    
    /**
     * @brief Check if more batches available
     */
    bool has_next() const {
        return running_ && current_batch_ < batches_.size();
    }
    
    /**
     * @brief Reset to beginning
     */
    void reset() {
        stop();
        start();
    }
    
    size_t num_batches() const { return batches_.size(); }
    size_t size() const { return dataset_->size(); }
    
    void set_epoch(int epoch) {
        epoch_ = epoch;
        if (shuffle_) {
            prepare_batches();
        }
    }
    
private:
    std::shared_ptr<DatasetType> dataset_;
    size_t batch_size_;
    bool shuffle_;
    bool drop_last_;
    size_t num_workers_;
    size_t prefetch_factor_;
    unsigned int seed_;
    int epoch_;
    std::atomic<bool> running_;
    size_t current_batch_;
    std::vector<std::vector<size_t>> batches_;
    std::queue<std::future<BatchType>> prefetch_queue_;
    
    void prepare_batches() {
        batches_.clear();
        
        std::vector<size_t> indices(dataset_->size());
        std::iota(indices.begin(), indices.end(), 0);
        
        if (shuffle_) {
            std::mt19937 gen(seed_ + epoch_);
            std::shuffle(indices.begin(), indices.end(), gen);
        }
        
        for (size_t i = 0; i < indices.size(); i += batch_size_) {
            size_t end = std::min(i + batch_size_, indices.size());
            if (drop_last_ && end - i < batch_size_) break;
            batches_.emplace_back(indices.begin() + i, indices.begin() + end);
        }
    }
    
    BatchType load_batch(size_t batch_idx) {
        if (batch_idx >= batches_.size()) {
            return BatchType{};
        }
        
        const auto& indices = batches_[batch_idx];
        std::vector<std::tuple<std::vector<T>, L>> samples;
        samples.reserve(indices.size());
        
        for (size_t idx : indices) {
            samples.push_back(dataset_->get(idx));
        }
        
        return default_collate<T, L>(samples);
    }
    
    void prefetch_next_batches() {
        size_t target = std::min(current_batch_ + num_workers_ * prefetch_factor_,
                                 batches_.size());
        
        while (prefetch_queue_.size() < num_workers_ * prefetch_factor_ &&
               current_batch_ + prefetch_queue_.size() < target) {
            size_t batch_idx = current_batch_ + prefetch_queue_.size();
            prefetch_queue_.push(
                std::async(std::launch::async, [this, batch_idx]() {
                    return load_batch(batch_idx);
                })
            );
        }
    }
};

} // namespace data
} // namespace neurova

#endif // NEUROVA_DATA_DATALOADER_HPP
