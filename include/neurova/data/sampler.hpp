// Copyright (c) 2026 @Squid Consultancy Group (SCG)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file sampler.hpp
 * @brief Sampler classes for controlling data iteration order
 */

#ifndef NEUROVA_DATA_SAMPLER_HPP
#define NEUROVA_DATA_SAMPLER_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace neurova {
namespace data {

/**
 * @brief Abstract base class for samplers
 * 
 * Samplers define the order in which samples are yielded.
 */
class Sampler {
public:
    virtual ~Sampler() = default;
    
    /**
     * @brief Get all indices for one epoch
     */
    virtual std::vector<size_t> indices() const = 0;
    
    /**
     * @brief Get number of samples
     */
    virtual size_t size() const = 0;
};

/**
 * @brief Sample elements sequentially in order
 */
class SequentialSampler : public Sampler {
public:
    explicit SequentialSampler(size_t data_size)
        : data_size_(data_size) {}
    
    std::vector<size_t> indices() const override {
        std::vector<size_t> idx(data_size_);
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
    }
    
    size_t size() const override {
        return data_size_;
    }
    
private:
    size_t data_size_;
};

/**
 * @brief Sample elements randomly
 */
class RandomSampler : public Sampler {
public:
    /**
     * @brief Construct random sampler
     * @param data_size Size of the dataset
     * @param replacement If true, sample with replacement
     * @param num_samples Number of samples (default: data_size)
     * @param seed Random seed
     */
    RandomSampler(size_t data_size,
                  bool replacement = false,
                  size_t num_samples = 0,
                  unsigned int seed = std::random_device{}())
        : data_size_(data_size),
          replacement_(replacement),
          num_samples_(num_samples > 0 ? num_samples : data_size),
          seed_(seed) {}
    
    std::vector<size_t> indices() const override {
        std::mt19937 gen(seed_);
        std::vector<size_t> idx;
        idx.reserve(num_samples_);
        
        if (replacement_) {
            std::uniform_int_distribution<size_t> dist(0, data_size_ - 1);
            for (size_t i = 0; i < num_samples_; ++i) {
                idx.push_back(dist(gen));
            }
        } else {
            idx.resize(data_size_);
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), gen);
            if (num_samples_ < data_size_) {
                idx.resize(num_samples_);
            }
        }
        
        return idx;
    }
    
    size_t size() const override {
        return num_samples_;
    }
    
    void set_seed(unsigned int seed) {
        seed_ = seed;
    }
    
private:
    size_t data_size_;
    bool replacement_;
    size_t num_samples_;
    mutable unsigned int seed_;
};

/**
 * @brief Sample randomly from a subset of indices
 */
class SubsetRandomSampler : public Sampler {
public:
    SubsetRandomSampler(std::vector<size_t> indices,
                        unsigned int seed = std::random_device{}())
        : indices_(std::move(indices)), seed_(seed) {}
    
    std::vector<size_t> indices() const override {
        std::vector<size_t> shuffled = indices_;
        std::mt19937 gen(seed_);
        std::shuffle(shuffled.begin(), shuffled.end(), gen);
        return shuffled;
    }
    
    size_t size() const override {
        return indices_.size();
    }
    
private:
    std::vector<size_t> indices_;
    mutable unsigned int seed_;
};

/**
 * @brief Sample elements with given probabilities (weights)
 */
class WeightedRandomSampler : public Sampler {
public:
    /**
     * @brief Construct weighted sampler
     * @param weights Sampling weights (not necessarily normalized)
     * @param num_samples Number of samples to draw
     * @param replacement Must be true (weighted sampling requires replacement)
     * @param seed Random seed
     */
    WeightedRandomSampler(std::vector<float> weights,
                          size_t num_samples,
                          bool replacement = true,
                          unsigned int seed = std::random_device{}())
        : weights_(std::move(weights)),
          num_samples_(num_samples),
          replacement_(replacement),
          seed_(seed) {
        if (!replacement) {
            throw std::invalid_argument(
                "WeightedRandomSampler requires replacement=true");
        }
        
        // Normalize weights
        float sum = 0.0f;
        for (float w : weights_) sum += w;
        for (float& w : weights_) w /= sum;
    }
    
    std::vector<size_t> indices() const override {
        std::mt19937 gen(seed_);
        std::discrete_distribution<size_t> dist(weights_.begin(), weights_.end());
        
        std::vector<size_t> idx;
        idx.reserve(num_samples_);
        
        for (size_t i = 0; i < num_samples_; ++i) {
            idx.push_back(dist(gen));
        }
        
        return idx;
    }
    
    size_t size() const override {
        return num_samples_;
    }
    
private:
    std::vector<float> weights_;
    size_t num_samples_;
    bool replacement_;
    mutable unsigned int seed_;
};

/**
 * @brief Wrap a sampler to yield batches of indices
 */
class BatchSampler : public Sampler {
public:
    /**
     * @brief Construct batch sampler
     * @param sampler Underlying sampler
     * @param batch_size Size of each batch
     * @param drop_last If true, drop the last incomplete batch
     */
    BatchSampler(std::shared_ptr<Sampler> sampler,
                 size_t batch_size,
                 bool drop_last = false)
        : sampler_(std::move(sampler)),
          batch_size_(batch_size),
          drop_last_(drop_last) {}
    
    /**
     * @brief Get batched indices (flattened)
     */
    std::vector<size_t> indices() const override {
        auto all_indices = sampler_->indices();
        
        size_t num_batches = all_indices.size() / batch_size_;
        if (!drop_last_ && all_indices.size() % batch_size_ != 0) {
            num_batches++;
        }
        
        size_t total = drop_last_ ? num_batches * batch_size_ : all_indices.size();
        all_indices.resize(total);
        
        return all_indices;
    }
    
    /**
     * @brief Get batches as vector of vectors
     */
    std::vector<std::vector<size_t>> batches() const {
        auto all_indices = sampler_->indices();
        std::vector<std::vector<size_t>> result;
        
        for (size_t i = 0; i < all_indices.size(); i += batch_size_) {
            size_t end = std::min(i + batch_size_, all_indices.size());
            if (drop_last_ && end - i < batch_size_) break;
            
            result.emplace_back(all_indices.begin() + i, all_indices.begin() + end);
        }
        
        return result;
    }
    
    size_t size() const override {
        size_t n = sampler_->size();
        if (drop_last_) {
            return n / batch_size_ * batch_size_;
        }
        return n;
    }
    
    size_t num_batches() const {
        size_t n = sampler_->size();
        if (drop_last_) {
            return n / batch_size_;
        }
        return (n + batch_size_ - 1) / batch_size_;
    }
    
    size_t batch_size() const { return batch_size_; }
    
private:
    std::shared_ptr<Sampler> sampler_;
    size_t batch_size_;
    bool drop_last_;
};

/**
 * @brief Sampler for distributed training
 */
class DistributedSampler : public Sampler {
public:
    /**
     * @brief Construct distributed sampler
     * @param data_size Size of the dataset
     * @param num_replicas Number of processes
     * @param rank Rank of current process
     * @param shuffle Whether to shuffle indices
     * @param seed Random seed
     * @param drop_last Whether to drop tail data
     */
    DistributedSampler(size_t data_size,
                       int num_replicas = 1,
                       int rank = 0,
                       bool shuffle = true,
                       unsigned int seed = 0,
                       bool drop_last = false)
        : data_size_(data_size),
          num_replicas_(num_replicas),
          rank_(rank),
          shuffle_(shuffle),
          seed_(seed),
          drop_last_(drop_last),
          epoch_(0) {
        if (rank >= num_replicas || rank < 0) {
            throw std::invalid_argument("Invalid rank");
        }
        
        // Calculate samples per replica
        if (drop_last) {
            num_samples_ = data_size_ / num_replicas;
        } else {
            num_samples_ = (data_size_ + num_replicas - 1) / num_replicas;
        }
        total_size_ = num_samples_ * num_replicas;
    }
    
    std::vector<size_t> indices() const override {
        std::vector<size_t> all_indices(data_size_);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        if (shuffle_) {
            std::mt19937 gen(seed_ + epoch_);
            std::shuffle(all_indices.begin(), all_indices.end(), gen);
        }
        
        // Pad to make evenly divisible
        while (all_indices.size() < total_size_) {
            all_indices.push_back(all_indices[all_indices.size() - total_size_ + data_size_]);
        }
        
        // Extract this replica's portion
        std::vector<size_t> local_indices;
        local_indices.reserve(num_samples_);
        
        for (size_t i = rank_; i < total_size_; i += num_replicas_) {
            local_indices.push_back(all_indices[i]);
        }
        
        return local_indices;
    }
    
    size_t size() const override {
        return num_samples_;
    }
    
    void set_epoch(int epoch) {
        epoch_ = epoch;
    }
    
private:
    size_t data_size_;
    int num_replicas_;
    int rank_;
    bool shuffle_;
    unsigned int seed_;
    bool drop_last_;
    int epoch_;
    size_t num_samples_;
    size_t total_size_;
};

} // namespace data
} // namespace neurova

#endif // NEUROVA_DATA_SAMPLER_HPP
