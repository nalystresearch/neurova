// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file embedding.hpp
 * @brief Embedding layers for NLP and categorical features
 * 
 * Neurova implementation of embedding layers.
 */

#pragma once

#include "tensor.hpp"
#include "layers.hpp"
#include <random>
#include <unordered_map>

namespace neurova {
namespace nn {

/**
 * @brief Standard embedding layer
 * 
 * Maps integer indices to dense vectors.
 */
class Embedding : public Module {
private:
    int num_embeddings_;
    int embedding_dim_;
    int padding_idx_;
    float max_norm_;
    float norm_type_;
    bool scale_grad_by_freq_;
    bool sparse_;
    Parameter weight_;
    
public:
    Embedding(int num_embeddings, int embedding_dim, int padding_idx = -1,
              float max_norm = 0.0f, float norm_type = 2.0f,
              bool scale_grad_by_freq = false, bool sparse = false)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
          padding_idx_(padding_idx), max_norm_(max_norm), norm_type_(norm_type),
          scale_grad_by_freq_(scale_grad_by_freq), sparse_(sparse) {
        
        // Initialize embeddings from N(0, 1)
        std::vector<float> data(num_embeddings * embedding_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& v : data) {
            v = dist(gen);
        }
        
        weight_ = Parameter(Tensor(data, {num_embeddings, embedding_dim}));
        
        // Zero padding vector if specified
        if (padding_idx >= 0 && padding_idx < num_embeddings) {
            for (int i = 0; i < embedding_dim; ++i) {
                weight_.data().data()[padding_idx * embedding_dim + i] = 0.0f;
            }
        }
        
        register_parameter("weight", weight_);
    }
    
    Tensor forward(const std::vector<int>& indices) {
        std::vector<float> result;
        result.reserve(indices.size() * embedding_dim_);
        
        for (int idx : indices) {
            if (idx < 0 || idx >= num_embeddings_) {
                throw std::out_of_range("Embedding index out of range");
            }
            
            for (int i = 0; i < embedding_dim_; ++i) {
                result.push_back(weight_.data().data()[idx * embedding_dim_ + i]);
            }
        }
        
        return Tensor(result, {static_cast<int>(indices.size()), embedding_dim_});
    }
    
    Tensor forward(const Tensor& indices) override {
        // Convert tensor to indices
        std::vector<int> idx_vec;
        idx_vec.reserve(indices.numel());
        
        for (float v : indices.data()) {
            idx_vec.push_back(static_cast<int>(v));
        }
        
        return forward(idx_vec);
    }
    
    // Factory method from pretrained weights
    static Embedding from_pretrained(const Tensor& embeddings, bool freeze = true,
                                      int padding_idx = -1) {
        auto shape = embeddings.shape();
        Embedding emb(shape[0], shape[1], padding_idx);
        emb.weight_.data() = embeddings;
        
        if (freeze) {
            emb.weight_.set_requires_grad(false);
        }
        
        return emb;
    }
    
    int num_embeddings() const { return num_embeddings_; }
    int embedding_dim() const { return embedding_dim_; }
    const Parameter& weight() const { return weight_; }
};

/**
 * @brief Embedding bag - pooled embeddings
 * 
 * Computes sums or means of bags of embeddings.
 */
class EmbeddingBag : public Module {
public:
    enum class Mode { Sum, Mean, Max };
    
private:
    int num_embeddings_;
    int embedding_dim_;
    float max_norm_;
    float norm_type_;
    bool scale_grad_by_freq_;
    Mode mode_;
    bool sparse_;
    bool include_last_offset_;
    int padding_idx_;
    Parameter weight_;
    
public:
    EmbeddingBag(int num_embeddings, int embedding_dim, float max_norm = 0.0f,
                 float norm_type = 2.0f, bool scale_grad_by_freq = false,
                 Mode mode = Mode::Mean, bool sparse = false,
                 bool include_last_offset = false, int padding_idx = -1)
        : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim),
          max_norm_(max_norm), norm_type_(norm_type), scale_grad_by_freq_(scale_grad_by_freq),
          mode_(mode), sparse_(sparse), include_last_offset_(include_last_offset),
          padding_idx_(padding_idx) {
        
        // Initialize embeddings
        std::vector<float> data(num_embeddings * embedding_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& v : data) {
            v = dist(gen);
        }
        
        weight_ = Parameter(Tensor(data, {num_embeddings, embedding_dim}));
        register_parameter("weight", weight_);
    }
    
    Tensor forward(const std::vector<int>& indices, const std::vector<int>& offsets) {
        int num_bags = static_cast<int>(offsets.size());
        if (!include_last_offset_) {
            num_bags = static_cast<int>(offsets.size());
        } else {
            num_bags = static_cast<int>(offsets.size()) - 1;
        }
        
        std::vector<float> result(num_bags * embedding_dim_, 0.0f);
        
        for (int bag = 0; bag < num_bags; ++bag) {
            int start = offsets[bag];
            int end = (bag + 1 < static_cast<int>(offsets.size())) ? offsets[bag + 1] : static_cast<int>(indices.size());
            int count = end - start;
            
            if (count == 0) continue;
            
            if (mode_ == Mode::Max) {
                // Initialize with very negative values
                for (int i = 0; i < embedding_dim_; ++i) {
                    result[bag * embedding_dim_ + i] = -std::numeric_limits<float>::infinity();
                }
            }
            
            for (int j = start; j < end; ++j) {
                int idx = indices[j];
                if (idx == padding_idx_) continue;
                
                for (int i = 0; i < embedding_dim_; ++i) {
                    float val = weight_.data().data()[idx * embedding_dim_ + i];
                    
                    if (mode_ == Mode::Max) {
                        result[bag * embedding_dim_ + i] = std::max(result[bag * embedding_dim_ + i], val);
                    } else {
                        result[bag * embedding_dim_ + i] += val;
                    }
                }
            }
            
            if (mode_ == Mode::Mean && count > 0) {
                for (int i = 0; i < embedding_dim_; ++i) {
                    result[bag * embedding_dim_ + i] /= count;
                }
            }
        }
        
        return Tensor(result, {num_bags, embedding_dim_});
    }
    
    Tensor forward(const Tensor& indices) override {
        // For single bag
        std::vector<int> idx_vec;
        for (float v : indices.data()) {
            idx_vec.push_back(static_cast<int>(v));
        }
        return forward(idx_vec, {0});
    }
    
    int num_embeddings() const { return num_embeddings_; }
    int embedding_dim() const { return embedding_dim_; }
};

/**
 * @brief Positional encoding for transformers
 */
class PositionalEncoding : public Module {
private:
    int d_model_;
    int max_len_;
    float dropout_p_;
    Tensor pe_;
    bool training_ = true;
    
public:
    PositionalEncoding(int d_model, int max_len = 5000, float dropout = 0.1f)
        : d_model_(d_model), max_len_(max_len), dropout_p_(dropout) {
        
        // Create positional encoding
        std::vector<float> pe(max_len * d_model, 0.0f);
        
        for (int pos = 0; pos < max_len; ++pos) {
            for (int i = 0; i < d_model; i += 2) {
                float div_term = std::exp(-static_cast<float>(i) * std::log(10000.0f) / d_model);
                pe[pos * d_model + i] = std::sin(pos * div_term);
                if (i + 1 < d_model) {
                    pe[pos * d_model + i + 1] = std::cos(pos * div_term);
                }
            }
        }
        
        pe_ = Tensor(pe, {max_len, d_model});
    }
    
    Tensor forward(const Tensor& x) override {
        auto shape = x.shape();
        int seq_len = shape[0];
        
        // Add positional encoding
        std::vector<float> result(x.numel());
        int batch_stride = (shape.size() > 2) ? shape[2] : 1;
        
        for (int pos = 0; pos < seq_len && pos < max_len_; ++pos) {
            for (int i = 0; i < d_model_; ++i) {
                result[pos * d_model_ + i] = x.data()[pos * d_model_ + i] + pe_.data()[pos * d_model_ + i];
            }
        }
        
        // Apply dropout during training
        if (training_ && dropout_p_ > 0.0f) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            for (auto& v : result) {
                if (dist(gen) < dropout_p_) {
                    v = 0.0f;
                } else {
                    v /= (1.0f - dropout_p_);
                }
            }
        }
        
        return Tensor(result, shape);
    }
    
    void train(bool mode = true) { training_ = mode; }
    void eval() { train(false); }
};

/**
 * @brief Learned positional embedding
 */
class LearnedPositionalEmbedding : public Module {
private:
    int max_positions_;
    int embedding_dim_;
    Parameter weight_;
    
public:
    LearnedPositionalEmbedding(int max_positions, int embedding_dim)
        : max_positions_(max_positions), embedding_dim_(embedding_dim) {
        
        std::vector<float> data(max_positions * embedding_dim);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.02f);
        
        for (auto& v : data) {
            v = dist(gen);
        }
        
        weight_ = Parameter(Tensor(data, {max_positions, embedding_dim}));
        register_parameter("weight", weight_);
    }
    
    Tensor forward(const Tensor& positions) override {
        std::vector<float> result;
        
        for (float p : positions.data()) {
            int pos = static_cast<int>(p);
            if (pos >= 0 && pos < max_positions_) {
                for (int i = 0; i < embedding_dim_; ++i) {
                    result.push_back(weight_.data().data()[pos * embedding_dim_ + i]);
                }
            }
        }
        
        return Tensor(result, {static_cast<int>(positions.numel()), embedding_dim_});
    }
};

/**
 * @brief Rotary positional embedding (RoPE)
 */
class RotaryEmbedding : public Module {
private:
    int dim_;
    int max_seq_len_;
    float base_;
    std::vector<float> cos_cached_;
    std::vector<float> sin_cached_;
    
public:
    RotaryEmbedding(int dim, int max_seq_len = 2048, float base = 10000.0f)
        : dim_(dim), max_seq_len_(max_seq_len), base_(base) {
        
        // Precompute rotary embeddings
        std::vector<float> inv_freq(dim / 2);
        for (int i = 0; i < dim / 2; ++i) {
            inv_freq[i] = 1.0f / std::pow(base, 2.0f * i / dim);
        }
        
        cos_cached_.resize(max_seq_len * dim / 2);
        sin_cached_.resize(max_seq_len * dim / 2);
        
        for (int pos = 0; pos < max_seq_len; ++pos) {
            for (int i = 0; i < dim / 2; ++i) {
                float angle = pos * inv_freq[i];
                cos_cached_[pos * (dim / 2) + i] = std::cos(angle);
                sin_cached_[pos * (dim / 2) + i] = std::sin(angle);
            }
        }
    }
    
    std::pair<Tensor, Tensor> forward(int seq_len) {
        if (seq_len > max_seq_len_) {
            throw std::runtime_error("Sequence length exceeds max_seq_len");
        }
        
        std::vector<float> cos_out(seq_len * dim_ / 2);
        std::vector<float> sin_out(seq_len * dim_ / 2);
        
        std::copy(cos_cached_.begin(), cos_cached_.begin() + seq_len * dim_ / 2, cos_out.begin());
        std::copy(sin_cached_.begin(), sin_cached_.begin() + seq_len * dim_ / 2, sin_out.begin());
        
        return {
            Tensor(cos_out, {seq_len, dim_ / 2}),
            Tensor(sin_out, {seq_len, dim_ / 2})
        };
    }
    
    Tensor forward(const Tensor& x) override {
        // Apply rotary embedding to input
        auto shape = x.shape();
        int seq_len = shape[0];
        
        auto [cos_emb, sin_emb] = forward(seq_len);
        
        // Rotate pairs of dimensions
        std::vector<float> result(x.numel());
        int half_dim = dim_ / 2;
        
        for (int pos = 0; pos < seq_len; ++pos) {
            for (int i = 0; i < half_dim; ++i) {
                float cos_val = cos_emb.data()[pos * half_dim + i];
                float sin_val = sin_emb.data()[pos * half_dim + i];
                
                float x1 = x.data()[pos * dim_ + i];
                float x2 = x.data()[pos * dim_ + i + half_dim];
                
                result[pos * dim_ + i] = x1 * cos_val - x2 * sin_val;
                result[pos * dim_ + i + half_dim] = x1 * sin_val + x2 * cos_val;
            }
        }
        
        return Tensor(result, shape);
    }
};

} // namespace nn
} // namespace neurova
