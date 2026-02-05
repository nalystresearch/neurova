// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file attention.hpp
 * @brief Attention mechanisms for transformers
 * 
 * Neurova implementation of multi-head attention and transformer components.
 */

#pragma once

#include "tensor.hpp"
#include "layers.hpp"
#include "linear.hpp"
#include "dropout.hpp"
#include <cmath>
#include <random>
#include <limits>

namespace neurova {
namespace nn {

/**
 * @brief Scaled dot-product attention
 */
inline Tensor scaled_dot_product_attention(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor* mask = nullptr, float dropout_p = 0.0f, bool training = true) {
    
    int d_k = query.shape().back();
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    
    // Compute attention scores: Q @ K^T / sqrt(d_k)
    Tensor scores = query.matmul(key.transpose(-2, -1)) * scale;
    
    // Apply mask if provided
    if (mask != nullptr) {
        std::vector<float> masked_scores(scores.numel());
        for (size_t i = 0; i < scores.numel(); ++i) {
            if (mask->data()[i % mask->numel()] == 0.0f) {
                masked_scores[i] = -std::numeric_limits<float>::infinity();
            } else {
                masked_scores[i] = scores.data()[i];
            }
        }
        scores = Tensor(masked_scores, scores.shape());
    }
    
    // Softmax over last dimension
    auto shape = scores.shape();
    int last_dim = shape.back();
    std::vector<float> attention_weights(scores.numel());
    
    int num_rows = scores.numel() / last_dim;
    for (int row = 0; row < num_rows; ++row) {
        // Find max for numerical stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < last_dim; ++i) {
            max_val = std::max(max_val, scores.data()[row * last_dim + i]);
        }
        
        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < last_dim; ++i) {
            attention_weights[row * last_dim + i] = std::exp(scores.data()[row * last_dim + i] - max_val);
            sum_exp += attention_weights[row * last_dim + i];
        }
        
        // Normalize
        for (int i = 0; i < last_dim; ++i) {
            attention_weights[row * last_dim + i] /= sum_exp;
        }
    }
    
    Tensor attn_weights(attention_weights, shape);
    
    // Apply dropout
    if (training && dropout_p > 0.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        std::vector<float> dropped(attn_weights.numel());
        for (size_t i = 0; i < attn_weights.numel(); ++i) {
            if (dist(gen) < dropout_p) {
                dropped[i] = 0.0f;
            } else {
                dropped[i] = attn_weights.data()[i] / (1.0f - dropout_p);
            }
        }
        attn_weights = Tensor(dropped, shape);
    }
    
    // Compute output: attention_weights @ V
    return attn_weights.matmul(value);
}

/**
 * @brief Multi-head attention
 */
class MultiheadAttention : public Module {
private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    float dropout_p_;
    bool bias_;
    bool add_zero_attn_;
    bool batch_first_;
    
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear out_proj_;
    
    bool training_ = true;
    
public:
    MultiheadAttention(int embed_dim, int num_heads, float dropout = 0.0f,
                       bool bias = true, bool add_zero_attn = false,
                       int kdim = -1, int vdim = -1, bool batch_first = false)
        : embed_dim_(embed_dim), num_heads_(num_heads), 
          head_dim_(embed_dim / num_heads), dropout_p_(dropout),
          bias_(bias), add_zero_attn_(add_zero_attn), batch_first_(batch_first),
          q_proj_(embed_dim, embed_dim, bias),
          k_proj_(kdim > 0 ? kdim : embed_dim, embed_dim, bias),
          v_proj_(vdim > 0 ? vdim : embed_dim, embed_dim, bias),
          out_proj_(embed_dim, embed_dim, bias) {
        
        if (embed_dim % num_heads != 0) {
            throw std::invalid_argument("embed_dim must be divisible by num_heads");
        }
    }
    
    std::pair<Tensor, Tensor> forward(const Tensor& query, const Tensor& key, const Tensor& value,
                                       const Tensor* key_padding_mask = nullptr,
                                       bool need_weights = true,
                                       const Tensor* attn_mask = nullptr,
                                       bool average_attn_weights = true) {
        
        // Project Q, K, V
        Tensor q = q_proj_.forward(query);
        Tensor k = k_proj_.forward(key);
        Tensor v = v_proj_.forward(value);
        
        // Reshape for multi-head attention
        auto q_shape = q.shape();
        int seq_len = q_shape[0];
        
        // Simplified: single batch, compute attention
        Tensor attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p_, training_);
        
        // Project output
        Tensor output = out_proj_.forward(attn_output);
        
        // Attention weights (if needed)
        Tensor attn_weights;
        if (need_weights) {
            float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
            attn_weights = q.matmul(k.transpose(-2, -1)) * scale;
        }
        
        return {output, attn_weights};
    }
    
    Tensor forward(const Tensor& x) override {
        // Self-attention
        auto [output, _] = forward(x, x, x);
        return output;
    }
    
    void train(bool mode = true) { training_ = mode; }
    void eval() { train(false); }
    
    int embed_dim() const { return embed_dim_; }
    int num_heads() const { return num_heads_; }
};

/**
 * @brief Transformer encoder layer
 */
class TransformerEncoderLayer : public Module {
private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_p_;
    std::string activation_;
    bool batch_first_;
    bool norm_first_;
    
    MultiheadAttention self_attn_;
    Linear linear1_;
    Linear linear2_;
    Dropout dropout_;
    Dropout dropout1_;
    Dropout dropout2_;
    
    // Layer norms (simplified)
    std::vector<float> norm1_weight_;
    std::vector<float> norm1_bias_;
    std::vector<float> norm2_weight_;
    std::vector<float> norm2_bias_;
    float norm_eps_ = 1e-5f;
    
    bool training_ = true;
    
    Tensor layer_norm(const Tensor& x, const std::vector<float>& weight, 
                      const std::vector<float>& bias) {
        auto shape = x.shape();
        int normalized_shape = shape.back();
        
        std::vector<float> result(x.numel());
        int num_elements = x.numel() / normalized_shape;
        
        for (int i = 0; i < num_elements; ++i) {
            // Compute mean and variance
            float mean = 0.0f, var = 0.0f;
            for (int j = 0; j < normalized_shape; ++j) {
                mean += x.data()[i * normalized_shape + j];
            }
            mean /= normalized_shape;
            
            for (int j = 0; j < normalized_shape; ++j) {
                float diff = x.data()[i * normalized_shape + j] - mean;
                var += diff * diff;
            }
            var /= normalized_shape;
            
            // Normalize
            float inv_std = 1.0f / std::sqrt(var + norm_eps_);
            for (int j = 0; j < normalized_shape; ++j) {
                float normalized = (x.data()[i * normalized_shape + j] - mean) * inv_std;
                result[i * normalized_shape + j] = normalized * weight[j] + bias[j];
            }
        }
        
        return Tensor(result, shape);
    }
    
    Tensor activation_fn(const Tensor& x) {
        if (activation_ == "gelu") {
            return x.gelu();
        } else {
            return x.relu();
        }
    }
    
public:
    TransformerEncoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, const std::string& activation = "relu",
                            bool batch_first = false, bool norm_first = false)
        : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
          dropout_p_(dropout), activation_(activation), batch_first_(batch_first),
          norm_first_(norm_first),
          self_attn_(d_model, nhead, dropout),
          linear1_(d_model, dim_feedforward),
          linear2_(dim_feedforward, d_model),
          dropout_(dropout), dropout1_(dropout), dropout2_(dropout) {
        
        // Initialize layer norm parameters
        norm1_weight_.resize(d_model, 1.0f);
        norm1_bias_.resize(d_model, 0.0f);
        norm2_weight_.resize(d_model, 1.0f);
        norm2_bias_.resize(d_model, 0.0f);
    }
    
    Tensor forward(const Tensor& src, const Tensor* src_mask = nullptr,
                   const Tensor* src_key_padding_mask = nullptr) {
        
        Tensor x = src;
        
        if (norm_first_) {
            // Pre-norm architecture
            Tensor normed = layer_norm(x, norm1_weight_, norm1_bias_);
            auto [attn_out, _] = self_attn_.forward(normed, normed, normed, src_key_padding_mask, false, src_mask);
            x = x + dropout1_.forward(attn_out);
            
            normed = layer_norm(x, norm2_weight_, norm2_bias_);
            Tensor ff_out = linear2_.forward(dropout_.forward(activation_fn(linear1_.forward(normed))));
            x = x + dropout2_.forward(ff_out);
        } else {
            // Post-norm architecture
            auto [attn_out, _] = self_attn_.forward(x, x, x, src_key_padding_mask, false, src_mask);
            x = layer_norm(x + dropout1_.forward(attn_out), norm1_weight_, norm1_bias_);
            
            Tensor ff_out = linear2_.forward(dropout_.forward(activation_fn(linear1_.forward(x))));
            x = layer_norm(x + dropout2_.forward(ff_out), norm2_weight_, norm2_bias_);
        }
        
        return x;
    }
    
    Tensor forward(const Tensor& x) override {
        return forward(x, nullptr, nullptr);
    }
    
    void train(bool mode = true) { 
        training_ = mode;
        self_attn_.train(mode);
        dropout_.train(mode);
        dropout1_.train(mode);
        dropout2_.train(mode);
    }
    void eval() { train(false); }
};

/**
 * @brief Transformer decoder layer
 */
class TransformerDecoderLayer : public Module {
private:
    int d_model_;
    int nhead_;
    int dim_feedforward_;
    float dropout_p_;
    std::string activation_;
    bool batch_first_;
    bool norm_first_;
    
    MultiheadAttention self_attn_;
    MultiheadAttention multihead_attn_;
    Linear linear1_;
    Linear linear2_;
    Dropout dropout_;
    Dropout dropout1_;
    Dropout dropout2_;
    Dropout dropout3_;
    
    std::vector<float> norm1_weight_, norm1_bias_;
    std::vector<float> norm2_weight_, norm2_bias_;
    std::vector<float> norm3_weight_, norm3_bias_;
    float norm_eps_ = 1e-5f;
    
    bool training_ = true;
    
    Tensor layer_norm(const Tensor& x, const std::vector<float>& weight,
                      const std::vector<float>& bias) {
        auto shape = x.shape();
        int normalized_shape = shape.back();
        
        std::vector<float> result(x.numel());
        int num_elements = x.numel() / normalized_shape;
        
        for (int i = 0; i < num_elements; ++i) {
            float mean = 0.0f, var = 0.0f;
            for (int j = 0; j < normalized_shape; ++j) {
                mean += x.data()[i * normalized_shape + j];
            }
            mean /= normalized_shape;
            
            for (int j = 0; j < normalized_shape; ++j) {
                float diff = x.data()[i * normalized_shape + j] - mean;
                var += diff * diff;
            }
            var /= normalized_shape;
            
            float inv_std = 1.0f / std::sqrt(var + norm_eps_);
            for (int j = 0; j < normalized_shape; ++j) {
                float normalized = (x.data()[i * normalized_shape + j] - mean) * inv_std;
                result[i * normalized_shape + j] = normalized * weight[j] + bias[j];
            }
        }
        
        return Tensor(result, shape);
    }
    
    Tensor activation_fn(const Tensor& x) {
        if (activation_ == "gelu") {
            return x.gelu();
        }
        return x.relu();
    }
    
public:
    TransformerDecoderLayer(int d_model, int nhead, int dim_feedforward = 2048,
                            float dropout = 0.1f, const std::string& activation = "relu",
                            bool batch_first = false, bool norm_first = false)
        : d_model_(d_model), nhead_(nhead), dim_feedforward_(dim_feedforward),
          dropout_p_(dropout), activation_(activation), batch_first_(batch_first),
          norm_first_(norm_first),
          self_attn_(d_model, nhead, dropout),
          multihead_attn_(d_model, nhead, dropout),
          linear1_(d_model, dim_feedforward),
          linear2_(dim_feedforward, d_model),
          dropout_(dropout), dropout1_(dropout), dropout2_(dropout), dropout3_(dropout) {
        
        norm1_weight_.resize(d_model, 1.0f);
        norm1_bias_.resize(d_model, 0.0f);
        norm2_weight_.resize(d_model, 1.0f);
        norm2_bias_.resize(d_model, 0.0f);
        norm3_weight_.resize(d_model, 1.0f);
        norm3_bias_.resize(d_model, 0.0f);
    }
    
    Tensor forward(const Tensor& tgt, const Tensor& memory,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr,
                   const Tensor* tgt_key_padding_mask = nullptr,
                   const Tensor* memory_key_padding_mask = nullptr) {
        
        Tensor x = tgt;
        
        if (norm_first_) {
            // Self-attention
            Tensor normed = layer_norm(x, norm1_weight_, norm1_bias_);
            auto [self_attn_out, _1] = self_attn_.forward(normed, normed, normed, tgt_key_padding_mask, false, tgt_mask);
            x = x + dropout1_.forward(self_attn_out);
            
            // Cross-attention
            normed = layer_norm(x, norm2_weight_, norm2_bias_);
            auto [cross_attn_out, _2] = multihead_attn_.forward(normed, memory, memory, memory_key_padding_mask, false, memory_mask);
            x = x + dropout2_.forward(cross_attn_out);
            
            // Feed-forward
            normed = layer_norm(x, norm3_weight_, norm3_bias_);
            Tensor ff_out = linear2_.forward(dropout_.forward(activation_fn(linear1_.forward(normed))));
            x = x + dropout3_.forward(ff_out);
        } else {
            // Post-norm
            auto [self_attn_out, _1] = self_attn_.forward(x, x, x, tgt_key_padding_mask, false, tgt_mask);
            x = layer_norm(x + dropout1_.forward(self_attn_out), norm1_weight_, norm1_bias_);
            
            auto [cross_attn_out, _2] = multihead_attn_.forward(x, memory, memory, memory_key_padding_mask, false, memory_mask);
            x = layer_norm(x + dropout2_.forward(cross_attn_out), norm2_weight_, norm2_bias_);
            
            Tensor ff_out = linear2_.forward(dropout_.forward(activation_fn(linear1_.forward(x))));
            x = layer_norm(x + dropout3_.forward(ff_out), norm3_weight_, norm3_bias_);
        }
        
        return x;
    }
    
    Tensor forward(const Tensor& x) override {
        return forward(x, x);
    }
    
    void train(bool mode = true) {
        training_ = mode;
        self_attn_.train(mode);
        multihead_attn_.train(mode);
        dropout_.train(mode);
        dropout1_.train(mode);
        dropout2_.train(mode);
        dropout3_.train(mode);
    }
    void eval() { train(false); }
};

/**
 * @brief Transformer encoder
 */
class TransformerEncoder : public Module {
private:
    std::vector<std::unique_ptr<TransformerEncoderLayer>> layers_;
    std::vector<float> norm_weight_;
    std::vector<float> norm_bias_;
    float norm_eps_ = 1e-5f;
    bool enable_nested_tensor_;
    
    Tensor layer_norm(const Tensor& x) {
        if (norm_weight_.empty()) return x;
        
        auto shape = x.shape();
        int normalized_shape = shape.back();
        
        std::vector<float> result(x.numel());
        int num_elements = x.numel() / normalized_shape;
        
        for (int i = 0; i < num_elements; ++i) {
            float mean = 0.0f, var = 0.0f;
            for (int j = 0; j < normalized_shape; ++j) {
                mean += x.data()[i * normalized_shape + j];
            }
            mean /= normalized_shape;
            
            for (int j = 0; j < normalized_shape; ++j) {
                float diff = x.data()[i * normalized_shape + j] - mean;
                var += diff * diff;
            }
            var /= normalized_shape;
            
            float inv_std = 1.0f / std::sqrt(var + norm_eps_);
            for (int j = 0; j < normalized_shape; ++j) {
                float normalized = (x.data()[i * normalized_shape + j] - mean) * inv_std;
                result[i * normalized_shape + j] = normalized * norm_weight_[j] + norm_bias_[j];
            }
        }
        
        return Tensor(result, shape);
    }
    
public:
    TransformerEncoder(const TransformerEncoderLayer& encoder_layer, int num_layers,
                       int d_model = -1, bool enable_nested_tensor = true)
        : enable_nested_tensor_(enable_nested_tensor) {
        
        // Create copies of encoder layer
        int d = (d_model > 0) ? d_model : 512;  // default
        int nhead = 8;
        
        for (int i = 0; i < num_layers; ++i) {
            layers_.push_back(std::make_unique<TransformerEncoderLayer>(d, nhead));
        }
        
        if (d_model > 0) {
            norm_weight_.resize(d_model, 1.0f);
            norm_bias_.resize(d_model, 0.0f);
        }
    }
    
    Tensor forward(const Tensor& src, const Tensor* mask = nullptr,
                   const Tensor* src_key_padding_mask = nullptr) {
        
        Tensor output = src;
        
        for (auto& layer : layers_) {
            output = layer->forward(output, mask, src_key_padding_mask);
        }
        
        return layer_norm(output);
    }
    
    Tensor forward(const Tensor& x) override {
        return forward(x, nullptr, nullptr);
    }
};

/**
 * @brief Transformer decoder
 */
class TransformerDecoder : public Module {
private:
    std::vector<std::unique_ptr<TransformerDecoderLayer>> layers_;
    std::vector<float> norm_weight_;
    std::vector<float> norm_bias_;
    float norm_eps_ = 1e-5f;
    
    Tensor layer_norm(const Tensor& x) {
        if (norm_weight_.empty()) return x;
        
        auto shape = x.shape();
        int normalized_shape = shape.back();
        
        std::vector<float> result(x.numel());
        int num_elements = x.numel() / normalized_shape;
        
        for (int i = 0; i < num_elements; ++i) {
            float mean = 0.0f, var = 0.0f;
            for (int j = 0; j < normalized_shape; ++j) {
                mean += x.data()[i * normalized_shape + j];
            }
            mean /= normalized_shape;
            
            for (int j = 0; j < normalized_shape; ++j) {
                float diff = x.data()[i * normalized_shape + j] - mean;
                var += diff * diff;
            }
            var /= normalized_shape;
            
            float inv_std = 1.0f / std::sqrt(var + norm_eps_);
            for (int j = 0; j < normalized_shape; ++j) {
                float normalized = (x.data()[i * normalized_shape + j] - mean) * inv_std;
                result[i * normalized_shape + j] = normalized * norm_weight_[j] + norm_bias_[j];
            }
        }
        
        return Tensor(result, shape);
    }
    
public:
    TransformerDecoder(const TransformerDecoderLayer& decoder_layer, int num_layers,
                       int d_model = -1)  {
        
        int d = (d_model > 0) ? d_model : 512;
        int nhead = 8;
        
        for (int i = 0; i < num_layers; ++i) {
            layers_.push_back(std::make_unique<TransformerDecoderLayer>(d, nhead));
        }
        
        if (d_model > 0) {
            norm_weight_.resize(d_model, 1.0f);
            norm_bias_.resize(d_model, 0.0f);
        }
    }
    
    Tensor forward(const Tensor& tgt, const Tensor& memory,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr,
                   const Tensor* tgt_key_padding_mask = nullptr,
                   const Tensor* memory_key_padding_mask = nullptr) {
        
        Tensor output = tgt;
        
        for (auto& layer : layers_) {
            output = layer->forward(output, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask, memory_key_padding_mask);
        }
        
        return layer_norm(output);
    }
    
    Tensor forward(const Tensor& x) override {
        return forward(x, x);
    }
};

/**
 * @brief Full Transformer model
 */
class Transformer : public Module {
private:
    int d_model_;
    int nhead_;
    int num_encoder_layers_;
    int num_decoder_layers_;
    int dim_feedforward_;
    float dropout_;
    std::string activation_;
    bool batch_first_;
    bool norm_first_;
    
    std::unique_ptr<TransformerEncoder> encoder_;
    std::unique_ptr<TransformerDecoder> decoder_;
    
public:
    Transformer(int d_model = 512, int nhead = 8, int num_encoder_layers = 6,
                int num_decoder_layers = 6, int dim_feedforward = 2048,
                float dropout = 0.1f, const std::string& activation = "relu",
                bool batch_first = false, bool norm_first = false)
        : d_model_(d_model), nhead_(nhead), num_encoder_layers_(num_encoder_layers),
          num_decoder_layers_(num_decoder_layers), dim_feedforward_(dim_feedforward),
          dropout_(dropout), activation_(activation), batch_first_(batch_first),
          norm_first_(norm_first) {
        
        TransformerEncoderLayer encoder_layer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, norm_first);
        TransformerDecoderLayer decoder_layer(d_model, nhead, dim_feedforward, dropout, activation, batch_first, norm_first);
        
        encoder_ = std::make_unique<TransformerEncoder>(encoder_layer, num_encoder_layers, d_model);
        decoder_ = std::make_unique<TransformerDecoder>(decoder_layer, num_decoder_layers, d_model);
    }
    
    Tensor forward(const Tensor& src, const Tensor& tgt,
                   const Tensor* src_mask = nullptr,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr,
                   const Tensor* src_key_padding_mask = nullptr,
                   const Tensor* tgt_key_padding_mask = nullptr,
                   const Tensor* memory_key_padding_mask = nullptr) {
        
        Tensor memory = encoder_->forward(src, src_mask, src_key_padding_mask);
        return decoder_->forward(tgt, memory, tgt_mask, memory_mask,
                                  tgt_key_padding_mask, memory_key_padding_mask);
    }
    
    Tensor forward(const Tensor& x) override {
        return forward(x, x);
    }
    
    Tensor encode(const Tensor& src, const Tensor* src_mask = nullptr,
                  const Tensor* src_key_padding_mask = nullptr) {
        return encoder_->forward(src, src_mask, src_key_padding_mask);
    }
    
    Tensor decode(const Tensor& tgt, const Tensor& memory,
                  const Tensor* tgt_mask = nullptr,
                  const Tensor* memory_mask = nullptr) {
        return decoder_->forward(tgt, memory, tgt_mask, memory_mask);
    }
    
    /**
     * @brief Generate causal mask for autoregressive decoding
     */
    static Tensor generate_square_subsequent_mask(int sz) {
        std::vector<float> mask(sz * sz, 0.0f);
        
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) {
                if (j > i) {
                    mask[i * sz + j] = -std::numeric_limits<float>::infinity();
                }
            }
        }
        
        return Tensor(mask, {sz, sz});
    }
    
    int d_model() const { return d_model_; }
    int nhead() const { return nhead_; }
};

} // namespace nn
} // namespace neurova
