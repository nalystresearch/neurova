// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file loss.hpp
 * @brief Loss functions for neural network training
 * 
 * Neurova implementation of common loss functions.
 */

#pragma once

#include "layers.hpp"
#include <cmath>
#include <algorithm>

namespace neurova {
namespace nn {

/**
 * @brief Mean Squared Error Loss
 * 
 * L = (1/n) * sum((y_pred - y_true)^2)
 */
class MSELoss : public Module {
private:
    std::string reduction_;  // "mean", "sum", "none"
    
public:
    explicit MSELoss(const std::string& reduction = "mean") : reduction_(reduction) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        auto diff = pred->sub(target);
        auto sq = diff->mul(diff);
        
        if (reduction_ == "none") {
            return sq;
        } else if (reduction_ == "sum") {
            return sq->sum();
        } else {
            return sq->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("MSELoss requires two inputs (pred, target)");
    }
};

/**
 * @brief L1 Loss (Mean Absolute Error)
 * 
 * L = (1/n) * sum(|y_pred - y_true|)
 */
class L1Loss : public Module {
private:
    std::string reduction_;
    
public:
    explicit L1Loss(const std::string& reduction = "mean") : reduction_(reduction) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> result(pred->size());
        for (size_t i = 0; i < pred->size(); ++i) {
            result[i] = std::abs(pred->data[i] - target->data[i]);
        }
        
        auto out = std::make_shared<Tensor>(result, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("L1Loss requires two inputs");
    }
};

/**
 * @brief Cross Entropy Loss
 * 
 * Combines LogSoftmax and NLLLoss.
 */
class CrossEntropyLoss : public Module {
private:
    std::string reduction_;
    float label_smoothing_;
    
public:
    CrossEntropyLoss(const std::string& reduction = "mean", float label_smoothing = 0.0f)
        : reduction_(reduction), label_smoothing_(label_smoothing) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        // pred: (N, C), target: (N,) class indices
        size_t N = pred->shape[0];
        size_t C = pred->shape[1];
        
        std::vector<float> losses(N);
        
        for (size_t n = 0; n < N; ++n) {
            // Compute log softmax for this sample
            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < C; ++c) {
                max_val = std::max(max_val, pred->data[n * C + c]);
            }
            
            float sum_exp = 0.0f;
            for (size_t c = 0; c < C; ++c) {
                sum_exp += std::exp(pred->data[n * C + c] - max_val);
            }
            float log_sum = max_val + std::log(sum_exp);
            
            // Get target class
            size_t target_class = static_cast<size_t>(target->data[n]);
            
            // Negative log likelihood
            losses[n] = -(pred->data[n * C + target_class] - log_sum);
            
            // Label smoothing
            if (label_smoothing_ > 0) {
                float smooth_loss = 0.0f;
                for (size_t c = 0; c < C; ++c) {
                    smooth_loss -= (pred->data[n * C + c] - log_sum) / C;
                }
                losses[n] = (1.0f - label_smoothing_) * losses[n] + label_smoothing_ * smooth_loss;
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, {N}, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("CrossEntropyLoss requires two inputs");
    }
};

/**
 * @brief Negative Log Likelihood Loss
 */
class NLLLoss : public Module {
private:
    std::string reduction_;
    
public:
    explicit NLLLoss(const std::string& reduction = "mean") : reduction_(reduction) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        size_t N = pred->shape[0];
        size_t C = pred->shape[1];
        
        std::vector<float> losses(N);
        
        for (size_t n = 0; n < N; ++n) {
            size_t target_class = static_cast<size_t>(target->data[n]);
            losses[n] = -pred->data[n * C + target_class];
        }
        
        auto out = std::make_shared<Tensor>(losses, {N}, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("NLLLoss requires two inputs");
    }
};

/**
 * @brief Binary Cross Entropy Loss
 */
class BCELoss : public Module {
private:
    std::string reduction_;
    
public:
    explicit BCELoss(const std::string& reduction = "mean") : reduction_(reduction) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        const float eps = 1e-7f;
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float p = std::clamp(pred->data[i], eps, 1.0f - eps);
            float t = target->data[i];
            losses[i] = -(t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("BCELoss requires two inputs");
    }
};

/**
 * @brief Binary Cross Entropy with Logits Loss (more numerically stable)
 */
class BCEWithLogitsLoss : public Module {
private:
    std::string reduction_;
    float pos_weight_;
    
public:
    BCEWithLogitsLoss(const std::string& reduction = "mean", float pos_weight = 1.0f)
        : reduction_(reduction), pos_weight_(pos_weight) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float x = pred->data[i];
            float t = target->data[i];
            // max(x, 0) - x*t + log(1 + exp(-|x|))
            float loss = std::max(x, 0.0f) - x * t * pos_weight_ + 
                        std::log(1.0f + std::exp(-std::abs(x)));
            losses[i] = loss;
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("BCEWithLogitsLoss requires two inputs");
    }
};

/**
 * @brief Smooth L1 Loss (Huber Loss)
 */
class SmoothL1Loss : public Module {
private:
    std::string reduction_;
    float beta_;
    
public:
    SmoothL1Loss(const std::string& reduction = "mean", float beta = 1.0f)
        : reduction_(reduction), beta_(beta) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float diff = std::abs(pred->data[i] - target->data[i]);
            if (diff < beta_) {
                losses[i] = 0.5f * diff * diff / beta_;
            } else {
                losses[i] = diff - 0.5f * beta_;
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("SmoothL1Loss requires two inputs");
    }
};

/**
 * @brief Huber Loss
 */
class HuberLoss : public Module {
private:
    std::string reduction_;
    float delta_;
    
public:
    HuberLoss(const std::string& reduction = "mean", float delta = 1.0f)
        : reduction_(reduction), delta_(delta) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float diff = std::abs(pred->data[i] - target->data[i]);
            if (diff <= delta_) {
                losses[i] = 0.5f * diff * diff;
            } else {
                losses[i] = delta_ * (diff - 0.5f * delta_);
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("HuberLoss requires two inputs");
    }
};

/**
 * @brief KL Divergence Loss
 */
class KLDivLoss : public Module {
private:
    std::string reduction_;
    bool log_target_;
    
public:
    KLDivLoss(const std::string& reduction = "mean", bool log_target = false)
        : reduction_(reduction), log_target_(log_target) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> losses(pred->size());
        const float eps = 1e-7f;
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float p = pred->data[i];  // log probabilities
            float q;
            if (log_target_) {
                q = target->data[i];  // also log probabilities
                losses[i] = std::exp(q) * (q - p);
            } else {
                q = std::max(target->data[i], eps);  // probabilities
                losses[i] = q * (std::log(q) - p);
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else if (reduction_ == "batchmean") {
            auto sum = out->sum();
            sum->data[0] /= pred->shape[0];
            return sum;
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("KLDivLoss requires two inputs");
    }
};

/**
 * @brief Margin Ranking Loss
 */
class MarginRankingLoss : public Module {
private:
    std::string reduction_;
    float margin_;
    
public:
    MarginRankingLoss(float margin = 0.0f, const std::string& reduction = "mean")
        : reduction_(reduction), margin_(margin) {}
    
    TensorPtr forward(const TensorPtr& x1, const TensorPtr& x2, const TensorPtr& target) {
        std::vector<float> losses(x1->size());
        
        for (size_t i = 0; i < x1->size(); ++i) {
            float y = target->data[i];  // 1 or -1
            losses[i] = std::max(0.0f, -y * (x1->data[i] - x2->data[i]) + margin_);
        }
        
        auto out = std::make_shared<Tensor>(losses, x1->shape, x1->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
};

/**
 * @brief Hinge Embedding Loss
 */
class HingeEmbeddingLoss : public Module {
private:
    std::string reduction_;
    float margin_;
    
public:
    HingeEmbeddingLoss(float margin = 1.0f, const std::string& reduction = "mean")
        : reduction_(reduction), margin_(margin) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float y = target->data[i];
            if (y == 1) {
                losses[i] = pred->data[i];
            } else {
                losses[i] = std::max(0.0f, margin_ - pred->data[i]);
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("HingeEmbeddingLoss requires two inputs");
    }
};

/**
 * @brief Cosine Embedding Loss
 */
class CosineEmbeddingLoss : public Module {
private:
    std::string reduction_;
    float margin_;
    
public:
    CosineEmbeddingLoss(float margin = 0.0f, const std::string& reduction = "mean")
        : reduction_(reduction), margin_(margin) {}
    
    TensorPtr forward(const TensorPtr& x1, const TensorPtr& x2, const TensorPtr& target) {
        size_t N = x1->shape[0];
        size_t D = x1->size() / N;
        std::vector<float> losses(N);
        
        for (size_t n = 0; n < N; ++n) {
            // Compute cosine similarity
            float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float v1 = x1->data[n * D + d];
                float v2 = x2->data[n * D + d];
                dot += v1 * v2;
                norm1 += v1 * v1;
                norm2 += v2 * v2;
            }
            float cos_sim = dot / (std::sqrt(norm1 * norm2) + 1e-8f);
            
            float y = target->data[n];
            if (y == 1) {
                losses[n] = 1.0f - cos_sim;
            } else {
                losses[n] = std::max(0.0f, cos_sim - margin_);
            }
        }
        
        auto out = std::make_shared<Tensor>(losses, {N}, x1->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
};

/**
 * @brief Multi-label Margin Loss
 */
class MultiLabelMarginLoss : public Module {
private:
    std::string reduction_;
    
public:
    explicit MultiLabelMarginLoss(const std::string& reduction = "mean") 
        : reduction_(reduction) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        size_t N = pred->shape[0];
        size_t C = pred->shape[1];
        std::vector<float> losses(N, 0.0f);
        
        for (size_t n = 0; n < N; ++n) {
            for (size_t i = 0; i < C && target->data[n * C + i] >= 0; ++i) {
                size_t y = static_cast<size_t>(target->data[n * C + i]);
                for (size_t j = 0; j < C; ++j) {
                    // Check if j is not a target class
                    bool is_target = false;
                    for (size_t k = 0; k < C && target->data[n * C + k] >= 0; ++k) {
                        if (static_cast<size_t>(target->data[n * C + k]) == j) {
                            is_target = true;
                            break;
                        }
                    }
                    if (!is_target) {
                        losses[n] += std::max(0.0f, 1.0f - pred->data[n * C + y] + pred->data[n * C + j]);
                    }
                }
            }
            losses[n] /= C;
        }
        
        auto out = std::make_shared<Tensor>(losses, {N}, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("MultiLabelMarginLoss requires two inputs");
    }
};

/**
 * @brief Triplet Margin Loss
 */
class TripletMarginLoss : public Module {
private:
    std::string reduction_;
    float margin_;
    float p_;
    bool swap_;
    
public:
    TripletMarginLoss(float margin = 1.0f, float p = 2.0f, bool swap = false,
                      const std::string& reduction = "mean")
        : reduction_(reduction), margin_(margin), p_(p), swap_(swap) {}
    
    TensorPtr forward(const TensorPtr& anchor, const TensorPtr& positive, 
                      const TensorPtr& negative) {
        size_t N = anchor->shape[0];
        size_t D = anchor->size() / N;
        std::vector<float> losses(N);
        
        for (size_t n = 0; n < N; ++n) {
            // Compute distances
            float dist_ap = 0.0f, dist_an = 0.0f;
            for (size_t d = 0; d < D; ++d) {
                float diff_ap = anchor->data[n * D + d] - positive->data[n * D + d];
                float diff_an = anchor->data[n * D + d] - negative->data[n * D + d];
                dist_ap += std::pow(std::abs(diff_ap), p_);
                dist_an += std::pow(std::abs(diff_an), p_);
            }
            dist_ap = std::pow(dist_ap, 1.0f / p_);
            dist_an = std::pow(dist_an, 1.0f / p_);
            
            if (swap_) {
                // Also compute positive-negative distance
                float dist_pn = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    float diff = positive->data[n * D + d] - negative->data[n * D + d];
                    dist_pn += std::pow(std::abs(diff), p_);
                }
                dist_pn = std::pow(dist_pn, 1.0f / p_);
                dist_an = std::min(dist_an, dist_pn);
            }
            
            losses[n] = std::max(0.0f, dist_ap - dist_an + margin_);
        }
        
        auto out = std::make_shared<Tensor>(losses, {N}, anchor->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
};

/**
 * @brief Focal Loss for class imbalance
 */
class FocalLoss : public Module {
private:
    std::string reduction_;
    float alpha_;
    float gamma_;
    
public:
    FocalLoss(float alpha = 0.25f, float gamma = 2.0f, const std::string& reduction = "mean")
        : reduction_(reduction), alpha_(alpha), gamma_(gamma) {}
    
    TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) override {
        // For binary classification
        const float eps = 1e-7f;
        std::vector<float> losses(pred->size());
        
        for (size_t i = 0; i < pred->size(); ++i) {
            float p = std::clamp(pred->data[i], eps, 1.0f - eps);
            float t = target->data[i];
            
            float pt = t * p + (1.0f - t) * (1.0f - p);
            float alpha_t = t * alpha_ + (1.0f - t) * (1.0f - alpha_);
            
            losses[i] = -alpha_t * std::pow(1.0f - pt, gamma_) * std::log(pt);
        }
        
        auto out = std::make_shared<Tensor>(losses, pred->shape, pred->requires_grad);
        
        if (reduction_ == "none") {
            return out;
        } else if (reduction_ == "sum") {
            return out->sum();
        } else {
            return out->mean();
        }
    }
    
    TensorPtr forward(const TensorPtr& x) override {
        throw std::runtime_error("FocalLoss requires two inputs");
    }
};

} // namespace nn
} // namespace neurova
