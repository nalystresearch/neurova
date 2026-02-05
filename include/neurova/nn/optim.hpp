// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file optim.hpp
 * @brief Optimization algorithms
 * 
 * Neurova implementation with SGD, Adam, RMSprop, etc.
 */

#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace neurova {
namespace nn {

/**
 * @brief Base class for all optimizers
 */
class Optimizer {
protected:
    std::vector<Parameter*> params_;
    float lr_;
    size_t t_ = 0;  // Time step for adaptive methods
    
public:
    Optimizer(std::vector<Parameter*> params, float lr = 0.01f)
        : params_(std::move(params)), lr_(lr) {}
    
    virtual ~Optimizer() = default;
    
    /**
     * @brief Perform a single optimization step
     */
    virtual void step() = 0;
    
    /**
     * @brief Zero out all parameter gradients
     */
    void zero_grad() {
        for (auto* param : params_) {
            param->zero_grad();
        }
    }
    
    /**
     * @brief Get/set learning rate
     */
    float get_lr() const { return lr_; }
    void set_lr(float lr) { lr_ = lr; }
};

/**
 * @brief Stochastic Gradient Descent optimizer
 */
class SGD : public Optimizer {
private:
    float momentum_;
    float weight_decay_;
    float dampening_;
    bool nesterov_;
    std::vector<std::vector<float>> velocity_;
    
public:
    SGD(std::vector<Parameter*> params, float lr = 0.01f, float momentum = 0.0f,
        float weight_decay = 0.0f, float dampening = 0.0f, bool nesterov = false)
        : Optimizer(std::move(params), lr), momentum_(momentum), 
          weight_decay_(weight_decay), dampening_(dampening), nesterov_(nesterov) {
        
        // Initialize velocity
        for (auto* param : params_) {
            velocity_.push_back(std::vector<float>(param->data->size(), 0.0f));
        }
    }
    
    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            auto& v = velocity_[i];
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                // Weight decay
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                // Momentum
                if (momentum_ != 0) {
                    v[j] = momentum_ * v[j] + (1.0f - dampening_) * g;
                    
                    if (nesterov_) {
                        g = g + momentum_ * v[j];
                    } else {
                        g = v[j];
                    }
                }
                
                // Update
                data[j] -= lr_ * g;
            }
        }
    }
};

/**
 * @brief Adam optimizer (Adaptive Moment Estimation)
 */
class Adam : public Optimizer {
protected:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    bool amsgrad_;
    std::vector<std::vector<float>> m_;  // First moment
    std::vector<std::vector<float>> v_;  // Second moment
    std::vector<std::vector<float>> v_max_;  // For AMSGrad
    
public:
    Adam(std::vector<Parameter*> params, float lr = 0.001f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
         float weight_decay = 0.0f, bool amsgrad = false)
        : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), weight_decay_(weight_decay), amsgrad_(amsgrad) {
        
        for (auto* param : params_) {
            m_.push_back(std::vector<float>(param->data->size(), 0.0f));
            v_.push_back(std::vector<float>(param->data->size(), 0.0f));
            if (amsgrad) {
                v_max_.push_back(std::vector<float>(param->data->size(), 0.0f));
            }
        }
    }
    
    void step() override {
        t_++;
        
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                // Weight decay (L2)
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                // Update biased first moment estimate
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;
                
                // Update biased second raw moment estimate
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;
                
                // Bias correction
                float m_hat = m_[i][j] / (1.0f - std::pow(beta1_, t_));
                float v_hat = v_[i][j] / (1.0f - std::pow(beta2_, t_));
                
                if (amsgrad_) {
                    v_max_[i][j] = std::max(v_max_[i][j], v_hat);
                    v_hat = v_max_[i][j];
                }
                
                // Update parameters
                data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

/**
 * @brief AdamW optimizer (Adam with decoupled weight decay)
 */
class AdamW : public Adam {
public:
    AdamW(std::vector<Parameter*> params, float lr = 0.001f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.01f, bool amsgrad = false)
        : Adam(std::move(params), lr, beta1, beta2, eps, 0.0f, amsgrad) {
        weight_decay_ = weight_decay;
    }
    
    void step() override {
        t_++;
        
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            // Decoupled weight decay
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] -= lr_ * weight_decay_ * data[j];
            }
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;
                
                float m_hat = m_[i][j] / (1.0f - std::pow(beta1_, t_));
                float v_hat = v_[i][j] / (1.0f - std::pow(beta2_, t_));
                
                if (amsgrad_) {
                    v_max_[i][j] = std::max(v_max_[i][j], v_hat);
                    v_hat = v_max_[i][j];
                }
                
                data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

/**
 * @brief RMSprop optimizer
 */
class RMSprop : public Optimizer {
private:
    float alpha_;
    float eps_;
    float weight_decay_;
    float momentum_;
    bool centered_;
    std::vector<std::vector<float>> square_avg_;
    std::vector<std::vector<float>> grad_avg_;
    std::vector<std::vector<float>> momentum_buffer_;
    
public:
    RMSprop(std::vector<Parameter*> params, float lr = 0.01f, float alpha = 0.99f,
            float eps = 1e-8f, float weight_decay = 0.0f, float momentum = 0.0f,
            bool centered = false)
        : Optimizer(std::move(params), lr), alpha_(alpha), eps_(eps),
          weight_decay_(weight_decay), momentum_(momentum), centered_(centered) {
        
        for (auto* param : params_) {
            square_avg_.push_back(std::vector<float>(param->data->size(), 0.0f));
            if (centered) {
                grad_avg_.push_back(std::vector<float>(param->data->size(), 0.0f));
            }
            if (momentum > 0) {
                momentum_buffer_.push_back(std::vector<float>(param->data->size(), 0.0f));
            }
        }
    }
    
    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                // Update running average of squared gradient
                square_avg_[i][j] = alpha_ * square_avg_[i][j] + (1.0f - alpha_) * g * g;
                
                float avg;
                if (centered_) {
                    grad_avg_[i][j] = alpha_ * grad_avg_[i][j] + (1.0f - alpha_) * g;
                    avg = std::sqrt(square_avg_[i][j] - grad_avg_[i][j] * grad_avg_[i][j] + eps_);
                } else {
                    avg = std::sqrt(square_avg_[i][j] + eps_);
                }
                
                if (momentum_ > 0) {
                    momentum_buffer_[i][j] = momentum_ * momentum_buffer_[i][j] + g / avg;
                    data[j] -= lr_ * momentum_buffer_[i][j];
                } else {
                    data[j] -= lr_ * g / avg;
                }
            }
        }
    }
};

/**
 * @brief Adagrad optimizer
 */
class Adagrad : public Optimizer {
private:
    float eps_;
    float weight_decay_;
    float lr_decay_;
    std::vector<std::vector<float>> sum_;
    
public:
    Adagrad(std::vector<Parameter*> params, float lr = 0.01f, float lr_decay = 0.0f,
            float weight_decay = 0.0f, float eps = 1e-10f)
        : Optimizer(std::move(params), lr), eps_(eps), weight_decay_(weight_decay),
          lr_decay_(lr_decay) {
        
        for (auto* param : params_) {
            sum_.push_back(std::vector<float>(param->data->size(), 0.0f));
        }
    }
    
    void step() override {
        t_++;
        float clr = lr_ / (1.0f + (t_ - 1) * lr_decay_);
        
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                sum_[i][j] += g * g;
                data[j] -= clr * g / (std::sqrt(sum_[i][j]) + eps_);
            }
        }
    }
};

/**
 * @brief Adadelta optimizer
 */
class Adadelta : public Optimizer {
private:
    float rho_;
    float eps_;
    float weight_decay_;
    std::vector<std::vector<float>> square_avg_;
    std::vector<std::vector<float>> acc_delta_;
    
public:
    Adadelta(std::vector<Parameter*> params, float lr = 1.0f, float rho = 0.9f,
             float eps = 1e-6f, float weight_decay = 0.0f)
        : Optimizer(std::move(params), lr), rho_(rho), eps_(eps), 
          weight_decay_(weight_decay) {
        
        for (auto* param : params_) {
            square_avg_.push_back(std::vector<float>(param->data->size(), 0.0f));
            acc_delta_.push_back(std::vector<float>(param->data->size(), 0.0f));
        }
    }
    
    void step() override {
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                square_avg_[i][j] = rho_ * square_avg_[i][j] + (1.0f - rho_) * g * g;
                
                float std = std::sqrt(acc_delta_[i][j] + eps_);
                float delta = std / std::sqrt(square_avg_[i][j] + eps_) * g;
                
                data[j] -= lr_ * delta;
                acc_delta_[i][j] = rho_ * acc_delta_[i][j] + (1.0f - rho_) * delta * delta;
            }
        }
    }
};

/**
 * @brief NAdam optimizer (Nesterov-accelerated Adam)
 */
class NAdam : public Optimizer {
private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    float momentum_decay_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    float mu_product_ = 1.0f;
    
public:
    NAdam(std::vector<Parameter*> params, float lr = 0.002f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.0f, float momentum_decay = 0.004f)
        : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), weight_decay_(weight_decay), momentum_decay_(momentum_decay) {
        
        for (auto* param : params_) {
            m_.push_back(std::vector<float>(param->data->size(), 0.0f));
            v_.push_back(std::vector<float>(param->data->size(), 0.0f));
        }
    }
    
    void step() override {
        t_++;
        
        float mu_t = beta1_ * (1.0f - 0.5f * std::pow(0.96f, t_ * momentum_decay_));
        float mu_t1 = beta1_ * (1.0f - 0.5f * std::pow(0.96f, (t_ + 1) * momentum_decay_));
        
        mu_product_ *= mu_t;
        float mu_product_next = mu_product_ * mu_t1;
        
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;
                
                float m_hat = mu_t1 * m_[i][j] / (1.0f - mu_product_next) +
                             (1.0f - mu_t) * g / (1.0f - mu_product_);
                float v_hat = v_[i][j] / (1.0f - std::pow(beta2_, t_));
                
                data[j] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

/**
 * @brief RAdam optimizer (Rectified Adam)
 */
class RAdam : public Optimizer {
private:
    float beta1_;
    float beta2_;
    float eps_;
    float weight_decay_;
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
    
public:
    RAdam(std::vector<Parameter*> params, float lr = 0.001f,
          float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
          float weight_decay = 0.0f)
        : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), weight_decay_(weight_decay) {
        
        for (auto* param : params_) {
            m_.push_back(std::vector<float>(param->data->size(), 0.0f));
            v_.push_back(std::vector<float>(param->data->size(), 0.0f));
        }
    }
    
    void step() override {
        t_++;
        
        float rho_inf = 2.0f / (1.0f - beta2_) - 1.0f;
        float rho_t = rho_inf - 2.0f * t_ * std::pow(beta2_, t_) / (1.0f - std::pow(beta2_, t_));
        
        for (size_t i = 0; i < params_.size(); ++i) {
            auto* param = params_[i];
            if (param->data->grad.empty()) continue;
            
            auto& data = param->data->data;
            auto& grad = param->data->grad;
            
            for (size_t j = 0; j < data.size(); ++j) {
                float g = grad[j];
                
                if (weight_decay_ != 0) {
                    g += weight_decay_ * data[j];
                }
                
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;
                
                float m_hat = m_[i][j] / (1.0f - std::pow(beta1_, t_));
                
                if (rho_t > 5.0f) {
                    // Variance is tractable
                    float v_hat = std::sqrt(v_[i][j] / (1.0f - std::pow(beta2_, t_)));
                    float r = std::sqrt(
                        ((rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf) /
                        ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t)
                    );
                    data[j] -= lr_ * r * m_hat / (v_hat + eps_);
                } else {
                    // Variance is intractable
                    data[j] -= lr_ * m_hat;
                }
            }
        }
    }
};

} // namespace nn
} // namespace neurova
