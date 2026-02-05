// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file scheduler.hpp
 * @brief Learning rate schedulers
 * 
 * Neurova implementation of learning rate scheduling strategies.
 */

#pragma once

#include "optim.hpp"
#include <cmath>
#include <vector>
#include <functional>
#include <algorithm>

namespace neurova {
namespace nn {

/**
 * @brief Base class for learning rate schedulers
 */
class LRScheduler {
protected:
    Optimizer* optimizer_;
    float base_lr_;
    int last_epoch_ = -1;
    float last_lr_;
    bool verbose_;
    
public:
    LRScheduler(Optimizer* optimizer, int last_epoch = -1, bool verbose = false)
        : optimizer_(optimizer), base_lr_(optimizer->get_lr()), 
          last_epoch_(last_epoch), last_lr_(base_lr_), verbose_(verbose) {
        if (last_epoch == -1) {
            step();
        }
    }
    
    virtual ~LRScheduler() = default;
    
    /**
     * @brief Compute new learning rate
     */
    virtual float get_lr() = 0;
    
    /**
     * @brief Step the scheduler
     */
    virtual void step(int epoch = -1) {
        if (epoch == -1) {
            last_epoch_++;
        } else {
            last_epoch_ = epoch;
        }
        
        float lr = get_lr();
        optimizer_->set_lr(lr);
        last_lr_ = lr;
    }
    
    float get_last_lr() const { return last_lr_; }
};

/**
 * @brief Step LR - decay by gamma every step_size epochs
 */
class StepLR : public LRScheduler {
private:
    int step_size_;
    float gamma_;
    
public:
    StepLR(Optimizer* optimizer, int step_size, float gamma = 0.1f,
           int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), step_size_(step_size), gamma_(gamma) {}
    
    float get_lr() override {
        return base_lr_ * std::pow(gamma_, last_epoch_ / step_size_);
    }
};

/**
 * @brief MultiStep LR - decay at each milestone
 */
class MultiStepLR : public LRScheduler {
private:
    std::vector<int> milestones_;
    float gamma_;
    
public:
    MultiStepLR(Optimizer* optimizer, std::vector<int> milestones, float gamma = 0.1f,
                int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), milestones_(std::move(milestones)), gamma_(gamma) {
        std::sort(milestones_.begin(), milestones_.end());
    }
    
    float get_lr() override {
        int count = 0;
        for (int m : milestones_) {
            if (last_epoch_ >= m) count++;
        }
        return base_lr_ * std::pow(gamma_, count);
    }
};

/**
 * @brief Exponential LR - decay every epoch
 */
class ExponentialLR : public LRScheduler {
private:
    float gamma_;
    
public:
    ExponentialLR(Optimizer* optimizer, float gamma, int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), gamma_(gamma) {}
    
    float get_lr() override {
        return base_lr_ * std::pow(gamma_, last_epoch_);
    }
};

/**
 * @brief Linear LR - linear interpolation
 */
class LinearLR : public LRScheduler {
private:
    float start_factor_;
    float end_factor_;
    int total_iters_;
    
public:
    LinearLR(Optimizer* optimizer, float start_factor = 1.0f / 3.0f, 
             float end_factor = 1.0f, int total_iters = 5,
             int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), start_factor_(start_factor),
          end_factor_(end_factor), total_iters_(total_iters) {}
    
    float get_lr() override {
        if (last_epoch_ >= total_iters_) {
            return base_lr_ * end_factor_;
        }
        float t = static_cast<float>(last_epoch_) / total_iters_;
        return base_lr_ * (start_factor_ + (end_factor_ - start_factor_) * t);
    }
};

/**
 * @brief Cosine Annealing LR
 */
class CosineAnnealingLR : public LRScheduler {
private:
    int T_max_;
    float eta_min_;
    
public:
    CosineAnnealingLR(Optimizer* optimizer, int T_max, float eta_min = 0.0f,
                      int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), T_max_(T_max), eta_min_(eta_min) {}
    
    float get_lr() override {
        return eta_min_ + (base_lr_ - eta_min_) * 
               (1.0f + std::cos(3.14159265f * last_epoch_ / T_max_)) / 2.0f;
    }
};

/**
 * @brief Cosine Annealing Warm Restarts
 */
class CosineAnnealingWarmRestarts : public LRScheduler {
private:
    int T_0_;
    int T_mult_;
    float eta_min_;
    int T_cur_ = 0;
    int T_i_;
    
public:
    CosineAnnealingWarmRestarts(Optimizer* optimizer, int T_0, int T_mult = 1,
                                 float eta_min = 0.0f, int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), T_0_(T_0), T_mult_(T_mult), 
          eta_min_(eta_min), T_i_(T_0) {}
    
    float get_lr() override {
        return eta_min_ + (base_lr_ - eta_min_) * 
               (1.0f + std::cos(3.14159265f * T_cur_ / T_i_)) / 2.0f;
    }
    
    void step(int epoch = -1) override {
        if (epoch == -1) {
            T_cur_++;
            if (T_cur_ >= T_i_) {
                T_cur_ = 0;
                T_i_ = T_i_ * T_mult_;
            }
        } else {
            // Handle arbitrary epoch
            T_cur_ = epoch;
        }
        LRScheduler::step(epoch);
    }
};

/**
 * @brief Reduce LR on Plateau
 */
class ReduceLROnPlateau {
private:
    Optimizer* optimizer_;
    std::string mode_;
    float factor_;
    int patience_;
    float threshold_;
    int cooldown_;
    float min_lr_;
    bool verbose_;
    
    float best_;
    int num_bad_epochs_ = 0;
    int cooldown_counter_ = 0;
    
public:
    ReduceLROnPlateau(Optimizer* optimizer, const std::string& mode = "min",
                      float factor = 0.1f, int patience = 10, float threshold = 1e-4f,
                      int cooldown = 0, float min_lr = 0.0f, bool verbose = false)
        : optimizer_(optimizer), mode_(mode), factor_(factor), patience_(patience),
          threshold_(threshold), cooldown_(cooldown), min_lr_(min_lr), verbose_(verbose) {
        best_ = (mode == "min") ? std::numeric_limits<float>::infinity() 
                               : -std::numeric_limits<float>::infinity();
    }
    
    void step(float metric) {
        bool is_better = (mode_ == "min") ? (metric < best_ - threshold_)
                                          : (metric > best_ + threshold_);
        
        if (is_better) {
            best_ = metric;
            num_bad_epochs_ = 0;
        } else {
            num_bad_epochs_++;
        }
        
        if (cooldown_counter_ > 0) {
            cooldown_counter_--;
            num_bad_epochs_ = 0;
        }
        
        if (num_bad_epochs_ > patience_) {
            float new_lr = std::max(optimizer_->get_lr() * factor_, min_lr_);
            optimizer_->set_lr(new_lr);
            cooldown_counter_ = cooldown_;
            num_bad_epochs_ = 0;
        }
    }
};

/**
 * @brief One Cycle LR
 */
class OneCycleLR : public LRScheduler {
private:
    float max_lr_;
    int total_steps_;
    float pct_start_;
    float div_factor_;
    float final_div_factor_;
    int anneal_step_;
    
public:
    OneCycleLR(Optimizer* optimizer, float max_lr, int total_steps,
               float pct_start = 0.3f, float div_factor = 25.0f,
               float final_div_factor = 1e4f, int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), max_lr_(max_lr),
          total_steps_(total_steps), pct_start_(pct_start), div_factor_(div_factor),
          final_div_factor_(final_div_factor) {
        anneal_step_ = static_cast<int>(pct_start * total_steps);
        base_lr_ = max_lr / div_factor;
    }
    
    float get_lr() override {
        if (last_epoch_ < anneal_step_) {
            // Warmup phase
            float t = static_cast<float>(last_epoch_) / anneal_step_;
            return base_lr_ + (max_lr_ - base_lr_) * t;
        } else {
            // Annealing phase
            float t = static_cast<float>(last_epoch_ - anneal_step_) / (total_steps_ - anneal_step_);
            float end_lr = max_lr_ / final_div_factor_;
            return end_lr + (max_lr_ - end_lr) * (1.0f + std::cos(3.14159265f * t)) / 2.0f;
        }
    }
};

/**
 * @brief Cyclic LR
 */
class CyclicLR : public LRScheduler {
private:
    float base_lr_min_;
    float max_lr_;
    int step_size_up_;
    int step_size_down_;
    std::string mode_;
    float gamma_;
    int cycle_ = 0;
    
public:
    CyclicLR(Optimizer* optimizer, float base_lr, float max_lr,
             int step_size_up = 2000, int step_size_down = -1,
             const std::string& mode = "triangular", float gamma = 1.0f,
             int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), base_lr_min_(base_lr), max_lr_(max_lr),
          step_size_up_(step_size_up), step_size_down_(step_size_down == -1 ? step_size_up : step_size_down),
          mode_(mode), gamma_(gamma) {
        base_lr_ = base_lr;
    }
    
    float get_lr() override {
        int cycle_length = step_size_up_ + step_size_down_;
        int position = last_epoch_ % cycle_length;
        cycle_ = last_epoch_ / cycle_length;
        
        float scale;
        if (mode_ == "triangular") {
            scale = 1.0f;
        } else if (mode_ == "triangular2") {
            scale = 1.0f / std::pow(2.0f, cycle_);
        } else {  // exp_range
            scale = std::pow(gamma_, last_epoch_);
        }
        
        float x;
        if (position < step_size_up_) {
            x = static_cast<float>(position) / step_size_up_;
        } else {
            x = 1.0f - static_cast<float>(position - step_size_up_) / step_size_down_;
        }
        
        return base_lr_min_ + (max_lr_ - base_lr_min_) * std::max(0.0f, x) * scale;
    }
};

/**
 * @brief Warmup scheduler wrapper
 */
class WarmupScheduler : public LRScheduler {
private:
    std::unique_ptr<LRScheduler> main_scheduler_;
    int warmup_epochs_;
    float warmup_factor_;
    
public:
    WarmupScheduler(Optimizer* optimizer, std::unique_ptr<LRScheduler> main_scheduler,
                    int warmup_epochs = 5, float warmup_factor = 0.1f,
                    int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose),
          main_scheduler_(std::move(main_scheduler)),
          warmup_epochs_(warmup_epochs), warmup_factor_(warmup_factor) {}
    
    float get_lr() override {
        if (last_epoch_ < warmup_epochs_) {
            float t = static_cast<float>(last_epoch_ + 1) / warmup_epochs_;
            return base_lr_ * (warmup_factor_ + (1.0f - warmup_factor_) * t);
        }
        return main_scheduler_->get_lr();
    }
    
    void step(int epoch = -1) override {
        LRScheduler::step(epoch);
        if (last_epoch_ >= warmup_epochs_) {
            main_scheduler_->step(last_epoch_ - warmup_epochs_);
        }
    }
};

/**
 * @brief Lambda LR - custom lambda function
 */
class LambdaLR : public LRScheduler {
private:
    std::function<float(int)> lr_lambda_;
    
public:
    LambdaLR(Optimizer* optimizer, std::function<float(int)> lr_lambda,
             int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), lr_lambda_(std::move(lr_lambda)) {}
    
    float get_lr() override {
        return base_lr_ * lr_lambda_(last_epoch_);
    }
};

/**
 * @brief Polynomial LR decay
 */
class PolynomialLR : public LRScheduler {
private:
    int total_iters_;
    float power_;
    
public:
    PolynomialLR(Optimizer* optimizer, int total_iters, float power = 1.0f,
                 int last_epoch = -1, bool verbose = false)
        : LRScheduler(optimizer, last_epoch, verbose), total_iters_(total_iters), power_(power) {}
    
    float get_lr() override {
        if (last_epoch_ >= total_iters_) {
            return 0.0f;
        }
        return base_lr_ * std::pow(1.0f - static_cast<float>(last_epoch_) / total_iters_, power_);
    }
};

} // namespace nn
} // namespace neurova
