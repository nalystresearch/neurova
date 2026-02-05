// Copyright (c) 2026 @squid consultancy group (scg)
// All rights reserved.
// licensed under the apache license 2.0.

/**
 * @file recurrent.hpp
 * @brief Recurrent neural network layers
 * 
 * Neurova implementation of RNN, LSTM, GRU layers.
 */

#pragma once

#include "tensor.hpp"
#include "layers.hpp"
#include <cmath>
#include <random>

namespace neurova {
namespace nn {

/**
 * @brief Basic RNN cell
 */
class RNNCell : public Module {
private:
    int input_size_;
    int hidden_size_;
    bool bias_;
    std::string nonlinearity_;
    
    Parameter weight_ih_;  // input-hidden weights
    Parameter weight_hh_;  // hidden-hidden weights
    Parameter bias_ih_;
    Parameter bias_hh_;
    
public:
    RNNCell(int input_size, int hidden_size, bool bias = true,
            const std::string& nonlinearity = "tanh")
        : input_size_(input_size), hidden_size_(hidden_size), bias_(bias),
          nonlinearity_(nonlinearity) {
        
        // Initialize weights
        std::random_device rd;
        std::mt19937 gen(rd());
        float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        std::uniform_real_distribution<float> dist(-k, k);
        
        std::vector<float> w_ih(hidden_size * input_size);
        std::vector<float> w_hh(hidden_size * hidden_size);
        
        for (auto& v : w_ih) v = dist(gen);
        for (auto& v : w_hh) v = dist(gen);
        
        weight_ih_ = Parameter(Tensor(w_ih, {hidden_size, input_size}));
        weight_hh_ = Parameter(Tensor(w_hh, {hidden_size, hidden_size}));
        
        register_parameter("weight_ih", weight_ih_);
        register_parameter("weight_hh", weight_hh_);
        
        if (bias) {
            std::vector<float> b_ih(hidden_size), b_hh(hidden_size);
            for (auto& v : b_ih) v = dist(gen);
            for (auto& v : b_hh) v = dist(gen);
            
            bias_ih_ = Parameter(Tensor(b_ih, {hidden_size}));
            bias_hh_ = Parameter(Tensor(b_hh, {hidden_size}));
            
            register_parameter("bias_ih", bias_ih_);
            register_parameter("bias_hh", bias_hh_);
        }
    }
    
    Tensor forward(const Tensor& input, const Tensor& hidden) {
        // h' = activation(W_ih @ x + b_ih + W_hh @ h + b_hh)
        Tensor igates = weight_ih_.data().matmul(input);
        Tensor hgates = weight_hh_.data().matmul(hidden);
        
        if (bias_) {
            igates = igates + bias_ih_.data();
            hgates = hgates + bias_hh_.data();
        }
        
        Tensor h_new = igates + hgates;
        
        if (nonlinearity_ == "tanh") {
            h_new = h_new.tanh();
        } else {
            h_new = h_new.relu();
        }
        
        return h_new;
    }
    
    Tensor forward(const Tensor& input) override {
        // Initialize hidden state with zeros
        Tensor hidden = Tensor::zeros({hidden_size_});
        return forward(input, hidden);
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
};

/**
 * @brief LSTM cell
 */
class LSTMCell : public Module {
private:
    int input_size_;
    int hidden_size_;
    bool bias_;
    
    Parameter weight_ih_;  // 4 * hidden_size x input_size
    Parameter weight_hh_;  // 4 * hidden_size x hidden_size
    Parameter bias_ih_;
    Parameter bias_hh_;
    
    Tensor sigmoid(const Tensor& x) {
        std::vector<float> result(x.numel());
        for (size_t i = 0; i < x.numel(); ++i) {
            result[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
        }
        return Tensor(result, x.shape());
    }
    
public:
    LSTMCell(int input_size, int hidden_size, bool bias = true)
        : input_size_(input_size), hidden_size_(hidden_size), bias_(bias) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        std::uniform_real_distribution<float> dist(-k, k);
        
        // 4 gates: input, forget, cell, output
        std::vector<float> w_ih(4 * hidden_size * input_size);
        std::vector<float> w_hh(4 * hidden_size * hidden_size);
        
        for (auto& v : w_ih) v = dist(gen);
        for (auto& v : w_hh) v = dist(gen);
        
        weight_ih_ = Parameter(Tensor(w_ih, {4 * hidden_size, input_size}));
        weight_hh_ = Parameter(Tensor(w_hh, {4 * hidden_size, hidden_size}));
        
        register_parameter("weight_ih", weight_ih_);
        register_parameter("weight_hh", weight_hh_);
        
        if (bias) {
            std::vector<float> b_ih(4 * hidden_size), b_hh(4 * hidden_size);
            for (auto& v : b_ih) v = dist(gen);
            for (auto& v : b_hh) v = dist(gen);
            
            // Initialize forget gate bias to 1 for better gradient flow
            for (int i = hidden_size; i < 2 * hidden_size; ++i) {
                b_ih[i] = 1.0f;
            }
            
            bias_ih_ = Parameter(Tensor(b_ih, {4 * hidden_size}));
            bias_hh_ = Parameter(Tensor(b_hh, {4 * hidden_size}));
            
            register_parameter("bias_ih", bias_ih_);
            register_parameter("bias_hh", bias_hh_);
        }
    }
    
    std::pair<Tensor, Tensor> forward(const Tensor& input, const Tensor& hx, const Tensor& cx) {
        // Compute gates
        Tensor gates = weight_ih_.data().matmul(input) + weight_hh_.data().matmul(hx);
        
        if (bias_) {
            gates = gates + bias_ih_.data() + bias_hh_.data();
        }
        
        // Split into 4 gates
        std::vector<float> i_gate(hidden_size_), f_gate(hidden_size_), 
                           g_gate(hidden_size_), o_gate(hidden_size_);
        
        for (int j = 0; j < hidden_size_; ++j) {
            i_gate[j] = gates.data()[j];
            f_gate[j] = gates.data()[j + hidden_size_];
            g_gate[j] = gates.data()[j + 2 * hidden_size_];
            o_gate[j] = gates.data()[j + 3 * hidden_size_];
        }
        
        // Apply activations
        Tensor i_t = sigmoid(Tensor(i_gate, {hidden_size_}));
        Tensor f_t = sigmoid(Tensor(f_gate, {hidden_size_}));
        Tensor g_t = Tensor(g_gate, {hidden_size_}).tanh();
        Tensor o_t = sigmoid(Tensor(o_gate, {hidden_size_}));
        
        // Cell state update: c_t = f_t * c_{t-1} + i_t * g_t
        Tensor c_t = f_t * cx + i_t * g_t;
        
        // Hidden state: h_t = o_t * tanh(c_t)
        Tensor h_t = o_t * c_t.tanh();
        
        return {h_t, c_t};
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor hx = Tensor::zeros({hidden_size_});
        Tensor cx = Tensor::zeros({hidden_size_});
        auto [h, c] = forward(input, hx, cx);
        return h;
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
};

/**
 * @brief GRU cell
 */
class GRUCell : public Module {
private:
    int input_size_;
    int hidden_size_;
    bool bias_;
    
    Parameter weight_ih_;  // 3 * hidden_size x input_size
    Parameter weight_hh_;  // 3 * hidden_size x hidden_size
    Parameter bias_ih_;
    Parameter bias_hh_;
    
    Tensor sigmoid(const Tensor& x) {
        std::vector<float> result(x.numel());
        for (size_t i = 0; i < x.numel(); ++i) {
            result[i] = 1.0f / (1.0f + std::exp(-x.data()[i]));
        }
        return Tensor(result, x.shape());
    }
    
public:
    GRUCell(int input_size, int hidden_size, bool bias = true)
        : input_size_(input_size), hidden_size_(hidden_size), bias_(bias) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        float k = 1.0f / std::sqrt(static_cast<float>(hidden_size));
        std::uniform_real_distribution<float> dist(-k, k);
        
        // 3 gates: reset, update, new
        std::vector<float> w_ih(3 * hidden_size * input_size);
        std::vector<float> w_hh(3 * hidden_size * hidden_size);
        
        for (auto& v : w_ih) v = dist(gen);
        for (auto& v : w_hh) v = dist(gen);
        
        weight_ih_ = Parameter(Tensor(w_ih, {3 * hidden_size, input_size}));
        weight_hh_ = Parameter(Tensor(w_hh, {3 * hidden_size, hidden_size}));
        
        register_parameter("weight_ih", weight_ih_);
        register_parameter("weight_hh", weight_hh_);
        
        if (bias) {
            std::vector<float> b_ih(3 * hidden_size), b_hh(3 * hidden_size);
            for (auto& v : b_ih) v = dist(gen);
            for (auto& v : b_hh) v = dist(gen);
            
            bias_ih_ = Parameter(Tensor(b_ih, {3 * hidden_size}));
            bias_hh_ = Parameter(Tensor(b_hh, {3 * hidden_size}));
            
            register_parameter("bias_ih", bias_ih_);
            register_parameter("bias_hh", bias_hh_);
        }
    }
    
    Tensor forward(const Tensor& input, const Tensor& hx) {
        Tensor i_gates = weight_ih_.data().matmul(input);
        Tensor h_gates = weight_hh_.data().matmul(hx);
        
        if (bias_) {
            i_gates = i_gates + bias_ih_.data();
            h_gates = h_gates + bias_hh_.data();
        }
        
        // Split gates
        std::vector<float> r_i(hidden_size_), z_i(hidden_size_), n_i(hidden_size_);
        std::vector<float> r_h(hidden_size_), z_h(hidden_size_), n_h(hidden_size_);
        
        for (int j = 0; j < hidden_size_; ++j) {
            r_i[j] = i_gates.data()[j];
            z_i[j] = i_gates.data()[j + hidden_size_];
            n_i[j] = i_gates.data()[j + 2 * hidden_size_];
            
            r_h[j] = h_gates.data()[j];
            z_h[j] = h_gates.data()[j + hidden_size_];
            n_h[j] = h_gates.data()[j + 2 * hidden_size_];
        }
        
        // Reset gate
        Tensor r_t = sigmoid(Tensor(r_i, {hidden_size_}) + Tensor(r_h, {hidden_size_}));
        
        // Update gate
        Tensor z_t = sigmoid(Tensor(z_i, {hidden_size_}) + Tensor(z_h, {hidden_size_}));
        
        // New gate (with reset applied)
        Tensor n_t = (Tensor(n_i, {hidden_size_}) + r_t * Tensor(n_h, {hidden_size_})).tanh();
        
        // Hidden state: h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        Tensor one_minus_z = Tensor::ones({hidden_size_}) + (z_t * -1.0f);
        Tensor h_t = one_minus_z * n_t + z_t * hx;
        
        return h_t;
    }
    
    Tensor forward(const Tensor& input) override {
        Tensor hx = Tensor::zeros({hidden_size_});
        return forward(input, hx);
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
};

/**
 * @brief Multi-layer RNN
 */
class RNN : public Module {
private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool bias_;
    bool batch_first_;
    float dropout_;
    bool bidirectional_;
    std::string nonlinearity_;
    
    std::vector<std::unique_ptr<RNNCell>> cells_;
    std::vector<std::unique_ptr<RNNCell>> reverse_cells_;
    
public:
    RNN(int input_size, int hidden_size, int num_layers = 1, bool bias = true,
        bool batch_first = false, float dropout = 0.0f, bool bidirectional = false,
        const std::string& nonlinearity = "tanh")
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          bias_(bias), batch_first_(batch_first), dropout_(dropout), 
          bidirectional_(bidirectional), nonlinearity_(nonlinearity) {
        
        int directions = bidirectional ? 2 : 1;
        
        for (int layer = 0; layer < num_layers; ++layer) {
            int layer_input_size = (layer == 0) ? input_size : hidden_size * directions;
            cells_.push_back(std::make_unique<RNNCell>(layer_input_size, hidden_size, bias, nonlinearity));
            
            if (bidirectional) {
                reverse_cells_.push_back(std::make_unique<RNNCell>(layer_input_size, hidden_size, bias, nonlinearity));
            }
        }
    }
    
    std::pair<Tensor, Tensor> forward(const Tensor& input, const Tensor& h0) {
        auto shape = input.shape();
        int seq_len = batch_first_ ? shape[1] : shape[0];
        int batch_size = batch_first_ ? shape[0] : shape[1];
        int directions = bidirectional_ ? 2 : 1;
        
        // Process sequence through layers
        Tensor output = input;
        Tensor h_n = h0;
        
        // Simplified: just process forward direction, one timestep at a time
        std::vector<float> outputs;
        Tensor h = h0;
        
        for (int t = 0; t < seq_len; ++t) {
            // Extract input at time t
            std::vector<float> x_t(input_size_);
            for (int i = 0; i < input_size_; ++i) {
                x_t[i] = input.data()[t * input_size_ + i];
            }
            
            Tensor input_t(x_t, {input_size_});
            h = cells_[0]->forward(input_t, h);
            
            for (float v : h.data()) {
                outputs.push_back(v);
            }
        }
        
        return {
            Tensor(outputs, {seq_len, hidden_size_}),
            h
        };
    }
    
    Tensor forward(const Tensor& input) override {
        int directions = bidirectional_ ? 2 : 1;
        Tensor h0 = Tensor::zeros({num_layers_ * directions, hidden_size_});
        auto [output, h_n] = forward(input, h0);
        return output;
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }
    bool bidirectional() const { return bidirectional_; }
};

/**
 * @brief Multi-layer LSTM
 */
class LSTM : public Module {
private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool bias_;
    bool batch_first_;
    float dropout_;
    bool bidirectional_;
    
    std::vector<std::unique_ptr<LSTMCell>> cells_;
    
public:
    LSTM(int input_size, int hidden_size, int num_layers = 1, bool bias = true,
         bool batch_first = false, float dropout = 0.0f, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          bias_(bias), batch_first_(batch_first), dropout_(dropout), bidirectional_(bidirectional) {
        
        int directions = bidirectional ? 2 : 1;
        
        for (int layer = 0; layer < num_layers; ++layer) {
            int layer_input_size = (layer == 0) ? input_size : hidden_size * directions;
            cells_.push_back(std::make_unique<LSTMCell>(layer_input_size, hidden_size, bias));
        }
    }
    
    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor& input, const Tensor& h0, const Tensor& c0) {
        auto shape = input.shape();
        int seq_len = batch_first_ ? shape[1] : shape[0];
        
        std::vector<float> outputs;
        Tensor h = h0;
        Tensor c = c0;
        
        for (int t = 0; t < seq_len; ++t) {
            std::vector<float> x_t(input_size_);
            for (int i = 0; i < input_size_; ++i) {
                x_t[i] = input.data()[t * input_size_ + i];
            }
            
            Tensor input_t(x_t, {input_size_});
            auto [h_new, c_new] = cells_[0]->forward(input_t, h, c);
            h = h_new;
            c = c_new;
            
            for (float v : h.data()) {
                outputs.push_back(v);
            }
        }
        
        return {
            Tensor(outputs, {seq_len, hidden_size_}),
            h,
            c
        };
    }
    
    Tensor forward(const Tensor& input) override {
        int directions = bidirectional_ ? 2 : 1;
        Tensor h0 = Tensor::zeros({num_layers_ * directions, hidden_size_});
        Tensor c0 = Tensor::zeros({num_layers_ * directions, hidden_size_});
        auto [output, h_n, c_n] = forward(input, h0, c0);
        return output;
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }
    bool bidirectional() const { return bidirectional_; }
};

/**
 * @brief Multi-layer GRU
 */
class GRU : public Module {
private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    bool bias_;
    bool batch_first_;
    float dropout_;
    bool bidirectional_;
    
    std::vector<std::unique_ptr<GRUCell>> cells_;
    
public:
    GRU(int input_size, int hidden_size, int num_layers = 1, bool bias = true,
        bool batch_first = false, float dropout = 0.0f, bool bidirectional = false)
        : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers),
          bias_(bias), batch_first_(batch_first), dropout_(dropout), bidirectional_(bidirectional) {
        
        int directions = bidirectional ? 2 : 1;
        
        for (int layer = 0; layer < num_layers; ++layer) {
            int layer_input_size = (layer == 0) ? input_size : hidden_size * directions;
            cells_.push_back(std::make_unique<GRUCell>(layer_input_size, hidden_size, bias));
        }
    }
    
    std::pair<Tensor, Tensor> forward(const Tensor& input, const Tensor& h0) {
        auto shape = input.shape();
        int seq_len = batch_first_ ? shape[1] : shape[0];
        
        std::vector<float> outputs;
        Tensor h = h0;
        
        for (int t = 0; t < seq_len; ++t) {
            std::vector<float> x_t(input_size_);
            for (int i = 0; i < input_size_; ++i) {
                x_t[i] = input.data()[t * input_size_ + i];
            }
            
            Tensor input_t(x_t, {input_size_});
            h = cells_[0]->forward(input_t, h);
            
            for (float v : h.data()) {
                outputs.push_back(v);
            }
        }
        
        return {
            Tensor(outputs, {seq_len, hidden_size_}),
            h
        };
    }
    
    Tensor forward(const Tensor& input) override {
        int directions = bidirectional_ ? 2 : 1;
        Tensor h0 = Tensor::zeros({num_layers_ * directions, hidden_size_});
        auto [output, h_n] = forward(input, h0);
        return output;
    }
    
    int input_size() const { return input_size_; }
    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }
    bool bidirectional() const { return bidirectional_; }
};

} // namespace nn
} // namespace neurova
