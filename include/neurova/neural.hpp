// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/neural.hpp - Neural Network Core Components
 * 
 * This header provides all neural network building blocks:
 * - Layers (Dense, Conv2D, BatchNorm, Dropout, etc.)
 * - Activations (ReLU, Sigmoid, Tanh, Softmax, etc.)
 * - Loss functions (MSE, CrossEntropy, etc.)
 * - Optimizers (SGD, Adam, RMSprop, etc.)
 * - Automatic differentiation
 */

#ifndef NEUROVA_NEURAL_HPP
#define NEUROVA_NEURAL_HPP

#include "core.hpp"
#include <string>
#include <unordered_map>

namespace neurova {
namespace nn {

// ============================================================================
// Forward declarations
// ============================================================================

class Module;
class Layer;
class Optimizer;
class Loss;

// ============================================================================
// Gradient tracking
// ============================================================================

struct GradTensor {
    Tensor data;
    Tensor grad;
    bool requires_grad;
    std::string name;
    
    GradTensor() : requires_grad(false) {}
    GradTensor(const Tensor& t, bool grad = false) : data(t), requires_grad(grad) {}
    
    void zero_grad() {
        if (requires_grad) {
            grad = Tensor::zeros(data.shape(), data.dtype());
        }
    }
};

// ============================================================================
// Activation Functions
// ============================================================================

namespace activation {

Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& grad, const Tensor& x);

Tensor leaky_relu(const Tensor& x, double alpha = 0.01);
Tensor leaky_relu_backward(const Tensor& grad, const Tensor& x, double alpha = 0.01);

Tensor elu(const Tensor& x, double alpha = 1.0);
Tensor elu_backward(const Tensor& grad, const Tensor& x, double alpha = 1.0);

Tensor selu(const Tensor& x);
Tensor selu_backward(const Tensor& grad, const Tensor& x);

Tensor gelu(const Tensor& x);
Tensor gelu_backward(const Tensor& grad, const Tensor& x);

Tensor swish(const Tensor& x);
Tensor swish_backward(const Tensor& grad, const Tensor& x);

Tensor mish(const Tensor& x);
Tensor mish_backward(const Tensor& grad, const Tensor& x);

Tensor sigmoid(const Tensor& x);
Tensor sigmoid_backward(const Tensor& grad, const Tensor& output);

Tensor tanh(const Tensor& x);
Tensor tanh_backward(const Tensor& grad, const Tensor& output);

Tensor softmax(const Tensor& x, int axis = -1);
Tensor softmax_backward(const Tensor& grad, const Tensor& output);

Tensor log_softmax(const Tensor& x, int axis = -1);
Tensor log_softmax_backward(const Tensor& grad, const Tensor& output);

Tensor softplus(const Tensor& x);
Tensor softplus_backward(const Tensor& grad, const Tensor& x);

Tensor softsign(const Tensor& x);
Tensor softsign_backward(const Tensor& grad, const Tensor& x);

Tensor hardswish(const Tensor& x);
Tensor hardswish_backward(const Tensor& grad, const Tensor& x);

Tensor hardsigmoid(const Tensor& x);
Tensor hardsigmoid_backward(const Tensor& grad, const Tensor& x);

} // namespace activation

// ============================================================================
// Loss Functions
// ============================================================================

namespace loss {

// mean squared error
double mse(const Tensor& pred, const Tensor& target);
Tensor mse_backward(const Tensor& pred, const Tensor& target);

// mean absolute error
double mae(const Tensor& pred, const Tensor& target);
Tensor mae_backward(const Tensor& pred, const Tensor& target);

// huber loss
double huber(const Tensor& pred, const Tensor& target, double delta = 1.0);
Tensor huber_backward(const Tensor& pred, const Tensor& target, double delta = 1.0);

// binary cross entropy
double binary_cross_entropy(const Tensor& pred, const Tensor& target);
Tensor binary_cross_entropy_backward(const Tensor& pred, const Tensor& target);

// cross entropy (with logits)
double cross_entropy(const Tensor& pred, const Tensor& target);
Tensor cross_entropy_backward(const Tensor& pred, const Tensor& target);

// negative log likelihood
double nll_loss(const Tensor& pred, const Tensor& target);
Tensor nll_loss_backward(const Tensor& pred, const Tensor& target);

// focal loss
double focal_loss(const Tensor& pred, const Tensor& target, double gamma = 2.0, double alpha = 0.25);
Tensor focal_loss_backward(const Tensor& pred, const Tensor& target, double gamma = 2.0, double alpha = 0.25);

// hinge loss
double hinge_loss(const Tensor& pred, const Tensor& target);
Tensor hinge_loss_backward(const Tensor& pred, const Tensor& target);

// cosine embedding loss
double cosine_embedding_loss(const Tensor& x1, const Tensor& x2, const Tensor& target, double margin = 0.0);

// triplet margin loss
double triplet_margin_loss(const Tensor& anchor, const Tensor& positive, const Tensor& negative, double margin = 1.0);

// contrastive loss
double contrastive_loss(const Tensor& x1, const Tensor& x2, const Tensor& target, double margin = 1.0);

} // namespace loss

// ============================================================================
// Weight Initialization
// ============================================================================

namespace init {

void zeros(Tensor& t);
void ones(Tensor& t);
void constant(Tensor& t, double val);
void uniform(Tensor& t, double low = 0.0, double high = 1.0);
void normal(Tensor& t, double mean = 0.0, double std = 1.0);
void xavier_uniform(Tensor& t, double gain = 1.0);
void xavier_normal(Tensor& t, double gain = 1.0);
void kaiming_uniform(Tensor& t, double a = 0.0);
void kaiming_normal(Tensor& t, double a = 0.0);
void orthogonal(Tensor& t, double gain = 1.0);
void sparse(Tensor& t, double sparsity, double std = 0.01);

} // namespace init

// ============================================================================
// Base Layer Class
// ============================================================================

class Layer {
public:
    Layer() : training_(true) {}
    virtual ~Layer() = default;
    
    virtual Tensor forward(const Tensor& x) = 0;
    virtual Tensor backward(const Tensor& grad) = 0;
    
    virtual std::vector<GradTensor*> parameters() { return {}; }
    virtual void train(bool mode = true) { training_ = mode; }
    virtual void eval() { train(false); }
    bool is_training() const { return training_; }
    
    virtual std::string name() const = 0;
    virtual size_t num_parameters() const { return 0; }
    
protected:
    bool training_;
};

// ============================================================================
// Linear / Dense Layer
// ============================================================================

class Linear : public Layer {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Linear"; }
    size_t num_parameters() const override;
    
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    
    GradTensor weight;
    GradTensor bias;

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Tensor input_cache_;
};

// ============================================================================
// Convolutional Layers
// ============================================================================

class Conv2D : public Layer {
public:
    Conv2D(size_t in_channels, size_t out_channels, 
           size_t kernel_size, size_t stride = 1, 
           size_t padding = 0, size_t dilation = 1,
           size_t groups = 1, bool bias = true);
    
    Conv2D(size_t in_channels, size_t out_channels,
           std::pair<size_t, size_t> kernel_size,
           std::pair<size_t, size_t> stride = {1, 1},
           std::pair<size_t, size_t> padding = {0, 0},
           std::pair<size_t, size_t> dilation = {1, 1},
           size_t groups = 1, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Conv2D"; }
    size_t num_parameters() const override;
    
    GradTensor weight;
    GradTensor bias;

private:
    size_t in_channels_, out_channels_;
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    std::pair<size_t, size_t> dilation_;
    size_t groups_;
    bool use_bias_;
    Tensor input_cache_;
};

class ConvTranspose2D : public Layer {
public:
    ConvTranspose2D(size_t in_channels, size_t out_channels,
                    size_t kernel_size, size_t stride = 1,
                    size_t padding = 0, size_t output_padding = 0,
                    size_t groups = 1, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ConvTranspose2D"; }
    
    GradTensor weight;
    GradTensor bias;

private:
    size_t in_channels_, out_channels_;
    size_t kernel_size_, stride_, padding_, output_padding_, groups_;
    bool use_bias_;
    Tensor input_cache_;
};

// Depthwise separable convolution
class DepthwiseConv2D : public Layer {
public:
    DepthwiseConv2D(size_t channels, size_t kernel_size, 
                    size_t stride = 1, size_t padding = 0, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "DepthwiseConv2D"; }
    
    GradTensor weight;
    GradTensor bias;

private:
    size_t channels_, kernel_size_, stride_, padding_;
    bool use_bias_;
    Tensor input_cache_;
};

// ============================================================================
// Pooling Layers
// ============================================================================

class MaxPool2D : public Layer {
public:
    MaxPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    MaxPool2D(std::pair<size_t, size_t> kernel_size,
              std::pair<size_t, size_t> stride = {0, 0},
              std::pair<size_t, size_t> padding = {0, 0});
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "MaxPool2D"; }

private:
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    Tensor indices_cache_;
    Shape input_shape_;
};

class AvgPool2D : public Layer {
public:
    AvgPool2D(size_t kernel_size, size_t stride = 0, size_t padding = 0);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "AvgPool2D"; }

private:
    std::pair<size_t, size_t> kernel_size_;
    std::pair<size_t, size_t> stride_;
    std::pair<size_t, size_t> padding_;
    Shape input_shape_;
};

class AdaptiveAvgPool2D : public Layer {
public:
    AdaptiveAvgPool2D(size_t output_height, size_t output_width);
    AdaptiveAvgPool2D(size_t output_size);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "AdaptiveAvgPool2D"; }

private:
    size_t output_height_, output_width_;
    Shape input_shape_;
};

class AdaptiveMaxPool2D : public Layer {
public:
    AdaptiveMaxPool2D(size_t output_height, size_t output_width);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "AdaptiveMaxPool2D"; }

private:
    size_t output_height_, output_width_;
    Tensor indices_cache_;
    Shape input_shape_;
};

class GlobalAvgPool2D : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "GlobalAvgPool2D"; }

private:
    Shape input_shape_;
};

class GlobalMaxPool2D : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "GlobalMaxPool2D"; }

private:
    Tensor indices_cache_;
    Shape input_shape_;
};

// ============================================================================
// Normalization Layers
// ============================================================================

class BatchNorm1D : public Layer {
public:
    BatchNorm1D(size_t num_features, double eps = 1e-5, double momentum = 0.1, bool affine = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "BatchNorm1D"; }
    
    GradTensor gamma;
    GradTensor beta;
    Tensor running_mean;
    Tensor running_var;

private:
    size_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    Tensor x_norm_cache_;
    Tensor std_cache_;
    Tensor mean_cache_;
};

class BatchNorm2D : public Layer {
public:
    BatchNorm2D(size_t num_features, double eps = 1e-5, double momentum = 0.1, bool affine = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "BatchNorm2D"; }
    
    GradTensor gamma;
    GradTensor beta;
    Tensor running_mean;
    Tensor running_var;

private:
    size_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    Tensor x_norm_cache_;
    Tensor std_cache_;
    Tensor mean_cache_;
};

class LayerNorm : public Layer {
public:
    LayerNorm(const std::vector<size_t>& normalized_shape, double eps = 1e-5);
    LayerNorm(size_t normalized_shape, double eps = 1e-5);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "LayerNorm"; }
    
    GradTensor gamma;
    GradTensor beta;

private:
    std::vector<size_t> normalized_shape_;
    double eps_;
    Tensor x_norm_cache_;
    Tensor std_cache_;
    Tensor mean_cache_;
};

class GroupNorm : public Layer {
public:
    GroupNorm(size_t num_groups, size_t num_channels, double eps = 1e-5, bool affine = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "GroupNorm"; }
    
    GradTensor gamma;
    GradTensor beta;

private:
    size_t num_groups_;
    size_t num_channels_;
    double eps_;
    bool affine_;
    Tensor x_norm_cache_;
    Tensor std_cache_;
};

class InstanceNorm2D : public Layer {
public:
    InstanceNorm2D(size_t num_features, double eps = 1e-5, bool affine = false);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "InstanceNorm2D"; }
    
    GradTensor gamma;
    GradTensor beta;

private:
    size_t num_features_;
    double eps_;
    bool affine_;
};

// ============================================================================
// Regularization Layers
// ============================================================================

class Dropout : public Layer {
public:
    Dropout(double p = 0.5);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "Dropout"; }

private:
    double p_;
    Tensor mask_;
};

class Dropout2D : public Layer {
public:
    Dropout2D(double p = 0.5);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "Dropout2D"; }

private:
    double p_;
    Tensor mask_;
};

class AlphaDropout : public Layer {
public:
    AlphaDropout(double p = 0.5);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "AlphaDropout"; }

private:
    double p_;
    Tensor mask_;
};

// ============================================================================
// Activation Layers (as layers)
// ============================================================================

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "ReLU"; }
private:
    Tensor input_cache_;
};

class LeakyReLU : public Layer {
public:
    LeakyReLU(double alpha = 0.01);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "LeakyReLU"; }
private:
    double alpha_;
    Tensor input_cache_;
};

class ELU : public Layer {
public:
    ELU(double alpha = 1.0);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "ELU"; }
private:
    double alpha_;
    Tensor input_cache_;
};

class SELU : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "SELU"; }
private:
    Tensor input_cache_;
};

class GELU : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "GELU"; }
private:
    Tensor input_cache_;
};

class Swish : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Swish"; }
private:
    Tensor input_cache_;
};

class Mish : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Mish"; }
private:
    Tensor input_cache_;
};

class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Sigmoid"; }
private:
    Tensor output_cache_;
};

class Tanh : public Layer {
public:
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Tanh"; }
private:
    Tensor output_cache_;
};

class Softmax : public Layer {
public:
    Softmax(int dim = -1);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Softmax"; }
private:
    int dim_;
    Tensor output_cache_;
};

class LogSoftmax : public Layer {
public:
    LogSoftmax(int dim = -1);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "LogSoftmax"; }
private:
    int dim_;
    Tensor output_cache_;
};

// ============================================================================
// Reshape Layers
// ============================================================================

class Flatten : public Layer {
public:
    Flatten(int start_dim = 1, int end_dim = -1);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Flatten"; }
private:
    int start_dim_, end_dim_;
    Shape input_shape_;
};

class Reshape : public Layer {
public:
    Reshape(const Shape& shape);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Reshape"; }
private:
    Shape target_shape_;
    Shape input_shape_;
};

class Unsqueeze : public Layer {
public:
    Unsqueeze(int dim);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Unsqueeze"; }
private:
    int dim_;
};

class Squeeze : public Layer {
public:
    Squeeze(int dim = -1);
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "Squeeze"; }
private:
    int dim_;
    Shape input_shape_;
};

// ============================================================================
// Embedding Layer
// ============================================================================

class Embedding : public Layer {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim, int padding_idx = -1);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Embedding"; }
    size_t num_parameters() const override;
    
    GradTensor weight;

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
    int padding_idx_;
    Tensor indices_cache_;
};

// ============================================================================
// Recurrent Layers
// ============================================================================

class RNNCell : public Layer {
public:
    RNNCell(size_t input_size, size_t hidden_size, bool bias = true, 
            const std::string& nonlinearity = "tanh");
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, const Tensor& h);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "RNNCell"; }
    
    GradTensor weight_ih;
    GradTensor weight_hh;
    GradTensor bias_ih;
    GradTensor bias_hh;
    
    Tensor hidden_state;

private:
    size_t input_size_, hidden_size_;
    bool use_bias_;
    std::string nonlinearity_;
};

class LSTMCell : public Layer {
public:
    LSTMCell(size_t input_size, size_t hidden_size, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& h, const Tensor& c);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "LSTMCell"; }
    
    GradTensor weight_ih;
    GradTensor weight_hh;
    GradTensor bias_ih;
    GradTensor bias_hh;
    
    Tensor hidden_state;
    Tensor cell_state;

private:
    size_t input_size_, hidden_size_;
    bool use_bias_;
};

class GRUCell : public Layer {
public:
    GRUCell(size_t input_size, size_t hidden_size, bool bias = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, const Tensor& h);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "GRUCell"; }
    
    GradTensor weight_ih;
    GradTensor weight_hh;
    GradTensor bias_ih;
    GradTensor bias_hh;
    
    Tensor hidden_state;

private:
    size_t input_size_, hidden_size_;
    bool use_bias_;
};

class RNN : public Layer {
public:
    RNN(size_t input_size, size_t hidden_size, size_t num_layers = 1,
        bool bias = true, bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false, const std::string& nonlinearity = "tanh");
    
    Tensor forward(const Tensor& x) override;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& h0);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "RNN"; }

private:
    size_t input_size_, hidden_size_, num_layers_;
    bool bias_, batch_first_, bidirectional_;
    double dropout_;
    std::string nonlinearity_;
    std::vector<std::unique_ptr<RNNCell>> cells_;
};

class LSTM : public Layer {
public:
    LSTM(size_t input_size, size_t hidden_size, size_t num_layers = 1,
         bool bias = true, bool batch_first = false, double dropout = 0.0,
         bool bidirectional = false);
    
    Tensor forward(const Tensor& x) override;
    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor& x, const Tensor& h0, const Tensor& c0);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "LSTM"; }

private:
    size_t input_size_, hidden_size_, num_layers_;
    bool bias_, batch_first_, bidirectional_;
    double dropout_;
    std::vector<std::unique_ptr<LSTMCell>> cells_;
};

class GRU : public Layer {
public:
    GRU(size_t input_size, size_t hidden_size, size_t num_layers = 1,
        bool bias = true, bool batch_first = false, double dropout = 0.0,
        bool bidirectional = false);
    
    Tensor forward(const Tensor& x) override;
    std::pair<Tensor, Tensor> forward(const Tensor& x, const Tensor& h0);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "GRU"; }

private:
    size_t input_size_, hidden_size_, num_layers_;
    bool bias_, batch_first_, bidirectional_;
    double dropout_;
    std::vector<std::unique_ptr<GRUCell>> cells_;
};

// ============================================================================
// Attention Layers
// ============================================================================

class MultiHeadAttention : public Layer {
public:
    MultiHeadAttention(size_t embed_dim, size_t num_heads, 
                       double dropout = 0.0, bool bias = true,
                       bool add_bias_kv = false, bool add_zero_attn = false);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* key_padding_mask = nullptr,
                   const Tensor* attn_mask = nullptr);
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "MultiHeadAttention"; }
    
    GradTensor in_proj_weight;
    GradTensor in_proj_bias;
    GradTensor out_proj_weight;
    GradTensor out_proj_bias;

private:
    size_t embed_dim_, num_heads_, head_dim_;
    double dropout_;
    bool bias_;
    Tensor attn_weights_cache_;
};

class ScaledDotProductAttention : public Layer {
public:
    ScaledDotProductAttention(double dropout = 0.0);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& query, const Tensor& key, const Tensor& value,
                   const Tensor* mask = nullptr);
    Tensor backward(const Tensor& grad) override;
    
    std::string name() const override { return "ScaledDotProductAttention"; }

private:
    double dropout_;
    Tensor attn_weights_cache_;
    Tensor q_cache_, k_cache_, v_cache_;
};

// ============================================================================
// Sequential Container
// ============================================================================

class Sequential : public Layer {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Layer>> layers);
    
    void add(std::shared_ptr<Layer> layer);
    
    template<typename T, typename... Args>
    void add(Args&&... args) {
        layers_.push_back(std::make_shared<T>(std::forward<Args>(args)...));
    }
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    
    std::vector<GradTensor*> parameters() override;
    void train(bool mode = true) override;
    std::string name() const override { return "Sequential"; }
    size_t num_parameters() const override;
    
    size_t size() const { return layers_.size(); }
    std::shared_ptr<Layer> operator[](size_t idx) { return layers_[idx]; }

private:
    std::vector<std::shared_ptr<Layer>> layers_;
};

// ============================================================================
// Optimizers
// ============================================================================

class Optimizer {
public:
    Optimizer(std::vector<GradTensor*> params, double lr = 0.001);
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual void zero_grad();
    
    void set_lr(double lr) { lr_ = lr; }
    double get_lr() const { return lr_; }

protected:
    std::vector<GradTensor*> params_;
    double lr_;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<GradTensor*> params, double lr = 0.01, 
        double momentum = 0.0, double weight_decay = 0.0,
        double dampening = 0.0, bool nesterov = false);
    
    void step() override;

private:
    double momentum_;
    double weight_decay_;
    double dampening_;
    bool nesterov_;
    std::vector<Tensor> velocities_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<GradTensor*> params, double lr = 0.001,
         double beta1 = 0.9, double beta2 = 0.999,
         double eps = 1e-8, double weight_decay = 0.0,
         bool amsgrad = false);
    
    void step() override;

private:
    double beta1_, beta2_, eps_, weight_decay_;
    bool amsgrad_;
    size_t t_;
    std::vector<Tensor> m_, v_, v_max_;
};

class AdamW : public Optimizer {
public:
    AdamW(std::vector<GradTensor*> params, double lr = 0.001,
          double beta1 = 0.9, double beta2 = 0.999,
          double eps = 1e-8, double weight_decay = 0.01);
    
    void step() override;

private:
    double beta1_, beta2_, eps_, weight_decay_;
    size_t t_;
    std::vector<Tensor> m_, v_;
};

class RMSprop : public Optimizer {
public:
    RMSprop(std::vector<GradTensor*> params, double lr = 0.01,
            double alpha = 0.99, double eps = 1e-8,
            double weight_decay = 0.0, double momentum = 0.0,
            bool centered = false);
    
    void step() override;

private:
    double alpha_, eps_, weight_decay_, momentum_;
    bool centered_;
    std::vector<Tensor> square_avg_, grad_avg_, momentum_buffer_;
};

class Adagrad : public Optimizer {
public:
    Adagrad(std::vector<GradTensor*> params, double lr = 0.01,
            double lr_decay = 0.0, double eps = 1e-10,
            double weight_decay = 0.0);
    
    void step() override;

private:
    double lr_decay_, eps_, weight_decay_;
    size_t step_count_;
    std::vector<Tensor> sum_;
};

class Adadelta : public Optimizer {
public:
    Adadelta(std::vector<GradTensor*> params, double lr = 1.0,
             double rho = 0.9, double eps = 1e-6,
             double weight_decay = 0.0);
    
    void step() override;

private:
    double rho_, eps_, weight_decay_;
    std::vector<Tensor> square_avg_, acc_delta_;
};

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

class LRScheduler {
public:
    LRScheduler(Optimizer& optimizer) : optimizer_(optimizer), last_epoch_(-1) {}
    virtual ~LRScheduler() = default;
    
    virtual void step();
    virtual double get_lr() const = 0;

protected:
    Optimizer& optimizer_;
    int last_epoch_;
};

class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int step_size, double gamma = 0.1);
    double get_lr() const override;

private:
    int step_size_;
    double gamma_;
    double base_lr_;
};

class MultiStepLR : public LRScheduler {
public:
    MultiStepLR(Optimizer& optimizer, std::vector<int> milestones, double gamma = 0.1);
    double get_lr() const override;

private:
    std::vector<int> milestones_;
    double gamma_;
    double base_lr_;
};

class ExponentialLR : public LRScheduler {
public:
    ExponentialLR(Optimizer& optimizer, double gamma);
    double get_lr() const override;

private:
    double gamma_;
    double base_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, double eta_min = 0.0);
    double get_lr() const override;

private:
    int T_max_;
    double eta_min_;
    double base_lr_;
};

class ReduceLROnPlateau {
public:
    ReduceLROnPlateau(Optimizer& optimizer, const std::string& mode = "min",
                      double factor = 0.1, int patience = 10,
                      double threshold = 1e-4, int cooldown = 0,
                      double min_lr = 0.0);
    
    void step(double metric);

private:
    Optimizer& optimizer_;
    std::string mode_;
    double factor_;
    int patience_;
    double threshold_;
    int cooldown_;
    double min_lr_;
    double best_;
    int num_bad_epochs_;
    int cooldown_counter_;
};

class OneCycleLR : public LRScheduler {
public:
    OneCycleLR(Optimizer& optimizer, double max_lr, int total_steps,
               double pct_start = 0.3, double div_factor = 25.0,
               double final_div_factor = 1e4);
    double get_lr() const override;

private:
    double max_lr_;
    int total_steps_;
    double pct_start_;
    double div_factor_;
    double final_div_factor_;
};

} // namespace nn
} // namespace neurova

#endif // NEUROVA_NEURAL_HPP
