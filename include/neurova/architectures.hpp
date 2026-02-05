// Copyright (c) 2026 @squid consultancy group (scg)
// all rights reserved.
// licensed under the apache license 2.0.

/**
 * neurova/architectures.hpp - Pre-built Neural Network Architectures
 * 
 * This header provides complete neural network architectures:
 * - CNN architectures (LeNet, AlexNet, VGG, ResNet, etc.)
 * - RNN architectures
 * - Transformer architectures
 * - Autoencoders (VAE, etc.)
 * - Object detection (grid-based, SSD, etc.)
 * - Segmentation (U-Net, etc.)
 */

#ifndef NEUROVA_ARCHITECTURES_HPP
#define NEUROVA_ARCHITECTURES_HPP

#include "neural.hpp"

namespace neurova {
namespace arch {

using namespace nn;

// ============================================================================
// Basic Building Blocks
// ============================================================================

// Conv + BatchNorm + ReLU block
class ConvBNReLU : public Layer {
public:
    ConvBNReLU(size_t in_channels, size_t out_channels, 
               size_t kernel_size = 3, size_t stride = 1, 
               size_t padding = 1, size_t groups = 1);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ConvBNReLU"; }

private:
    Conv2D conv_;
    BatchNorm2D bn_;
    ReLU relu_;
};

// Residual block for ResNet
class ResidualBlock : public Layer {
public:
    ResidualBlock(size_t in_channels, size_t out_channels, 
                  size_t stride = 1, bool downsample = false);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ResidualBlock"; }

private:
    Conv2D conv1_, conv2_;
    BatchNorm2D bn1_, bn2_;
    ReLU relu_;
    std::unique_ptr<Conv2D> downsample_conv_;
    std::unique_ptr<BatchNorm2D> downsample_bn_;
    bool do_downsample_;
    Tensor identity_cache_;
};

// Bottleneck block for deeper ResNet
class Bottleneck : public Layer {
public:
    Bottleneck(size_t in_channels, size_t out_channels, 
               size_t stride = 1, size_t groups = 1,
               size_t base_width = 64, bool downsample = false);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Bottleneck"; }

private:
    Conv2D conv1_, conv2_, conv3_;
    BatchNorm2D bn1_, bn2_, bn3_;
    ReLU relu_;
    std::unique_ptr<Conv2D> downsample_conv_;
    std::unique_ptr<BatchNorm2D> downsample_bn_;
    bool do_downsample_;
};

// Inverted residual block for MobileNet
class InvertedResidual : public Layer {
public:
    InvertedResidual(size_t in_channels, size_t out_channels,
                     size_t stride, size_t expand_ratio);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "InvertedResidual"; }

private:
    std::unique_ptr<Sequential> layers_;
    bool use_residual_;
};

// Squeeze-and-Excitation block
class SEBlock : public Layer {
public:
    SEBlock(size_t channels, size_t reduction = 16);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "SEBlock"; }

private:
    AdaptiveAvgPool2D pool_;
    Linear fc1_, fc2_;
    ReLU relu_;
    Sigmoid sigmoid_;
};

// Inception module
class InceptionModule : public Layer {
public:
    InceptionModule(size_t in_channels, size_t ch1x1, 
                    size_t ch3x3red, size_t ch3x3,
                    size_t ch5x5red, size_t ch5x5, size_t pool_proj);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "InceptionModule"; }

private:
    Conv2D branch1_;
    Conv2D branch2a_, branch2b_;
    Conv2D branch3a_, branch3b_;
    MaxPool2D branch4a_;
    Conv2D branch4b_;
};

// Dense block for DenseNet
class DenseLayer : public Layer {
public:
    DenseLayer(size_t in_channels, size_t growth_rate, size_t bn_size = 4);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "DenseLayer"; }

private:
    BatchNorm2D bn1_, bn2_;
    ReLU relu_;
    Conv2D conv1_, conv2_;
};

class DenseBlock : public Layer {
public:
    DenseBlock(size_t num_layers, size_t in_channels, 
               size_t growth_rate, size_t bn_size = 4);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "DenseBlock"; }

private:
    std::vector<std::unique_ptr<DenseLayer>> layers_;
};

// Transition layer for DenseNet
class Transition : public Layer {
public:
    Transition(size_t in_channels, size_t out_channels);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Transition"; }

private:
    BatchNorm2D bn_;
    ReLU relu_;
    Conv2D conv_;
    AvgPool2D pool_;
};

// ============================================================================
// Transformer Blocks
// ============================================================================

class TransformerEncoderLayer : public Layer {
public:
    TransformerEncoderLayer(size_t d_model, size_t nhead, 
                            size_t dim_feedforward = 2048,
                            double dropout = 0.1,
                            const std::string& activation = "relu");
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, const Tensor* src_mask = nullptr,
                   const Tensor* src_key_padding_mask = nullptr);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "TransformerEncoderLayer"; }

private:
    MultiHeadAttention self_attn_;
    Linear linear1_, linear2_;
    LayerNorm norm1_, norm2_;
    Dropout dropout_;
    std::string activation_;
};

class TransformerDecoderLayer : public Layer {
public:
    TransformerDecoderLayer(size_t d_model, size_t nhead,
                            size_t dim_feedforward = 2048,
                            double dropout = 0.1,
                            const std::string& activation = "relu");
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& tgt, const Tensor& memory,
                   const Tensor* tgt_mask = nullptr,
                   const Tensor* memory_mask = nullptr);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "TransformerDecoderLayer"; }

private:
    MultiHeadAttention self_attn_, cross_attn_;
    Linear linear1_, linear2_;
    LayerNorm norm1_, norm2_, norm3_;
    Dropout dropout_;
    std::string activation_;
};

class TransformerEncoder : public Layer {
public:
    TransformerEncoder(size_t d_model, size_t nhead, size_t num_layers,
                       size_t dim_feedforward = 2048, double dropout = 0.1);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "TransformerEncoder"; }

private:
    std::vector<std::unique_ptr<TransformerEncoderLayer>> layers_;
    LayerNorm norm_;
};

class TransformerDecoder : public Layer {
public:
    TransformerDecoder(size_t d_model, size_t nhead, size_t num_layers,
                       size_t dim_feedforward = 2048, double dropout = 0.1);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& tgt, const Tensor& memory);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "TransformerDecoder"; }

private:
    std::vector<std::unique_ptr<TransformerDecoderLayer>> layers_;
    LayerNorm norm_;
};

// Positional Encoding
class PositionalEncoding : public Layer {
public:
    PositionalEncoding(size_t d_model, double dropout = 0.1, size_t max_len = 5000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::string name() const override { return "PositionalEncoding"; }

private:
    Tensor pe_;
    Dropout dropout_;
};

// ============================================================================
// Complete CNN Architectures
// ============================================================================

// LeNet-5
class LeNet5 : public Layer {
public:
    LeNet5(size_t num_classes = 10);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "LeNet5"; }

private:
    Conv2D conv1_, conv2_;
    MaxPool2D pool_;
    Linear fc1_, fc2_, fc3_;
    ReLU relu_;
    Flatten flatten_;
};

// AlexNet
class AlexNet : public Layer {
public:
    AlexNet(size_t num_classes = 1000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "AlexNet"; }

private:
    Sequential features_;
    AdaptiveAvgPool2D avgpool_;
    Sequential classifier_;
};

// VGG (configurable)
class VGG : public Layer {
public:
    VGG(const std::vector<int>& cfg, size_t num_classes = 1000, bool batch_norm = false);
    
    static VGG vgg11(size_t num_classes = 1000, bool batch_norm = false);
    static VGG vgg13(size_t num_classes = 1000, bool batch_norm = false);
    static VGG vgg16(size_t num_classes = 1000, bool batch_norm = false);
    static VGG vgg19(size_t num_classes = 1000, bool batch_norm = false);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "VGG"; }

private:
    Sequential features_;
    AdaptiveAvgPool2D avgpool_;
    Sequential classifier_;
    
    Sequential make_layers(const std::vector<int>& cfg, bool batch_norm);
};

// ResNet
class ResNet : public Layer {
public:
    ResNet(const std::vector<int>& layers, size_t num_classes = 1000, 
           bool use_bottleneck = false);
    
    static ResNet resnet18(size_t num_classes = 1000);
    static ResNet resnet34(size_t num_classes = 1000);
    static ResNet resnet50(size_t num_classes = 1000);
    static ResNet resnet101(size_t num_classes = 1000);
    static ResNet resnet152(size_t num_classes = 1000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ResNet"; }

private:
    Conv2D conv1_;
    BatchNorm2D bn1_;
    ReLU relu_;
    MaxPool2D maxpool_;
    Sequential layer1_, layer2_, layer3_, layer4_;
    AdaptiveAvgPool2D avgpool_;
    Flatten flatten_;
    Linear fc_;
    
    Sequential make_layer(size_t in_channels, size_t out_channels, 
                          int blocks, size_t stride, bool use_bottleneck);
};

// MobileNetV2
class MobileNetV2 : public Layer {
public:
    MobileNetV2(size_t num_classes = 1000, double width_mult = 1.0);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "MobileNetV2"; }

private:
    Sequential features_;
    Sequential classifier_;
};

// EfficientNet
class EfficientNet : public Layer {
public:
    EfficientNet(double width_mult, double depth_mult, double dropout, 
                 size_t num_classes = 1000);
    
    static EfficientNet efficientnet_b0(size_t num_classes = 1000);
    static EfficientNet efficientnet_b1(size_t num_classes = 1000);
    static EfficientNet efficientnet_b2(size_t num_classes = 1000);
    static EfficientNet efficientnet_b3(size_t num_classes = 1000);
    static EfficientNet efficientnet_b4(size_t num_classes = 1000);
    static EfficientNet efficientnet_b5(size_t num_classes = 1000);
    static EfficientNet efficientnet_b6(size_t num_classes = 1000);
    static EfficientNet efficientnet_b7(size_t num_classes = 1000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "EfficientNet"; }

private:
    Sequential features_;
    AdaptiveAvgPool2D avgpool_;
    Dropout dropout_;
    Linear classifier_;
};

// DenseNet
class DenseNet : public Layer {
public:
    DenseNet(const std::vector<int>& block_config, size_t num_init_features = 64,
             size_t growth_rate = 32, size_t bn_size = 4, 
             double drop_rate = 0.0, size_t num_classes = 1000);
    
    static DenseNet densenet121(size_t num_classes = 1000);
    static DenseNet densenet169(size_t num_classes = 1000);
    static DenseNet densenet201(size_t num_classes = 1000);
    static DenseNet densenet264(size_t num_classes = 1000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "DenseNet"; }

private:
    Sequential features_;
    Linear classifier_;
};

// Inception/GoogLeNet
class InceptionV3 : public Layer {
public:
    InceptionV3(size_t num_classes = 1000, bool aux_logits = true);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "InceptionV3"; }

private:
    Sequential features_;
    AdaptiveAvgPool2D avgpool_;
    Dropout dropout_;
    Linear fc_;
    bool aux_logits_;
};

// ============================================================================
// Complete Transformer Architectures
// ============================================================================

// Transformer
class Transformer : public Layer {
public:
    Transformer(size_t d_model = 512, size_t nhead = 8,
                size_t num_encoder_layers = 6, size_t num_decoder_layers = 6,
                size_t dim_feedforward = 2048, double dropout = 0.1);
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& src, const Tensor& tgt,
                   const Tensor* src_mask = nullptr,
                   const Tensor* tgt_mask = nullptr);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Transformer"; }
    
    // Generate square mask for sequence
    static Tensor generate_square_subsequent_mask(size_t sz);

private:
    TransformerEncoder encoder_;
    TransformerDecoder decoder_;
};

// Vision Transformer (ViT)
class VisionTransformer : public Layer {
public:
    VisionTransformer(size_t image_size = 224, size_t patch_size = 16,
                      size_t num_classes = 1000, size_t dim = 768,
                      size_t depth = 12, size_t heads = 12, size_t mlp_dim = 3072,
                      double dropout = 0.0, double emb_dropout = 0.0);
    
    static VisionTransformer vit_base_patch16_224(size_t num_classes = 1000);
    static VisionTransformer vit_base_patch32_224(size_t num_classes = 1000);
    static VisionTransformer vit_large_patch16_224(size_t num_classes = 1000);
    static VisionTransformer vit_huge_patch14_224(size_t num_classes = 1000);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "VisionTransformer"; }

private:
    size_t patch_size_, num_patches_;
    Conv2D patch_embed_;
    GradTensor cls_token_;
    GradTensor pos_embed_;
    Dropout dropout_;
    TransformerEncoder transformer_;
    LayerNorm norm_;
    Linear head_;
};

// BERT-style encoder
class BERT : public Layer {
public:
    BERT(size_t vocab_size, size_t d_model = 768, size_t nhead = 12,
         size_t num_layers = 12, size_t dim_feedforward = 3072,
         size_t max_seq_len = 512, double dropout = 0.1);
    
    static BERT bert_base();
    static BERT bert_large();
    
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& input_ids, const Tensor* attention_mask = nullptr,
                   const Tensor* token_type_ids = nullptr);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "BERT"; }

private:
    Embedding word_embed_;
    Embedding position_embed_;
    Embedding token_type_embed_;
    LayerNorm embed_norm_;
    Dropout embed_dropout_;
    TransformerEncoder encoder_;
    Linear pooler_;
};

// GPT-style decoder
class GPT : public Layer {
public:
    GPT(size_t vocab_size, size_t d_model = 768, size_t nhead = 12,
        size_t num_layers = 12, size_t dim_feedforward = 3072,
        size_t max_seq_len = 1024, double dropout = 0.1);
    
    static GPT gpt2_small();
    static GPT gpt2_medium();
    static GPT gpt2_large();
    static GPT gpt2_xl();
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "GPT"; }
    
    // Generate text autoregressively
    Tensor generate(const Tensor& input_ids, size_t max_length,
                    double temperature = 1.0, int top_k = 0,
                    double top_p = 1.0);

private:
    Embedding token_embed_;
    Embedding position_embed_;
    Dropout dropout_;
    std::vector<std::unique_ptr<TransformerDecoderLayer>> layers_;
    LayerNorm norm_;
    Linear lm_head_;
};

// ============================================================================
// Autoencoder Architectures
// ============================================================================

// Basic Autoencoder
class Autoencoder : public Layer {
public:
    Autoencoder(size_t input_dim, const std::vector<size_t>& hidden_dims);
    
    Tensor forward(const Tensor& x) override;
    Tensor encode(const Tensor& x);
    Tensor decode(const Tensor& z);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Autoencoder"; }

private:
    Sequential encoder_;
    Sequential decoder_;
};

// Variational Autoencoder (VAE)
class VAE : public Layer {
public:
    VAE(size_t input_dim, const std::vector<size_t>& hidden_dims, size_t latent_dim);
    
    Tensor forward(const Tensor& x) override;
    std::tuple<Tensor, Tensor, Tensor> forward_with_latent(const Tensor& x);
    Tensor encode(const Tensor& x);
    std::pair<Tensor, Tensor> encode_distribution(const Tensor& x);
    Tensor decode(const Tensor& z);
    Tensor reparameterize(const Tensor& mu, const Tensor& logvar);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "VAE"; }
    
    // VAE loss = reconstruction loss + KL divergence
    double vae_loss(const Tensor& x, const Tensor& recon, 
                    const Tensor& mu, const Tensor& logvar);

private:
    Sequential encoder_;
    Linear fc_mu_, fc_logvar_;
    Sequential decoder_;
    size_t latent_dim_;
};

// Convolutional VAE
class ConvVAE : public Layer {
public:
    ConvVAE(size_t in_channels, const std::vector<size_t>& hidden_channels,
            size_t latent_dim, size_t image_size = 64);
    
    Tensor forward(const Tensor& x) override;
    std::tuple<Tensor, Tensor, Tensor> forward_with_latent(const Tensor& x);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ConvVAE"; }

private:
    Sequential encoder_;
    Linear fc_mu_, fc_logvar_, fc_decode_;
    Sequential decoder_;
    size_t latent_dim_;
};

// ============================================================================
// Object Detection Architectures
// ============================================================================

// Simple feature pyramid network
class FPN : public Layer {
public:
    FPN(const std::vector<size_t>& in_channels, size_t out_channels);
    
    Tensor forward(const Tensor& x) override;
    std::vector<Tensor> forward_multi(const std::vector<Tensor>& features);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "FPN"; }

private:
    std::vector<std::unique_ptr<Conv2D>> lateral_convs_;
    std::vector<std::unique_ptr<Conv2D>> output_convs_;
};

// Grid-based detection head
class GridHead : public Layer {
public:
    GridHead(size_t in_channels, size_t num_classes, size_t num_anchors = 3);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "GridHead"; }

private:
    Conv2D conv_;
    size_t num_classes_, num_anchors_;
};

// SSD-style multi-box head
class SSDHead : public Layer {
public:
    SSDHead(const std::vector<size_t>& in_channels, 
            const std::vector<size_t>& num_anchors,
            size_t num_classes);
    
    Tensor forward(const Tensor& x) override;
    std::pair<Tensor, Tensor> forward_multi(const std::vector<Tensor>& features);
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "SSDHead"; }

private:
    std::vector<std::unique_ptr<Conv2D>> cls_heads_;
    std::vector<std::unique_ptr<Conv2D>> reg_heads_;
    size_t num_classes_;
};

// ============================================================================
// Segmentation Architectures
// ============================================================================

// U-Net
class UNet : public Layer {
public:
    UNet(size_t in_channels, size_t num_classes, size_t base_features = 64);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "UNet"; }

private:
    // Encoder
    std::vector<std::unique_ptr<ConvBNReLU>> enc_convs_;
    MaxPool2D pool_;
    
    // Bottleneck
    ConvBNReLU bottleneck1_, bottleneck2_;
    
    // Decoder
    std::vector<std::unique_ptr<ConvTranspose2D>> up_convs_;
    std::vector<std::unique_ptr<ConvBNReLU>> dec_convs_;
    
    // Output
    Conv2D out_conv_;
};

// DeepLab v3+
class DeepLabV3Plus : public Layer {
public:
    DeepLabV3Plus(size_t num_classes, const std::string& backbone = "resnet50");
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "DeepLabV3Plus"; }

private:
    std::unique_ptr<ResNet> backbone_;
    // ASPP module
    std::vector<std::unique_ptr<Conv2D>> aspp_convs_;
    AdaptiveAvgPool2D global_pool_;
    Conv2D project_conv_;
    Conv2D low_level_conv_;
    Conv2D output_conv_;
};

// ============================================================================
// GAN Components
// ============================================================================

// Basic Generator for image generation
class Generator : public Layer {
public:
    Generator(size_t latent_dim, size_t img_channels = 3, size_t img_size = 64);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Generator"; }

private:
    Sequential model_;
};

// Basic Discriminator
class Discriminator : public Layer {
public:
    Discriminator(size_t img_channels = 3, size_t img_size = 64);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "Discriminator"; }

private:
    Sequential model_;
};

// ============================================================================
// MLP Architectures
// ============================================================================

// Simple MLP
class MLP : public Layer {
public:
    MLP(size_t input_dim, const std::vector<size_t>& hidden_dims, 
        size_t output_dim, double dropout = 0.0,
        const std::string& activation = "relu");
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "MLP"; }

private:
    Sequential model_;
};

// Classifier head
class ClassifierHead : public Layer {
public:
    ClassifierHead(size_t in_features, size_t num_classes, 
                   double dropout = 0.0);
    
    Tensor forward(const Tensor& x) override;
    Tensor backward(const Tensor& grad) override;
    std::vector<GradTensor*> parameters() override;
    std::string name() const override { return "ClassifierHead"; }

private:
    GlobalAvgPool2D pool_;
    Flatten flatten_;
    Dropout dropout_;
    Linear fc_;
};

} // namespace arch
} // namespace neurova

#endif // NEUROVA_ARCHITECTURES_HPP
