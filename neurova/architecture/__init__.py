# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Neurova Architecture Module

Comprehensive pre-built neural network architectures with easy-to-use interfaces.
Minimal code required - just configure, train, and evaluate.

Features
--------
- 100+ Pre-built architectures across all major neural network families
- Automatic parameter validation
- Built-in training with progress tracking
- Plotting for loss, accuracy, and metrics
- Hyperparameter tuning integration (PARAM_SPACE on every model)
- Model save/load functionality

ARCHITECTURE FAMILIES
---------------------

Basic/Foundational:
- Perceptron, SingleLayerNetwork, MultiLayerPerceptron, FeedforwardNetwork, DenseNet

CNN Architectures:
- SimpleCNN, LeNet, AlexNet, VGGNet, CNN (customizable)
- FCN, GoogLeNet, Inception, ResNet (18/34/50/101/152)
- DenseNetCNN, MobileNet (V1/V2), EfficientNet (B0-B7)
- Xception, SENet, NASNet

RNN/Sequence Models:
- RNN, LSTM, GRU, BiLSTM, StackedLSTM, Seq2Seq

Transformer/Attention:
- Transformer, TransformerEncoder, VisionTransformer
- BERT (Base/Large/Tiny), GPT (Small/Medium/Large)
- T5, RoBERTa, ALBERT, DistilBERT, XLNet
- SwinTransformer, CLIP, Perceiver

Generative Models:
- GAN, WGAN, ConditionalGAN
- DCGAN, WGAN_GP, Pix2Pix, CycleGAN, StyleGAN
- DDPM, DDIM (Diffusion models)
- VAE, BetaVAE, CVAE

Graph Neural Networks:
- GNN, GCN, GAT, GraphSAGE, MPNN, GIN, ChebNet

Reinforcement Learning:
- DQN, DoubleDQN, DuelingDQN
- PolicyGradient, ActorCritic, A2C, A3C, PPO
- DDPG, SAC, TD3

Specialized Architectures:
- CapsuleNetwork, SiameseNetwork, TripletNetwork
- MemoryNetwork, NTM (Neural Turing Machine)
- MixtureOfExperts, LiquidNeuralNetwork
- SpikingNeuralNetwork, HopfieldNetwork, RBM

Quick Start
-----------
>>> from neurova.architecture import MLP
>>> 
>>> model = MLP(input_shape=784, output_shape=10, hidden_layers=[256, 128])
>>> model.fit(X_train, y_train, epochs=20)
>>> accuracy = model.score(X_test, y_test)
>>> model.plot_history()

Available Utilities
-------------------
- Callbacks: EarlyStopping, ModelCheckpoint, LearningRateScheduler
- Metrics: accuracy, f1_score, confusion_matrix, mse, rmse
- Plotting: plot_training_history, plot_confusion_matrix, plot_roc_curve
- Tuning: GridSearchCV, RandomSearchCV, BayesianOptimization, AutoML
"""

# Base architecture
from .base import (
    BaseArchitecture,
    ParameterValidator,
    TrainingHistory,
    EarlyStopping as BaseEarlyStopping,
    LearningRateScheduler as BaseLRScheduler,
)

# CNN architectures
from .cnn import (
    CNN,
    LeNet,
    AlexNet,
    VGGNet,
    SimpleCNN,
    ResidualBlock,
    ConvLayer,
    MaxPoolLayer,
    BatchNormLayer,
    DropoutLayer,
    DenseLayer as ConvDenseLayer,
    create_cnn,
)

# RNN architectures
from .rnn import (
    RNN,
    LSTM,
    GRU,
    BiLSTM,
    StackedLSTM,
    SimpleRNN,
    Seq2Seq,
    Seq2SeqEncoder,
    Seq2SeqDecoder,
    RNNCell,
    LSTMCell,
    GRUCell,
    create_rnn,
)

# Transformer architectures
from .transformer import (
    Transformer,
    TransformerEncoder,
    VisionTransformer,
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    LayerNorm as TransformerLayerNorm,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    create_transformer,
)

# MLP, Autoencoder, GAN
from .mlp import (
    MLP,
    Autoencoder,
    VAE,
    GAN,
    WGAN,
    ConditionalGAN,
    DenseLayer,
    BatchNorm1D,
    Dropout,
    create_mlp,
    create_autoencoder,
)

# Training utilities
from .utils import (
    # Callbacks
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ReduceLROnPlateau,
    TensorBoardLogger,
    ProgressBar,
    Callback,
    CallbackList,
    
    # Metrics
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix,
    classification_report,
    mse,
    rmse,
    mae,
    r2_score,
    
    # Plotting
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_learning_curves,
    plot_feature_importance,
    visualize_weights,
    
    # Data utilities
    train_test_split,
    cross_val_score,
    one_hot_encode,
    standardize,
    normalize,
)

# Hyperparameter tuning
from .tuning import (
    GridSearchCV,
    RandomSearchCV,
    BayesianOptimization,
    AutoML,
    HyperparameterSpace,
    TuningResult,
    tune_model,
)

# NEW COMPREHENSIVE ARCHITECTURE MODULES (Phase 2)

# Foundational architectures
from .foundational import (
    Perceptron,
    SingleLayerNetwork,
    MultiLayerPerceptron,
    FeedforwardNetwork,
    DenseNet,
    DenseBlock,
    Layer,
    create_feedforward,
)

# Advanced CNN architectures
from .cnn_advanced import (
    FCN,
    GoogLeNet,
    Inception,
    InceptionModule,
    ResNet,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    DenseNetCNN,
    DenseBlock as DenseBlockCNN,
    TransitionLayer,
    MobileNet,
    MobileNetV1,
    MobileNetV2,
    InvertedResidual,
    EfficientNet,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    Xception,
    SENet,
    SqueezeExcitation,
)

# Advanced Transformer architectures
from .transformer_advanced import (
    BERT,
    BERTBase,
    BERTLarge,
    BERTTiny,
    GPT,
    GPT2Small,
    GPT2Medium,
    GPT2Large,
    T5,
    T5Small,
    T5Base,
    T5Large,
    RoBERTa,
    ALBERT,
    ALBERTBase,
    ALBERTLarge,
    DistilBERT,
    XLNet,
    SwinTransformer,
    SwinTiny,
    SwinSmall,
    SwinBase,
    CLIP,
    Perceiver,
)

# Generative models
from .generative import (
    DCGAN,
    DCGANGenerator,
    DCGANDiscriminator,
    WGAN_GP,
    Pix2Pix,
    CycleGAN,
    StyleGAN,
    MappingNetwork,
    StyleBlock,
    DDPM,
    DDIM,
    BetaVAE,
    CVAE,
    create_generative_model,
)

# Graph Neural Networks
from .graph import (
    GNN,
    GCN,
    GAT,
    GraphSAGE,
    MPNN,
    GIN,
    ChebNet,
    GraphConvLayer,
    GraphAttentionLayer,
    SAGEConvLayer,
    MessagePassingLayer,
    GINLayer,
    ChebConvLayer,
    EdgeConvLayer,
    GlobalMeanPool,
    GlobalSumPool,
    GlobalMaxPool,
    normalize_adjacency,
    compute_laplacian,
    create_gnn,
)

# Reinforcement Learning
from .reinforcement import (
    DQN,
    DoubleDQN,
    DuelingDQN,
    PolicyGradient,
    ActorCritic,
    A2C,
    A3C,
    PPO,
    DDPG,
    SAC,
    TD3,
    QNetwork,
    DuelingQNetwork,
    PolicyNetwork,
    ValueNetwork,
    CriticNetwork,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    create_rl_agent,
)

# Specialized architectures
from .specialized import (
    CapsuleNetwork,
    PrimaryCapsule,
    DigitCapsule,
    SiameseNetwork,
    TripletNetwork,
    MemoryNetwork,
    NTM,
    MixtureOfExperts,
    LiquidNeuralNetwork,
    SpikingNeuralNetwork,
    HopfieldNetwork,
    RestrictedBoltzmannMachine,
    EnergyBasedModel,
    create_specialized_model,
)

# Version
__version__ = '0.2.0'

# All public exports
__all__ = [
    # Base
    'BaseArchitecture',
    'ParameterValidator',
    'TrainingHistory',
    
    # FOUNDATIONAL NETWORKS
    'Perceptron',
    'SingleLayerNetwork',
    'MultiLayerPerceptron',
    'FeedforwardNetwork',
    'DenseNet',
    'DenseBlock',
    'Layer',
    'create_feedforward',
    
    # CNN ARCHITECTURES
    # Basic
    'CNN',
    'LeNet',
    'AlexNet',
    'VGGNet',
    'SimpleCNN',
    'ResidualBlock',
    'ConvLayer',
    'MaxPoolLayer',
    'BatchNormLayer',
    'DropoutLayer',
    'create_cnn',
    
    # Advanced CNN
    'FCN',
    'GoogLeNet',
    'Inception',
    'InceptionModule',
    'ResNet',
    'ResNet18',
    'ResNet34',
    'ResNet50',
    'ResNet101',
    'ResNet152',
    'DenseNetCNN',
    'DenseBlock',
    'TransitionLayer',
    'MobileNet',
    'MobileNetV1',
    'MobileNetV2',
    'InvertedResidual',
    'EfficientNet',
    'EfficientNetB0',
    'EfficientNetB1',
    'EfficientNetB2',
    'EfficientNetB3',
    'Xception',
    'SENet',
    'SqueezeExcitation',
    
    # RNN ARCHITECTURES
    'RNN',
    'LSTM',
    'GRU',
    'BiLSTM',
    'StackedLSTM',
    'SimpleRNN',
    'Seq2Seq',
    'Seq2SeqEncoder',
    'Seq2SeqDecoder',
    'RNNCell',
    'LSTMCell',
    'GRUCell',
    'create_rnn',
    
    # TRANSFORMER ARCHITECTURES
    # Basic
    'Transformer',
    'TransformerEncoder',
    'VisionTransformer',
    'MultiHeadAttention',
    'PositionalEncoding',
    'FeedForward',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'create_transformer',
    
    # Advanced Transformers
    'BERT',
    'BERTBase',
    'BERTLarge',
    'BERTTiny',
    'GPT',
    'GPT2Small',
    'GPT2Medium',
    'GPT2Large',
    'T5',
    'T5Small',
    'T5Base',
    'T5Large',
    'RoBERTa',
    'ALBERT',
    'ALBERTBase',
    'ALBERTLarge',
    'DistilBERT',
    'XLNet',
    'SwinTransformer',
    'SwinTiny',
    'SwinSmall',
    'SwinBase',
    'CLIP',
    'Perceiver',
    
    # MLP / AUTOENCODER / BASIC GENERATIVE
    'MLP',
    'Autoencoder',
    'VAE',
    'GAN',
    'WGAN',
    'ConditionalGAN',
    'DenseLayer',
    'BatchNorm1D',
    'Dropout',
    'create_mlp',
    'create_autoencoder',
    
    # ADVANCED GENERATIVE MODELS
    'DCGAN',
    'DCGANGenerator',
    'DCGANDiscriminator',
    'WGAN_GP',
    'Pix2Pix',
    'CycleGAN',
    'StyleGAN',
    'MappingNetwork',
    'StyleBlock',
    'DDPM',
    'DDIM',
    'BetaVAE',
    'CVAE',
    'create_generative_model',
    
    # GRAPH NEURAL NETWORKS
    'GNN',
    'GCN',
    'GAT',
    'GraphSAGE',
    'MPNN',
    'GIN',
    'ChebNet',
    'GraphConvLayer',
    'GraphAttentionLayer',
    'SAGEConvLayer',
    'MessagePassingLayer',
    'GINLayer',
    'ChebConvLayer',
    'EdgeConvLayer',
    'GlobalMeanPool',
    'GlobalSumPool',
    'GlobalMaxPool',
    'normalize_adjacency',
    'compute_laplacian',
    'create_gnn',
    
    # REINFORCEMENT LEARNING
    'DQN',
    'DoubleDQN',
    'DuelingDQN',
    'PolicyGradient',
    'ActorCritic',
    'A2C',
    'A3C',
    'PPO',
    'DDPG',
    'SAC',
    'TD3',
    'QNetwork',
    'DuelingQNetwork',
    'PolicyNetwork',
    'ValueNetwork',
    'CriticNetwork',
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'create_rl_agent',
    
    # SPECIALIZED ARCHITECTURES
    'CapsuleNetwork',
    'PrimaryCapsule',
    'DigitCapsule',
    'SiameseNetwork',
    'TripletNetwork',
    'MemoryNetwork',
    'NTM',
    'MixtureOfExperts',
    'LiquidNeuralNetwork',
    'SpikingNeuralNetwork',
    'HopfieldNetwork',
    'RestrictedBoltzmannMachine',
    'EnergyBasedModel',
    'create_specialized_model',
    
    # CALLBACKS
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'ReduceLROnPlateau',
    'TensorBoardLogger',
    'ProgressBar',
    
    # METRICS
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'confusion_matrix',
    'classification_report',
    'mse',
    'rmse',
    'mae',
    'r2_score',
    
    # PLOTTING
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_learning_curves',
    'plot_feature_importance',
    'visualize_weights',
    
    # DATA UTILITIES
    'train_test_split',
    'cross_val_score',
    'one_hot_encode',
    'standardize',
    'normalize',
    
    # HYPERPARAMETER TUNING
    'GridSearchCV',
    'RandomSearchCV',
    'BayesianOptimization',
    'AutoML',
    'HyperparameterSpace',
    'TuningResult',
    'tune_model',
]


# Quick info function
def available_architectures():
    """Print available architectures and their use cases."""
    print("""
neurova architecture module v0.2.0
100+ neural network architectures

BASIC/FOUNDATIONAL NETWORKS
---------------------------
  Perceptron            - Single-layer, binary classification
  SingleLayerNetwork    - Basic neural network
  MultiLayerPerceptron  - Classic MLP, tabular data
  FeedforwardNetwork    - Customizable feedforward nets
  MLP                   - Flexible multi-layer perceptron

CNN ARCHITECTURES (Image Classification)
----------------------------------------
  SimpleCNN             - Quick prototype, small datasets
  LeNet                 - Classic, MNIST-like tasks
  AlexNet               - Deeper architecture
  VGGNet                - Very deep (11/13/16/19 layers)
  CNN                   - Fully customizable
  FCN                   - Fully Convolutional Networks
  GoogLeNet/Inception   - Inception modules
  ResNet                - Residual connections (18/34/50/101/152)
  DenseNetCNN           - Dense connections
  MobileNet             - Lightweight (V1/V2)
  EfficientNet          - Compound scaling (B0-B7)
  Xception              - Extreme Inception
  SENet                 - Squeeze-and-Excitation

RNN/SEQUENCE MODELS
-------------------
  RNN, SimpleRNN        - Basic recurrent network
  LSTM                  - Long short-term memory
  GRU                   - Gated recurrent unit
  BiLSTM                - Bidirectional LSTM
  StackedLSTM           - Deep stacked LSTM
  Seq2Seq               - Encoder-decoder for translation

TRANSFORMER/ATTENTION ARCHITECTURES
-----------------------------------
  Transformer           - Full encoder-decoder
  TransformerEncoder    - Encoder only
  VisionTransformer     - ViT for images
  BERT                  - Bidirectional encoder (Base/Large/Tiny)
  GPT                   - Autoregressive (GPT2 Small/Medium/Large)
  T5                    - Text-to-text (Small/Base/Large)
  RoBERTa               - Robustly optimized BERT
  ALBERT                - Lite BERT (Base/Large)
  DistilBERT            - Distilled BERT
  XLNet                 - Permutation language modeling
  SwinTransformer       - Shifted windows (Tiny/Small/Base)
  CLIP                  - Contrastive language-image
  Perceiver             - General-purpose architecture

GENERATIVE MODELS
-----------------
  GAN                   - Generative adversarial network
  WGAN                  - Wasserstein GAN
  ConditionalGAN        - Class-conditional generation
  DCGAN                 - Deep convolutional GAN
  WGAN_GP               - WGAN with gradient penalty
  Pix2Pix               - Image-to-image translation
  CycleGAN              - Unpaired image translation
  StyleGAN              - Style-based generator
  DDPM                  - Denoising diffusion
  DDIM                  - Denoising diffusion (fast)
  VAE                   - Variational autoencoder
  BetaVAE               - Disentangled VAE
  CVAE                  - Conditional VAE

GRAPH NEURAL NETWORKS
---------------------
  GNN                   - Basic graph neural network
  GCN                   - Graph convolutional network
  GAT                   - Graph attention network
  GraphSAGE             - Sample and aggregate
  MPNN                  - Message passing neural network
  GIN                   - Graph isomorphism network
  ChebNet               - Chebyshev spectral graph conv

REINFORCEMENT LEARNING
----------------------
  DQN                   - Deep Q-Network
  DoubleDQN             - Double DQN
  DuelingDQN            - Dueling architecture
  PolicyGradient        - REINFORCE algorithm
  ActorCritic           - Actor-critic method
  A2C                   - Advantage actor-critic
  A3C                   - Asynchronous A3C
  PPO                   - Proximal policy optimization
  DDPG                  - Deep deterministic PG
  SAC                   - Soft actor-critic
  TD3                   - Twin delayed DDPG

SPECIALIZED ARCHITECTURES
-------------------------
  CapsuleNetwork        - Capsules with dynamic routing
  SiameseNetwork        - Similarity learning
  TripletNetwork        - Metric learning
  MemoryNetwork         - Memory-augmented
  NTM                   - Neural Turing Machine
  MixtureOfExperts      - Sparse MoE
  LiquidNeuralNetwork   - Liquid time-constant
  SpikingNeuralNetwork  - LIF neurons
  HopfieldNetwork       - Associative memory
  RestrictedBoltzmannMachine - Energy-based generative

quick start examples

# MLP for classification
>>> from neurova.architecture import MLP
>>> model = MLP(input_shape=100, output_shape=10)
>>> model.fit(X, y, epochs=50)

# ResNet for images  
>>> from neurova.architecture import ResNet50
>>> model = ResNet50(input_shape=(224, 224, 3), num_classes=1000)

# BERT for NLP
>>> from neurova.architecture import BERTBase
>>> model = BERTBase(vocab_size=30000, max_seq_len=512)

# GCN for graphs
>>> from neurova.architecture import GCN
>>> model = GCN(input_dim=1433, hidden_dims=[16], output_dim=7)

# PPO for RL
>>> from neurova.architecture import PPO
>>> agent = PPO(state_dim=4, action_dim=2)

# Hyperparameter tuning (all models have PARAM_SPACE)
>>> from neurova.architecture import RandomSearchCV, MLP
>>> search = RandomSearchCV(MLP, MLP.PARAM_SPACE, n_iter=20)
>>> search.fit(X, y, input_shape=100, output_shape=10)
>>> best_model = search.best_model_

""")


def quick_start():
    """Print quick start guide."""
    print("""

              NEUROVA ARCHITECTURE - QUICK START                   


1. IMPORT WHAT YOU NEED
   >>> from neurova.architecture import MLP, LSTM, create_cnn
   >>> from neurova.architecture import train_test_split, plot_training_history

2. PREPARE DATA
   >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

3. CREATE MODEL
   # For tabular data
   >>> model = MLP(input_shape=X.shape[1], output_shape=num_classes)
   
   # For images
   >>> model = create_cnn(input_shape=X.shape[1:], num_classes=num_classes)
   
   # For sequences
   >>> model = LSTM(input_shape=X.shape[1:], output_shape=num_classes)

4. TRAIN
   >>> model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

5. EVALUATE
   >>> accuracy = model.score(X_test, y_test)
   >>> print(f"Test accuracy: {accuracy:.4f}")

6. VISUALIZE
   >>> model.plot_history()
   >>> model.summary()

7. SAVE & LOAD
   >>> model.save('my_model.npz')
   >>> model = MLP.load('my_model.npz')

8. HYPERPARAMETER TUNING (Optional)
   >>> from neurova.architecture import RandomSearchCV
   >>> search = RandomSearchCV(MLP, {'learning_rate': [0.001, 0.01, 0.1]}, n_iter=10)
   >>> search.fit(X_train, y_train, input_shape=X.shape[1], output_shape=num_classes)
   >>> best_model = search.best_model_


""")
