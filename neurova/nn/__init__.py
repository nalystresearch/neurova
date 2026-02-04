# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Neurova Neural Networks - Deep Learning API.

This module provides automatic differentiation, neural network layers,
optimizers, and loss functions for building and training deep learning models.

Features:
- Automatic differentiation (autograd)
- 50+ neural network layers
- 20+ optimizers
- 50+ loss functions
- Modern deep learning API
"""

# core autograd
from neurova.nn.autograd import Tensor, Parameter, no_grad

# layers - Core
from neurova.nn.layers import (
    Module,
    Sequential,
    ModuleList,
    ModuleDict,
)

# layers - Linear
from neurova.nn.linear import (
    Linear,
    Bilinear,
    Identity,
)

# layers - Convolutional
from neurova.nn.conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

# layers - Pooling
from neurova.nn.pooling import (
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
)

# layers - Normalization
from neurova.nn.normalization import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    LayerNorm,
    GroupNorm,
    InstanceNorm1d,
    InstanceNorm2d,
)

# layers - Activation
from neurova.nn.activation import (
    ReLU,
    LeakyReLU,
    PReLU,
    ELU,
    SELU,
    GELU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    Swish,
    Mish,
    SiLU,
    Softplus,
    Softsign,
    Hardtanh,
    Hardswish,
    Hardsigmoid,
    ReLU6,
    CELU,
    GLU,
    Threshold,
    Softmin,
    Softmax2d,
)

# layers - Dropout
from neurova.nn.dropout import (
    Dropout,
    Dropout2d,
    Dropout3d,
    AlphaDropout,
)

# layers - Recurrent
from neurova.nn.recurrent import (
    RNN,
    LSTM,
    GRU,
    RNNCell,
    LSTMCell,
    GRUCell,
)

# layers - Attention
from neurova.nn.attention import (
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)

# layers - Embedding
from neurova.nn.embedding import (
    Embedding,
    EmbeddingBag,
)

# layers - Padding
from neurova.nn.padding import (
    ReflectionPad1d,
    ReflectionPad2d,
    ReflectionPad3d,
    ReplicationPad1d,
    ReplicationPad2d,
    ReplicationPad3d,
    ZeroPad1d,
    ZeroPad2d,
    ZeroPad3d,
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    CircularPad1d,
    CircularPad2d,
    CircularPad3d,
)

# optimizers
from neurova.nn.optim import (
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    NAdam,
    RAdam,
    ASGD,
    Rprop,
    LBFGS,
)

# loss Functions
from neurova.nn.loss import (
    MSELoss,
    L1Loss,
    CrossEntropyLoss,
    NLLLoss,
    BCELoss,
    BCEWithLogitsLoss,
    SmoothL1Loss,
    HuberLoss,
    KLDivLoss,
    PoissonNLLLoss,
    CTCLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
    CosineEmbeddingLoss,
    MarginRankingLoss,
    HingeEmbeddingLoss,
    MultiMarginLoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    SoftMarginLoss,
    FocalLoss,
    GaussianNLLLoss,
    DiceLoss,
    IoULoss,
    ContrastiveLoss,
)

# utilities - Functional API
from neurova.nn.functional import (
    relu,
    relu6,
    sigmoid,
    tanh,
    softmax,
    softmin,
    log_softmax,
    gelu,
    glu,
    selu,
    elu,
    celu,
    leaky_relu,
    silu,
    mish,
    hardtanh,
    hardswish,
    hardsigmoid,
    softplus,
    softshrink,
    softsign,
    tanhshrink,
    threshold,
    gumbel_softmax,
    conv2d,
    max_pool2d,
    avg_pool2d,
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
    batch_norm,
    layer_norm,
    group_norm,
    instance_norm,
    rms_norm,
    dropout,
    dropout2d,
    dropout3d,
    alpha_dropout,
    l1_loss,
    mse_loss,
    cross_entropy,
    nll_loss,
    binary_cross_entropy,
    binary_cross_entropy_with_logits,
    smooth_l1_loss,
    huber_loss,
    kl_div,
    triplet_margin_loss,
    cosine_embedding_loss,
    normalize,
    pairwise_distance,
    cosine_similarity,
    pdist,
    one_hot,
    embedding,
    pad,
    interpolate,
    pixel_shuffle,
    pixel_unshuffle,
    scaled_dot_product_attention,
    linear,
    bilinear,
)

# learning rate schedulers
from neurova.nn.scheduler import (
    _LRScheduler as LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    OneCycleLR,
    ReduceLROnPlateau,
    LinearLR,
    PolynomialLR,
    ConstantLR,
    SequentialLR,
    ChainedScheduler,
    LambdaLR,
    MultiplicativeLR,
)

# probability distributions
from neurova.nn.distributions import (
    Distribution,
    Normal,
    Uniform,
    Bernoulli,
    Categorical,
    Exponential,
    Gamma,
    Beta,
    Poisson,
    Binomial,
    Dirichlet,
    Laplace,
    Cauchy,
    StudentT,
    Chi2,
    LogNormal,
    Gumbel,
    Weibull,
    MultivariateNormal,
    VonMises,
    kl_divergence,
)

# linear algebra
from neurova.nn import linalg

# FFT operations
from neurova.nn import fft

# special functions
from neurova.nn import special

__all__ = [
    # autograd
    'Tensor',
    'Parameter',
    'no_grad',
    
    # core layers
    'Module',
    'Sequential',
    'ModuleList',
    'ModuleDict',
    
    # linear layers
    'Linear',
    'Bilinear',
    'Identity',
    
    # convolutional layers
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    
    # pooling layers
    'MaxPool1d',
    'MaxPool2d',
    'MaxPool3d',
    'AvgPool1d',
    'AvgPool2d',
    'AvgPool3d',
    'AdaptiveAvgPool1d',
    'AdaptiveAvgPool2d',
    'AdaptiveMaxPool1d',
    'AdaptiveMaxPool2d',
    
    # normalization layers
    'BatchNorm1d',
    'BatchNorm2d',
    'BatchNorm3d',
    'LayerNorm',
    'GroupNorm',
    'InstanceNorm1d',
    'InstanceNorm2d',
    
    # activation layers
    'ReLU',
    'LeakyReLU',
    'PReLU',
    'ELU',
    'SELU',
    'GELU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'LogSoftmax',
    'Swish',
    'Mish',
    'SiLU',
    'Softplus',
    'Softsign',
    'Hardtanh',
    'Hardswish',
    'Hardsigmoid',
    'ReLU6',
    'CELU',
    'GLU',
    'Threshold',
    'Softmin',
    'Softmax2d',
    
    # dropout layers
    'Dropout',
    'Dropout2d',
    'Dropout3d',
    'AlphaDropout',
    
    # recurrent layers
    'RNN',
    'LSTM',
    'GRU',
    'RNNCell',
    'LSTMCell',
    'GRUCell',
    
    # attention/Transformer layers
    'MultiheadAttention',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    
    # embedding layers
    'Embedding',
    'EmbeddingBag',
    
    # padding layers
    'ReflectionPad1d',
    'ReflectionPad2d',
    'ReflectionPad3d',
    'ReplicationPad1d',
    'ReplicationPad2d',
    'ReplicationPad3d',
    'ZeroPad1d',
    'ZeroPad2d',
    'ZeroPad3d',
    'ConstantPad1d',
    'ConstantPad2d',
    'ConstantPad3d',
    'CircularPad1d',
    'CircularPad2d',
    'CircularPad3d',
    
    # optimizers
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'Adamax',
    'NAdam',
    'RAdam',
    'ASGD',
    'Rprop',
    'LBFGS',
    
    # loss functions
    'MSELoss',
    'L1Loss',
    'CrossEntropyLoss',
    'NLLLoss',
    'BCELoss',
    'BCEWithLogitsLoss',
    'SmoothL1Loss',
    'HuberLoss',
    'KLDivLoss',
    'PoissonNLLLoss',
    'CTCLoss',
    'TripletMarginLoss',
    'TripletMarginWithDistanceLoss',
    'CosineEmbeddingLoss',
    'MarginRankingLoss',
    'HingeEmbeddingLoss',
    'MultiMarginLoss',
    'MultiLabelMarginLoss',
    'MultiLabelSoftMarginLoss',
    'SoftMarginLoss',
    'FocalLoss',
    'GaussianNLLLoss',
    'DiceLoss',
    'IoULoss',
    'ContrastiveLoss',
    
    # functional API
    'relu',
    'relu6',
    'sigmoid',
    'tanh',
    'softmax',
    'softmin',
    'log_softmax',
    'gelu',
    'glu',
    'selu',
    'elu',
    'celu',
    'leaky_relu',
    'silu',
    'mish',
    'hardtanh',
    'hardswish',
    'hardsigmoid',
    'softplus',
    'softshrink',
    'softsign',
    'tanhshrink',
    'threshold',
    'gumbel_softmax',
    'conv2d',
    'max_pool2d',
    'avg_pool2d',
    'adaptive_avg_pool2d',
    'adaptive_max_pool2d',
    'batch_norm',
    'layer_norm',
    'group_norm',
    'instance_norm',
    'rms_norm',
    'dropout',
    'dropout2d',
    'dropout3d',
    'alpha_dropout',
    'l1_loss',
    'mse_loss',
    'cross_entropy',
    'nll_loss',
    'binary_cross_entropy',
    'binary_cross_entropy_with_logits',
    'smooth_l1_loss',
    'huber_loss',
    'kl_div',
    'triplet_margin_loss',
    'cosine_embedding_loss',
    'normalize',
    'pairwise_distance',
    'cosine_similarity',
    'pdist',
    'one_hot',
    'embedding',
    'pad',
    'interpolate',
    'pixel_shuffle',
    'pixel_unshuffle',
    'scaled_dot_product_attention',
    'linear',
    'bilinear',
    
    # learning rate schedulers
    'LRScheduler',
    'StepLR',
    'MultiStepLR',
    'ExponentialLR',
    'CosineAnnealingLR',
    'CosineAnnealingWarmRestarts',
    'CyclicLR',
    'OneCycleLR',
    'ReduceLROnPlateau',
    'LinearLR',
    'PolynomialLR',
    'ConstantLR',
    'SequentialLR',
    'ChainedScheduler',
    'LambdaLR',
    'MultiplicativeLR',
    
    # probability distributions
    'Distribution',
    'Normal',
    'Uniform',
    'Bernoulli',
    'Categorical',
    'Exponential',
    'Gamma',
    'Beta',
    'Poisson',
    'Binomial',
    'Dirichlet',
    'Laplace',
    'Cauchy',
    'StudentT',
    'Chi2',
    'LogNormal',
    'Gumbel',
    'Weibull',
    'MultivariateNormal',
    'VonMises',
    'kl_divergence',
    
    # submodules
    'linalg',
    'fft',
    'special',
]
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.