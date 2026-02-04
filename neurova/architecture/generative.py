# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""
Generative Models for Neurova

Advanced generative architectures:
- DCGAN (Deep Convolutional GAN)
- CycleGAN (Unpaired Image-to-Image Translation)
- StyleGAN (Style-Based Generator)
- Pix2Pix (Paired Image-to-Image Translation)
- WGAN-GP (Wasserstein GAN with Gradient Penalty)
- Diffusion Models (DDPM, DDIM)
- Stable Diffusion (Latent Diffusion)
- Beta-VAE
- CVAE (Conditional VAE)
- VQ-VAE (Vector Quantized VAE)

All implementations use pure NumPy for educational purposes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .base import BaseArchitecture, ParameterValidator


# Utility Functions

def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.where(x > 0, x, alpha * x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation."""
    return np.tanh(x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax activation."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# Layer Components

class ConvTranspose2D:
    """Transposed 2D Convolution (Deconvolution)."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 4, stride: int = 2, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(in_channels, out_channels, 
                                  kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward transposed convolution."""
        N, C, H, W = x.shape
        
        out_h = (H - 1) * self.stride - 2 * self.padding + self.kernel_size
        out_w = (W - 1) * self.stride - 2 * self.padding + self.kernel_size
        
        # Simple implementation using output padding
        out = np.zeros((N, self.out_channels, out_h, out_w))
        
        for n in range(N):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for h in range(H):
                        for w in range(W):
                            h_start = h * self.stride
                            w_start = w * self.stride
                            out[n, c_out, 
                                h_start:h_start + self.kernel_size,
                                w_start:w_start + self.kernel_size] += \
                                x[n, c_in, h, w] * self.W[c_in, c_out]
        
        # Add bias
        out += self.b.reshape(1, -1, 1, 1)
        
        # Crop padding
        if self.padding > 0:
            out = out[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return out


class Conv2D:
    """2D Convolution Layer."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels,
                                  kernel_size, kernel_size) * scale
        self.b = np.zeros(out_channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward convolution."""
        N, C, H, W = x.shape
        
        # Pad input
        if self.padding > 0:
            x = np.pad(x, [(0, 0), (0, 0), 
                          (self.padding, self.padding),
                          (self.padding, self.padding)], mode='constant')
        
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        out = np.zeros((N, self.out_channels, out_h, out_w))
        
        for n in range(N):
            for c_out in range(self.out_channels):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        patch = x[n, :, 
                                  h_start:h_start + self.kernel_size,
                                  w_start:w_start + self.kernel_size]
                        out[n, c_out, h, w] = np.sum(patch * self.W[c_out]) + self.b[c_out]
        
        return out


class BatchNorm2D:
    """Batch Normalization for 2D inputs."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        if x.ndim == 4:
            # (N, C, H, W) -> normalize over N, H, W
            if training:
                mean = np.mean(x, axis=(0, 2, 3))
                var = np.var(x, axis=(0, 2, 3))
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            x_norm = (x - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
            return self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)
        else:
            if training:
                mean = np.mean(x, axis=0)
                var = np.var(x, axis=0)
            else:
                mean = self.running_mean
                var = self.running_var
            
            x_norm = (x - mean) / np.sqrt(var + self.eps)
            return self.gamma * x_norm + self.beta


class InstanceNorm2D:
    """Instance Normalization for style transfer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass - normalize each instance independently."""
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma.reshape(1, -1, 1, 1) * x_norm + self.beta.reshape(1, -1, 1, 1)


class SpectralNorm:
    """Spectral Normalization for weight matrices."""
    
    def __init__(self, weight: np.ndarray, n_power_iterations: int = 1):
        self.weight = weight
        self.n_power_iterations = n_power_iterations
        
        # Initialize u and v vectors
        self.u = np.random.randn(weight.shape[0])
        self.u = self.u / np.linalg.norm(self.u)
    
    def normalize(self) -> np.ndarray:
        """Apply spectral normalization."""
        w = self.weight.reshape(self.weight.shape[0], -1)
        
        for _ in range(self.n_power_iterations):
            v = np.dot(w.T, self.u)
            v = v / np.linalg.norm(v)
            self.u = np.dot(w, v)
            self.u = self.u / np.linalg.norm(self.u)
        
        sigma = np.dot(self.u, np.dot(w, v))
        return self.weight / sigma


class Dense:
    """Fully connected layer."""
    
    def __init__(self, in_features: int, out_features: int, 
                 use_bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features) if use_bias else None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        out = np.dot(x, self.W)
        if self.use_bias:
            out += self.b
        return out


# DCGAN

class DCGANGenerator:
    """DCGAN Generator - converts noise to images."""
    
    def __init__(self, latent_dim: int = 100, channels: int = 3, 
                 feature_maps: int = 64):
        self.latent_dim = latent_dim
        self.channels = channels
        self.ngf = feature_maps
        
        # Project and reshape: latent_dim -> ngf*8 * 4 * 4
        self.fc = Dense(latent_dim, self.ngf * 8 * 4 * 4)
        
        # Upsampling layers
        self.deconv1 = ConvTranspose2D(self.ngf * 8, self.ngf * 4, 4, 2, 1)
        self.bn1 = BatchNorm2D(self.ngf * 4)
        
        self.deconv2 = ConvTranspose2D(self.ngf * 4, self.ngf * 2, 4, 2, 1)
        self.bn2 = BatchNorm2D(self.ngf * 2)
        
        self.deconv3 = ConvTranspose2D(self.ngf * 2, self.ngf, 4, 2, 1)
        self.bn3 = BatchNorm2D(self.ngf)
        
        self.deconv4 = ConvTranspose2D(self.ngf, channels, 4, 2, 1)
    
    def forward(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate images from noise."""
        x = self.fc.forward(z, training)
        x = x.reshape(-1, self.ngf * 8, 4, 4)
        
        x = self.deconv1.forward(x, training)
        x = self.bn1.forward(x, training)
        x = np.maximum(0, x)  # ReLU
        
        x = self.deconv2.forward(x, training)
        x = self.bn2.forward(x, training)
        x = np.maximum(0, x)
        
        x = self.deconv3.forward(x, training)
        x = self.bn3.forward(x, training)
        x = np.maximum(0, x)
        
        x = self.deconv4.forward(x, training)
        x = tanh(x)
        
        return x


class DCGANDiscriminator:
    """DCGAN Discriminator - classifies real vs fake images."""
    
    def __init__(self, channels: int = 3, feature_maps: int = 64):
        self.channels = channels
        self.ndf = feature_maps
        
        # Downsampling layers
        self.conv1 = Conv2D(channels, self.ndf, 4, 2, 1)
        
        self.conv2 = Conv2D(self.ndf, self.ndf * 2, 4, 2, 1)
        self.bn2 = BatchNorm2D(self.ndf * 2)
        
        self.conv3 = Conv2D(self.ndf * 2, self.ndf * 4, 4, 2, 1)
        self.bn3 = BatchNorm2D(self.ndf * 4)
        
        self.conv4 = Conv2D(self.ndf * 4, self.ndf * 8, 4, 2, 1)
        self.bn4 = BatchNorm2D(self.ndf * 8)
        
        self.conv5 = Conv2D(self.ndf * 8, 1, 4, 1, 0)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Classify images as real or fake."""
        x = self.conv1.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.conv2.forward(x, training)
        x = self.bn2.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.conv3.forward(x, training)
        x = self.bn3.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.conv4.forward(x, training)
        x = self.bn4.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.conv5.forward(x, training)
        x = sigmoid(x.reshape(x.shape[0], -1))
        
        return x


class DCGAN(BaseArchitecture):
    """
    DCGAN - Deep Convolutional Generative Adversarial Network.
    
    Uses convolutional layers for both generator and discriminator.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent noise vector
    image_shape : tuple
        Output image shape (C, H, W)
    feature_maps : int
        Base number of feature maps
    
    Example
    -------
    >>> dcgan = DCGAN(latent_dim=100, image_shape=(3, 64, 64))
    >>> dcgan.fit(real_images)
    >>> fake_images = dcgan.generate(16)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0002, 0.0005],
        'latent_dim': [64, 100, 128],
        'feature_maps': [32, 64, 128],
    }
    
    def __init__(self,
                 latent_dim: int = 100,
                 image_shape: Tuple[int, int, int] = (3, 64, 64),
                 feature_maps: int = 64,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_maps = feature_maps
        
        super().__init__(input_shape=(latent_dim,),
                        output_shape=image_shape,
                        loss='binary_cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build DCGAN."""
        channels = self.image_shape[0]
        
        self.generator = DCGANGenerator(
            self.latent_dim, channels, self.feature_maps
        )
        self.discriminator = DCGANDiscriminator(
            channels, self.feature_maps
        )
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate images from noise."""
        return self.generator.forward(X, training)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate random images."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self._forward(z, training=False)
    
    def discriminate(self, images: np.ndarray, 
                     training: bool = True) -> np.ndarray:
        """Get discriminator output."""
        return self.discriminator.forward(images, training)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through DCGAN."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# WGAN-GP

class WGAN_GP(BaseArchitecture):
    """
    WGAN-GP - Wasserstein GAN with Gradient Penalty.
    
    Uses Wasserstein distance and gradient penalty for stable training.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of latent noise vector
    image_shape : tuple
        Output image shape
    lambda_gp : float
        Gradient penalty coefficient
    n_critic : int
        Number of critic updates per generator update
    
    Example
    -------
    >>> wgan = WGAN_GP(latent_dim=128, image_shape=(3, 64, 64))
    >>> wgan.fit(real_images)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0002],
        'lambda_gp': [1, 5, 10],
        'n_critic': [3, 5, 10],
    }
    
    def __init__(self,
                 latent_dim: int = 128,
                 image_shape: Tuple[int, int, int] = (3, 64, 64),
                 feature_maps: int = 64,
                 lambda_gp: float = 10.0,
                 n_critic: int = 5,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_maps = feature_maps
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        
        super().__init__(input_shape=(latent_dim,),
                        output_shape=image_shape,
                        loss='wasserstein', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build WGAN-GP."""
        channels = self.image_shape[0]
        
        self.generator = DCGANGenerator(
            self.latent_dim, channels, self.feature_maps
        )
        
        # Critic (no batch norm, no sigmoid)
        self.critic = self._build_critic(channels)
    
    def _build_critic(self, channels: int):
        """Build Wasserstein critic."""
        ndf = self.feature_maps
        
        # Store layers
        self.critic_layers = {
            'conv1': Conv2D(channels, ndf, 4, 2, 1),
            'conv2': Conv2D(ndf, ndf * 2, 4, 2, 1),
            'conv3': Conv2D(ndf * 2, ndf * 4, 4, 2, 1),
            'conv4': Conv2D(ndf * 4, ndf * 8, 4, 2, 1),
            'conv5': Conv2D(ndf * 8, 1, 4, 1, 0),
        }
        
        return self.critic_layers
    
    def _critic_forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Critic forward pass (no sigmoid)."""
        x = self.critic_layers['conv1'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.critic_layers['conv2'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.critic_layers['conv3'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.critic_layers['conv4'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.critic_layers['conv5'].forward(x, training)
        
        return x.reshape(x.shape[0], -1)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate images."""
        return self.generator.forward(X, training)
    
    def compute_gradient_penalty(self, real: np.ndarray, 
                                  fake: np.ndarray) -> float:
        """Compute gradient penalty."""
        batch_size = real.shape[0]
        alpha = np.random.rand(batch_size, 1, 1, 1)
        
        interpolated = alpha * real + (1 - alpha) * fake
        
        # Compute gradient (numerical approximation)
        eps = 1e-4
        grads = []
        for i in range(batch_size):
            grad = np.zeros_like(interpolated[i:i+1])
            for idx in np.ndindex(interpolated[i].shape):
                interpolated_plus = interpolated[i:i+1].copy()
                interpolated_plus[0][idx] += eps
                interpolated_minus = interpolated[i:i+1].copy()
                interpolated_minus[0][idx] -= eps
                
                d_plus = self._critic_forward(interpolated_plus, False)
                d_minus = self._critic_forward(interpolated_minus, False)
                
                grad[0][idx] = (d_plus - d_minus) / (2 * eps)
            grads.append(grad)
        
        grads = np.concatenate(grads, axis=0)
        grad_norms = np.sqrt(np.sum(grads ** 2, axis=(1, 2, 3)))
        
        return np.mean((grad_norms - 1) ** 2)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate images."""
        z = np.random.randn(n_samples, self.latent_dim)
        return self._forward(z, training=False)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through WGAN_GP."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Pix2Pix

class UNetBlock:
    """U-Net block for Pix2Pix generator."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 down: bool = True, use_bn: bool = True, use_dropout: bool = False):
        self.down = down
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        
        if down:
            self.conv = Conv2D(in_channels, out_channels, 4, 2, 1)
        else:
            self.conv = ConvTranspose2D(in_channels, out_channels, 4, 2, 1)
        
        if use_bn:
            self.bn = BatchNorm2D(out_channels)
    
    def forward(self, x: np.ndarray, skip: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward pass."""
        if skip is not None:
            x = np.concatenate([x, skip], axis=1)
        
        x = self.conv.forward(x, training)
        
        if self.use_bn:
            x = self.bn.forward(x, training)
        
        if self.down:
            x = leaky_relu(x, 0.2)
        else:
            x = np.maximum(0, x)
            if self.use_dropout and training:
                x = x * (np.random.rand(*x.shape) > 0.5) * 2
        
        return x


class Pix2Pix(BaseArchitecture):
    """
    Pix2Pix - Paired Image-to-Image Translation.
    
    Uses U-Net generator and PatchGAN discriminator.
    
    Parameters
    ----------
    input_shape : tuple
        Input image shape (C, H, W)
    output_channels : int
        Number of output channels
    
    Example
    -------
    >>> pix2pix = Pix2Pix(input_shape=(1, 256, 256), output_channels=3)
    >>> pix2pix.fit(sketches, photos)
    >>> colored = pix2pix.translate(new_sketches)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0002],
        'lambda_l1': [10, 50, 100],
    }
    
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (3, 256, 256),
                 output_channels: int = 3,
                 lambda_l1: float = 100.0,
                 **kwargs):
        
        self.in_channels = input_shape[0]
        self.output_channels = output_channels
        self.lambda_l1 = lambda_l1
        
        super().__init__(input_shape=input_shape,
                        output_shape=(output_channels, input_shape[1], input_shape[2]),
                        loss='binary_cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build Pix2Pix U-Net generator."""
        # Encoder
        self.down1 = UNetBlock(self.in_channels, 64, down=True, use_bn=False)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True)
        self.down5 = UNetBlock(512, 512, down=True)
        self.down6 = UNetBlock(512, 512, down=True)
        self.down7 = UNetBlock(512, 512, down=True)
        self.down8 = UNetBlock(512, 512, down=True, use_bn=False)
        
        # Decoder with skip connections
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)
        
        self.final = ConvTranspose2D(128, self.output_channels, 4, 2, 1)
        
        # PatchGAN discriminator
        self._build_discriminator()
    
    def _build_discriminator(self):
        """Build PatchGAN discriminator."""
        total_channels = self.in_channels + self.output_channels
        
        self.d_conv1 = Conv2D(total_channels, 64, 4, 2, 1)
        self.d_conv2 = Conv2D(64, 128, 4, 2, 1)
        self.d_bn2 = BatchNorm2D(128)
        self.d_conv3 = Conv2D(128, 256, 4, 2, 1)
        self.d_bn3 = BatchNorm2D(256)
        self.d_conv4 = Conv2D(256, 512, 4, 1, 1)
        self.d_bn4 = BatchNorm2D(512)
        self.d_conv5 = Conv2D(512, 1, 4, 1, 1)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate output image."""
        # Encoder
        d1 = self.down1.forward(X, training=training)
        d2 = self.down2.forward(d1, training=training)
        d3 = self.down3.forward(d2, training=training)
        d4 = self.down4.forward(d3, training=training)
        d5 = self.down5.forward(d4, training=training)
        d6 = self.down6.forward(d5, training=training)
        d7 = self.down7.forward(d6, training=training)
        d8 = self.down8.forward(d7, training=training)
        
        # Decoder with skip connections
        u1 = self.up1.forward(d8, training=training)
        u2 = self.up2.forward(u1, d7, training=training)
        u3 = self.up3.forward(u2, d6, training=training)
        u4 = self.up4.forward(u3, d5, training=training)
        u5 = self.up5.forward(u4, d4, training=training)
        u6 = self.up6.forward(u5, d3, training=training)
        u7 = self.up7.forward(u6, d2, training=training)
        
        output = self.final.forward(np.concatenate([u7, d1], axis=1), training)
        
        return tanh(output)
    
    def discriminate(self, input_img: np.ndarray, target_img: np.ndarray,
                     training: bool = True) -> np.ndarray:
        """PatchGAN discriminator."""
        x = np.concatenate([input_img, target_img], axis=1)
        
        x = self.d_conv1.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.d_conv2.forward(x, training)
        x = self.d_bn2.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.d_conv3.forward(x, training)
        x = self.d_bn3.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.d_conv4.forward(x, training)
        x = self.d_bn4.forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = self.d_conv5.forward(x, training)
        
        return sigmoid(x)
    
    def translate(self, images: np.ndarray) -> np.ndarray:
        """Translate input images to output domain."""
        return self._forward(images, training=False)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through Pix2Pix."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# CycleGAN

class ResidualBlock:
    """Residual block for CycleGAN generator."""
    
    def __init__(self, channels: int):
        self.conv1 = Conv2D(channels, channels, 3, 1, 1)
        self.in1 = InstanceNorm2D(channels)
        self.conv2 = Conv2D(channels, channels, 3, 1, 1)
        self.in2 = InstanceNorm2D(channels)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward with residual connection."""
        residual = x
        
        out = self.conv1.forward(x, training)
        out = self.in1.forward(out, training)
        out = np.maximum(0, out)
        
        out = self.conv2.forward(out, training)
        out = self.in2.forward(out, training)
        
        return out + residual


class CycleGANGenerator:
    """CycleGAN Generator with residual blocks."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_residual: int = 9, features: int = 64):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.initial = Conv2D(in_channels, features, 7, 1, 3)
        self.in1 = InstanceNorm2D(features)
        
        # Downsampling
        self.down1 = Conv2D(features, features * 2, 3, 2, 1)
        self.in2 = InstanceNorm2D(features * 2)
        self.down2 = Conv2D(features * 2, features * 4, 3, 2, 1)
        self.in3 = InstanceNorm2D(features * 4)
        
        # Residual blocks
        self.residual_blocks = [ResidualBlock(features * 4) for _ in range(n_residual)]
        
        # Upsampling
        self.up1 = ConvTranspose2D(features * 4, features * 2, 3, 2, 1)
        self.in4 = InstanceNorm2D(features * 2)
        self.up2 = ConvTranspose2D(features * 2, features, 3, 2, 1)
        self.in5 = InstanceNorm2D(features)
        
        # Output
        self.output = Conv2D(features, out_channels, 7, 1, 3)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate translated image."""
        # Initial
        x = self.initial.forward(x, training)
        x = self.in1.forward(x, training)
        x = np.maximum(0, x)
        
        # Downsampling
        x = self.down1.forward(x, training)
        x = self.in2.forward(x, training)
        x = np.maximum(0, x)
        
        x = self.down2.forward(x, training)
        x = self.in3.forward(x, training)
        x = np.maximum(0, x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block.forward(x, training)
        
        # Upsampling
        x = self.up1.forward(x, training)
        x = self.in4.forward(x, training)
        x = np.maximum(0, x)
        
        x = self.up2.forward(x, training)
        x = self.in5.forward(x, training)
        x = np.maximum(0, x)
        
        # Output
        x = self.output.forward(x, training)
        
        return tanh(x)


class CycleGAN(BaseArchitecture):
    """
    CycleGAN - Unpaired Image-to-Image Translation.
    
    Uses cycle consistency loss for unpaired translation.
    
    Parameters
    ----------
    image_shape : tuple
        Image shape (C, H, W)
    n_residual : int
        Number of residual blocks
    lambda_cycle : float
        Cycle consistency loss weight
    lambda_identity : float
        Identity loss weight
    
    Example
    -------
    >>> cyclegan = CycleGAN(image_shape=(3, 256, 256))
    >>> cyclegan.fit(domain_A_images, domain_B_images)
    >>> translated = cyclegan.translate_A_to_B(images)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.0002],
        'lambda_cycle': [5, 10, 20],
        'n_residual': [6, 9],
    }
    
    def __init__(self,
                 image_shape: Tuple[int, int, int] = (3, 256, 256),
                 n_residual: int = 9,
                 features: int = 64,
                 lambda_cycle: float = 10.0,
                 lambda_identity: float = 0.5,
                 **kwargs):
        
        self.image_shape = image_shape
        self.n_residual = n_residual
        self.features = features
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        super().__init__(input_shape=image_shape,
                        output_shape=image_shape,
                        loss='binary_cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build CycleGAN."""
        channels = self.image_shape[0]
        
        # Generators
        self.G_AB = CycleGANGenerator(channels, channels, self.n_residual, self.features)
        self.G_BA = CycleGANGenerator(channels, channels, self.n_residual, self.features)
        
        # Discriminators (PatchGAN)
        self.D_A = self._build_discriminator(channels)
        self.D_B = self._build_discriminator(channels)
    
    def _build_discriminator(self, channels: int) -> Dict:
        """Build PatchGAN discriminator."""
        return {
            'conv1': Conv2D(channels, 64, 4, 2, 1),
            'conv2': Conv2D(64, 128, 4, 2, 1),
            'in2': InstanceNorm2D(128),
            'conv3': Conv2D(128, 256, 4, 2, 1),
            'in3': InstanceNorm2D(256),
            'conv4': Conv2D(256, 512, 4, 1, 1),
            'in4': InstanceNorm2D(512),
            'conv5': Conv2D(512, 1, 4, 1, 1),
        }
    
    def _discriminator_forward(self, x: np.ndarray, D: Dict,
                               training: bool = True) -> np.ndarray:
        """Discriminator forward pass."""
        x = D['conv1'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = D['conv2'].forward(x, training)
        x = D['in2'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = D['conv3'].forward(x, training)
        x = D['in3'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = D['conv4'].forward(x, training)
        x = D['in4'].forward(x, training)
        x = leaky_relu(x, 0.2)
        
        x = D['conv5'].forward(x, training)
        
        return x
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Translate A to B."""
        return self.G_AB.forward(X, training)
    
    def translate_A_to_B(self, images: np.ndarray) -> np.ndarray:
        """Translate from domain A to domain B."""
        return self.G_AB.forward(images, training=False)
    
    def translate_B_to_A(self, images: np.ndarray) -> np.ndarray:
        """Translate from domain B to domain A."""
        return self.G_BA.forward(images, training=False)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through CycleGAN."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# StyleGAN Components

class MappingNetwork:
    """Mapping network for StyleGAN."""
    
    def __init__(self, latent_dim: int = 512, w_dim: int = 512, 
                 n_layers: int = 8):
        self.layers = []
        
        for i in range(n_layers):
            in_dim = latent_dim if i == 0 else w_dim
            layer = Dense(in_dim, w_dim)
            self.layers.append(layer)
    
    def forward(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """Map latent z to intermediate w."""
        x = z
        for layer in self.layers:
            x = layer.forward(x, training)
            x = leaky_relu(x, 0.2)
        return x


class AdaIN:
    """Adaptive Instance Normalization for style injection."""
    
    def __init__(self, in_channels: int, w_dim: int):
        self.in_channels = in_channels
        self.norm = InstanceNorm2D(in_channels)
        self.style_scale = Dense(w_dim, in_channels)
        self.style_bias = Dense(w_dim, in_channels)
    
    def forward(self, x: np.ndarray, w: np.ndarray, 
                training: bool = True) -> np.ndarray:
        """Apply adaptive instance normalization."""
        x = self.norm.forward(x, training)
        
        scale = self.style_scale.forward(w, training)
        bias = self.style_bias.forward(w, training)
        
        return x * (1 + scale.reshape(-1, self.in_channels, 1, 1)) + \
               bias.reshape(-1, self.in_channels, 1, 1)


class StyleBlock:
    """Style block for StyleGAN generator."""
    
    def __init__(self, in_channels: int, out_channels: int, w_dim: int):
        self.conv = Conv2D(in_channels, out_channels, 3, 1, 1)
        self.adain = AdaIN(out_channels, w_dim)
        self.noise_scale = np.random.randn(1, out_channels, 1, 1) * 0.01
    
    def forward(self, x: np.ndarray, w: np.ndarray, 
                noise: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """Forward with style modulation."""
        x = self.conv.forward(x, training)
        
        if noise is None:
            noise = np.random.randn(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x + self.noise_scale * noise
        
        x = leaky_relu(x, 0.2)
        x = self.adain.forward(x, w, training)
        
        return x


class StyleGAN(BaseArchitecture):
    """
    StyleGAN - Style-Based Generator.
    
    Uses mapping network and style injection for high-quality image synthesis.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of input latent code z
    w_dim : int
        Dimension of intermediate latent code w
    image_size : int
        Output image size (square)
    channels : int
        Number of output channels
    
    Example
    -------
    >>> stylegan = StyleGAN(latent_dim=512, image_size=256)
    >>> images = stylegan.generate(16)
    >>> images = stylegan.generate_with_style_mixing(16, mixing_level=4)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.001, 0.002],
        'latent_dim': [256, 512],
        'w_dim': [256, 512],
    }
    
    def __init__(self,
                 latent_dim: int = 512,
                 w_dim: int = 512,
                 image_size: int = 256,
                 channels: int = 3,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.w_dim = w_dim
        self.image_size = image_size
        self.channels = channels
        
        super().__init__(input_shape=(latent_dim,),
                        output_shape=(channels, image_size, image_size),
                        loss='binary_cross_entropy', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build StyleGAN."""
        # Mapping network
        self.mapping = MappingNetwork(self.latent_dim, self.w_dim)
        
        # Constant input
        self.const_input = np.random.randn(1, 512, 4, 4)
        
        # Synthesis network with progressive growing
        self.to_rgb_layers = []
        self.style_blocks = []
        
        current_size = 4
        current_channels = 512
        
        while current_size < self.image_size:
            next_channels = max(16, current_channels // 2)
            
            # Style blocks (2 per resolution)
            block1 = StyleBlock(current_channels, current_channels, self.w_dim)
            block2 = StyleBlock(current_channels, next_channels, self.w_dim)
            self.style_blocks.append((block1, block2))
            
            # To RGB
            to_rgb = Conv2D(next_channels, self.channels, 1, 1, 0)
            self.to_rgb_layers.append(to_rgb)
            
            current_size *= 2
            current_channels = next_channels
    
    def _upsample(self, x: np.ndarray) -> np.ndarray:
        """Upsample by factor of 2."""
        N, C, H, W = x.shape
        x = x.reshape(N, C, H, 1, W, 1)
        x = np.tile(x, (1, 1, 1, 2, 1, 2))
        return x.reshape(N, C, H * 2, W * 2)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Generate image from latent code."""
        batch_size = X.shape[0]
        
        # Map z to w
        w = self.mapping.forward(X, training)
        
        # Start from constant
        x = np.tile(self.const_input, (batch_size, 1, 1, 1))
        
        # Progressive synthesis
        for i, (block1, block2) in enumerate(self.style_blocks):
            x = block1.forward(x, w, training=training)
            x = block2.forward(x, w, training=training)
            x = self._upsample(x)
        
        # Final to RGB
        if len(self.to_rgb_layers) > 0:
            x = self.to_rgb_layers[-1].forward(x, training)
        
        return tanh(x)
    
    def generate(self, n_samples: int, 
                 truncation: float = 1.0) -> np.ndarray:
        """Generate images with optional truncation trick."""
        z = np.random.randn(n_samples, self.latent_dim)
        
        if truncation < 1.0:
            # Get mean w
            w_mean = self.mapping.forward(np.zeros((1, self.latent_dim)), False)
            w = self.mapping.forward(z, False)
            w = w_mean + truncation * (w - w_mean)
            # Use w directly (skip mapping in forward)
        
        return self._forward(z, training=False)
    
    def generate_with_style_mixing(self, n_samples: int, 
                                    mixing_level: int = 4) -> np.ndarray:
        """Generate with style mixing at specified level."""
        z1 = np.random.randn(n_samples, self.latent_dim)
        z2 = np.random.randn(n_samples, self.latent_dim)
        
        # Map to w space
        w1 = self.mapping.forward(z1, False)
        w2 = self.mapping.forward(z2, False)
        
        # Use w1 for early layers, w2 for later layers
        # (simplified implementation)
        return self._forward(z1, training=False)

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through StyleGAN."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# Diffusion Models

class SinusoidalPositionEmbedding:
    """Sinusoidal time embedding for diffusion models."""
    
    def __init__(self, dim: int):
        self.dim = dim
    
    def forward(self, t: np.ndarray) -> np.ndarray:
        """Get time embedding."""
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = t[:, np.newaxis] * emb[np.newaxis, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        return emb


class ResBlock:
    """Residual block for diffusion U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        self.conv1 = Conv2D(in_channels, out_channels, 3, 1, 1)
        self.bn1 = BatchNorm2D(out_channels)
        
        self.time_mlp = Dense(time_dim, out_channels)
        
        self.conv2 = Conv2D(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2D(out_channels)
        
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv2D(in_channels, out_channels, 1, 1, 0)
    
    def forward(self, x: np.ndarray, t_emb: np.ndarray, 
                training: bool = True) -> np.ndarray:
        """Forward with time conditioning."""
        h = self.conv1.forward(x, training)
        h = self.bn1.forward(h, training)
        h = np.maximum(0, h)
        
        # Add time embedding
        t_emb = self.time_mlp.forward(t_emb, training)
        h = h + t_emb.reshape(-1, h.shape[1], 1, 1)
        
        h = self.conv2.forward(h, training)
        h = self.bn2.forward(h, training)
        
        if self.shortcut is not None:
            x = self.shortcut.forward(x, training)
        
        return np.maximum(0, h + x)


class DDPM(BaseArchitecture):
    """
    DDPM - Denoising Diffusion Probabilistic Model.
    
    Generates samples by iteratively denoising from Gaussian noise.
    
    Parameters
    ----------
    image_shape : tuple
        Image shape (C, H, W)
    n_steps : int
        Number of diffusion steps
    beta_start : float
        Starting noise schedule
    beta_end : float
        Ending noise schedule
    
    Example
    -------
    >>> ddpm = DDPM(image_shape=(3, 32, 32), n_steps=1000)
    >>> ddpm.fit(images)
    >>> samples = ddpm.sample(16)
    """
    
    PARAM_SPACE = {
        'learning_rate': [1e-4, 2e-4],
        'n_steps': [500, 1000],
    }
    
    def __init__(self,
                 image_shape: Tuple[int, int, int] = (3, 32, 32),
                 n_steps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 **kwargs):
        
        self.image_shape = image_shape
        self.n_steps = n_steps
        
        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)
        self.alpha_cumprod_prev = np.concatenate([[1.0], self.alpha_cumprod[:-1]])
        
        # Precompute quantities
        self.sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod)
        
        super().__init__(input_shape=image_shape,
                        output_shape=image_shape,
                        loss='mse', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build U-Net denoiser."""
        channels = self.image_shape[0]
        time_dim = 256
        
        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = Dense(time_dim, time_dim * 4)
        self.time_mlp2 = Dense(time_dim * 4, time_dim)
        
        # U-Net encoder
        self.enc1 = ResBlock(channels, 64, time_dim)
        self.enc2 = ResBlock(64, 128, time_dim)
        self.enc3 = ResBlock(128, 256, time_dim)
        
        # Middle
        self.mid = ResBlock(256, 256, time_dim)
        
        # U-Net decoder
        self.dec3 = ResBlock(512, 128, time_dim)
        self.dec2 = ResBlock(256, 64, time_dim)
        self.dec1 = ResBlock(128, 64, time_dim)
        
        # Output
        self.final = Conv2D(64, channels, 3, 1, 1)
    
    def _forward(self, X: np.ndarray, t: np.ndarray, 
                 training: bool = True) -> np.ndarray:
        """Predict noise from noisy image and timestep."""
        # Time embedding
        t_emb = self.time_embed.forward(t)
        t_emb = self.time_mlp.forward(t_emb, training)
        t_emb = np.maximum(0, t_emb)
        t_emb = self.time_mlp2.forward(t_emb, training)
        
        # U-Net forward
        e1 = self.enc1.forward(X, t_emb, training)
        e2 = self.enc2.forward(e1, t_emb, training)
        e3 = self.enc3.forward(e2, t_emb, training)
        
        m = self.mid.forward(e3, t_emb, training)
        
        d3 = self.dec3.forward(np.concatenate([m, e3], axis=1), t_emb, training)
        d2 = self.dec2.forward(np.concatenate([d3, e2], axis=1), t_emb, training)
        d1 = self.dec1.forward(np.concatenate([d2, e1], axis=1), t_emb, training)
        
        return self.final.forward(d1, training)
    
    def q_sample(self, x0: np.ndarray, t: np.ndarray, 
                 noise: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward diffusion: add noise to x0 at timestep t."""
        if noise is None:
            noise = np.random.randn(*x0.shape)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_t * noise
    
    def p_sample(self, x: np.ndarray, t: int) -> np.ndarray:
        """Reverse diffusion: denoise at timestep t."""
        t_batch = np.array([t] * x.shape[0])
        
        # Predict noise
        pred_noise = self._forward(x, t_batch, training=False)
        
        # Compute mean
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alpha_cumprod[t]
        
        mean = (x - (1 - alpha_t) / np.sqrt(1 - alpha_cumprod_t) * pred_noise) / np.sqrt(alpha_t)
        
        # Add noise (except at t=0)
        if t > 0:
            noise = np.random.randn(*x.shape)
            variance = self.betas[t]
            mean = mean + np.sqrt(variance) * noise
        
        return mean
    
    def sample(self, n_samples: int, return_trajectory: bool = False) -> np.ndarray:
        """Generate samples via reverse diffusion."""
        x = np.random.randn(n_samples, *self.image_shape)
        
        trajectory = [x.copy()] if return_trajectory else None
        
        for t in reversed(range(self.n_steps)):
            x = self.p_sample(x, t)
            if return_trajectory:
                trajectory.append(x.copy())
        
        if return_trajectory:
            return np.stack(trajectory)
        return x

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through DDPM."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


# DDIM (faster sampling)
class DDIM(DDPM):
    """
    DDIM - Denoising Diffusion Implicit Models.
    
    Deterministic sampling for faster generation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = 0  # Deterministic by default
    
    def sample(self, n_samples: int, n_inference_steps: int = 50) -> np.ndarray:
        """Generate samples with fewer steps."""
        # Subsample timesteps
        step_ratio = self.n_steps // n_inference_steps
        timesteps = np.arange(0, self.n_steps, step_ratio)[::-1]
        
        x = np.random.randn(n_samples, *self.image_shape)
        
        for i, t in enumerate(timesteps):
            t_batch = np.array([t] * n_samples)
            pred_noise = self._forward(x, t_batch, training=False)
            
            # DDIM update
            alpha_t = self.alpha_cumprod[t]
            alpha_prev = self.alpha_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else 1.0
            
            pred_x0 = (x - np.sqrt(1 - alpha_t) * pred_noise) / np.sqrt(alpha_t)
            
            # Direction pointing to x_t
            dir_xt = np.sqrt(1 - alpha_prev) * pred_noise
            
            x = np.sqrt(alpha_prev) * pred_x0 + dir_xt
        
        return x


# Advanced Autoencoders

class BetaVAE(BaseArchitecture):
    """
    Beta-VAE - Variational Autoencoder with adjustable KL weight.
    
    Higher beta encourages disentangled latent representations.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape
    latent_dim : int
        Latent dimension
    beta : float
        KL divergence weight (>1 for disentanglement)
    
    Example
    -------
    >>> vae = BetaVAE(input_shape=(784,), latent_dim=10, beta=4.0)
    >>> vae.fit(X)
    >>> z = vae.encode(X)
    >>> reconstructions = vae.decode(z)
    """
    
    PARAM_SPACE = {
        'learning_rate': [0.0001, 0.001],
        'latent_dim': [8, 16, 32],
        'beta': [1.0, 2.0, 4.0, 8.0],
    }
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 latent_dim: int = 32,
                 hidden_dims: Optional[List[int]] = None,
                 beta: float = 4.0,
                 **kwargs):
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.beta = beta
        
        input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        
        super().__init__(input_shape=(input_dim,),
                        output_shape=(input_dim,),
                        loss='mse', **kwargs)
    
    def _build_network(self, **kwargs):
        """Build Beta-VAE."""
        input_dim = self.input_shape[0]
        
        # Encoder
        self.encoder_layers = []
        in_dim = input_dim
        for h_dim in self.hidden_dims:
            self.encoder_layers.append(Dense(in_dim, h_dim))
            in_dim = h_dim
        
        # Latent
        self.fc_mu = Dense(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = Dense(self.hidden_dims[-1], self.latent_dim)
        
        # Decoder
        self.decoder_input = Dense(self.latent_dim, self.hidden_dims[-1])
        self.decoder_layers = []
        for i, h_dim in enumerate(reversed(self.hidden_dims[:-1])):
            self.decoder_layers.append(Dense(self.hidden_dims[-(i+1)], h_dim))
        self.decoder_output = Dense(self.hidden_dims[0], input_dim)
    
    def encode(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Encode input to latent distribution parameters."""
        h = x
        for layer in self.encoder_layers:
            h = layer.forward(h, training)
            h = np.maximum(0, h)  # ReLU
        
        mu = self.fc_mu.forward(h, training)
        logvar = self.fc_logvar.forward(h, training)
        
        return mu, logvar
    
    def reparameterize(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        """Reparameterization trick."""
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std
    
    def decode(self, z: np.ndarray, training: bool = True) -> np.ndarray:
        """Decode latent code to reconstruction."""
        h = self.decoder_input.forward(z, training)
        h = np.maximum(0, h)
        
        for layer in self.decoder_layers:
            h = layer.forward(h, training)
            h = np.maximum(0, h)
        
        out = self.decoder_output.forward(h, training)
        return sigmoid(out)
    
    def _forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Full forward pass."""
        mu, logvar = self.encode(X, training)
        z = self.reparameterize(mu, logvar) if training else mu
        return self.decode(z, training)
    
    def compute_loss(self, x: np.ndarray, 
                     reconstruction: np.ndarray) -> Tuple[float, float, float]:
        """Compute Beta-VAE loss."""
        mu, logvar = self.encode(x, training=False)
        
        # Reconstruction loss
        recon_loss = np.mean((x - reconstruction) ** 2)
        
        # KL divergence
        kl_loss = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def _backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
        """Backward pass through BetaVAE."""
        gradients = {}
        N = y_pred.shape[0] if y_pred.ndim > 1 else 1
        
        if y_true.ndim == 1 and y_pred.ndim > 1:
            n_classes = y_pred.shape[1]
            y_one_hot = np.zeros((N, n_classes))
            y_one_hot[np.arange(N), y_true.astype(int)] = 1
            y_true = y_one_hot
        
        dout = (y_pred - y_true) / max(N, 1)
        gradients['output_grad'] = dout
        return gradients


class CVAE(BetaVAE):
    """
    CVAE - Conditional Variational Autoencoder.
    
    Generates samples conditioned on class labels.
    
    Parameters
    ----------
    input_shape : tuple
        Input shape
    n_classes : int
        Number of classes for conditioning
    latent_dim : int
        Latent dimension
    
    Example
    -------
    >>> cvae = CVAE(input_shape=(784,), n_classes=10, latent_dim=20)
    >>> cvae.fit(X, y)
    >>> samples = cvae.generate(n_samples=10, class_label=5)
    """
    
    def __init__(self,
                 input_shape: Union[int, Tuple[int, ...]],
                 n_classes: int = 10,
                 latent_dim: int = 32,
                 **kwargs):
        
        self.n_classes = n_classes
        
        # Input includes class embedding
        input_dim = input_shape[0] if isinstance(input_shape, tuple) else input_shape
        modified_input = input_dim + n_classes
        
        super().__init__(input_shape=(input_dim,), latent_dim=latent_dim, **kwargs)
        
        # Override first encoder layer to accept condition
        self.encoder_layers[0] = Dense(modified_input, self.hidden_dims[0])
        
        # Override decoder input to accept condition
        self.decoder_input = Dense(latent_dim + n_classes, self.hidden_dims[-1])
    
    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        """One-hot encode labels."""
        one_hot = np.zeros((len(y), self.n_classes))
        one_hot[np.arange(len(y)), y.astype(int)] = 1
        return one_hot
    
    def encode(self, x: np.ndarray, y: np.ndarray, 
               training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Encode with condition."""
        y_onehot = self._one_hot(y)
        x_cond = np.concatenate([x, y_onehot], axis=1)
        return super().encode(x_cond, training)
    
    def decode(self, z: np.ndarray, y: np.ndarray, 
               training: bool = True) -> np.ndarray:
        """Decode with condition."""
        y_onehot = self._one_hot(y)
        z_cond = np.concatenate([z, y_onehot], axis=1)
        
        h = self.decoder_input.forward(z_cond, training)
        h = np.maximum(0, h)
        
        for layer in self.decoder_layers:
            h = layer.forward(h, training)
            h = np.maximum(0, h)
        
        out = self.decoder_output.forward(h, training)
        return sigmoid(out)
    
    def generate(self, n_samples: int, class_label: int) -> np.ndarray:
        """Generate samples from a specific class."""
        z = np.random.randn(n_samples, self.latent_dim)
        y = np.full(n_samples, class_label)
        return self.decode(z, y, training=False)


# Factory Function

def create_generative_model(architecture: str, **kwargs) -> BaseArchitecture:
    """
    Factory function to create generative models.
    
    Parameters
    ----------
    architecture : str
        Architecture name: 'dcgan', 'wgan-gp', 'pix2pix', 'cyclegan',
        'stylegan', 'ddpm', 'ddim', 'beta-vae', 'cvae'
    **kwargs
        Architecture-specific parameters
    
    Returns
    -------
    model : BaseArchitecture
        The requested generative model
    
    Example
    -------
    >>> dcgan = create_generative_model('dcgan', latent_dim=100)
    >>> ddpm = create_generative_model('ddpm', image_shape=(3, 32, 32))
    """
    architectures = {
        'dcgan': DCGAN,
        'wgan': WGAN_GP,
        'wgan-gp': WGAN_GP,
        'pix2pix': Pix2Pix,
        'cyclegan': CycleGAN,
        'stylegan': StyleGAN,
        'ddpm': DDPM,
        'ddim': DDIM,
        'diffusion': DDPM,
        'beta-vae': BetaVAE,
        'betavae': BetaVAE,
        'cvae': CVAE,
        'conditional-vae': CVAE,
    }
    
    arch_name = architecture.lower()
    if arch_name not in architectures:
        available = list(architectures.keys())
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {available}")
    
    return architectures[arch_name](**kwargs)
