#!/usr/bin/env python
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Test script for Neurova new modules."""

import sys
import numpy as np

print('Testing Neurova nn module imports...')
print('=' * 50)

try:
    # Test core imports
    from neurova import nn
    print(' Core nn import')

    # Test functional API
    from neurova.nn import relu, gelu, silu, mish, scaled_dot_product_attention
    print(' Functional API imports')

    # Test schedulers
    from neurova.nn import StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
    print(' LR Schedulers imports')

    # Test distributions
    from neurova.nn import Normal, Uniform, Bernoulli, Categorical, MultivariateNormal
    print(' Distributions imports')

    # Test linalg submodule
    from neurova.nn import linalg
    print(' Linear algebra submodule')

    # Test fft submodule
    from neurova.nn import fft
    print(' FFT submodule')

    # Test special submodule
    from neurova.nn import special
    print(' Special functions submodule')

    print()
    print('Testing functionality...')
    print('=' * 50)

    # Test functional
    x = np.random.randn(2, 10)
    out = relu(x)
    print(f' relu: input {x.shape} -> output {out.shape}')

    out = gelu(x)
    print(f' gelu: input {x.shape} -> output {out.shape}')

    out = silu(x)
    print(f' silu: input {x.shape} -> output {out.shape}')

    out = mish(x)
    print(f' mish: input {x.shape} -> output {out.shape}')

    # Test scheduler
    class MockOptimizer:
        def __init__(self):
            self.lr = 0.1
            self.param_groups = [{'lr': 0.1}]
        def step(self):
            pass

    opt = MockOptimizer()
    scheduler = StepLR(opt, step_size=10, gamma=0.1)
    print(f' StepLR created with lr={scheduler.get_last_lr()}')

    scheduler2 = CosineAnnealingLR(opt, T_max=100)
    print(f' CosineAnnealingLR created')

    # Test distribution
    dist = Normal(0, 1)
    sample = dist.sample((5,))
    print(f' Normal distribution sample shape: {sample.shape}')

    dist2 = Uniform(0, 1)
    sample2 = dist2.sample((10,))
    print(f' Uniform distribution sample shape: {sample2.shape}')

    # Test linalg
    A = np.random.randn(3, 3)
    A = A @ A.T + 0.1 * np.eye(3)  # Make positive definite
    L = linalg.cholesky(A)
    print(f' Cholesky decomposition: {A.shape} -> {L.shape}')

    U, S, Vh = linalg.svd(A)
    print(f' SVD: {A.shape} -> U{U.shape}, S{S.shape}, Vh{Vh.shape}')

    det_val = linalg.det(A)
    print(f' Determinant: {det_val:.4f}')

    # Test fft
    x_fft = np.random.randn(64)
    X = fft.fft(x_fft)
    print(f' FFT: {x_fft.shape} -> {X.shape}')

    x_back = fft.ifft(X)
    print(f' IFFT: {X.shape} -> {x_back.shape}')

    # Test special functions
    from neurova.nn.special import gamma, erf, bessel_j0
    g = gamma(np.array([1, 2, 3, 4, 5]))
    print(f' Gamma([1,2,3,4,5]) = {g}')

    e = erf(np.array([0, 0.5, 1, 2]))
    print(f' Erf([0,0.5,1,2]) = {np.round(e, 4)}')

    print()
    print('=' * 50)
    print('ALL IMPORTS AND TESTS PASSED!')
    print('=' * 50)
    sys.exit(0)

except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
