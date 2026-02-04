#!/usr/bin/env python
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Quick Phase 1 regression test."""

import sys
sys.path.insert(0, '/Users/harrythapa/Desktop/nalyst-research/Neurova')

import numpy as np

print('Testing Phase 1 Architectures...')

# Test imports
from neurova.architecture import (
    CNN, RNN, LSTM, GRU, Transformer, MLP, VAE, GAN,
    BaseArchitecture
)

# Test CNN
print('  Testing CNN...', end=' ')
cnn = CNN(input_shape=(3, 32, 32), output_shape=(10,))
x = np.random.randn(1, 3, 32, 32)
out = cnn._forward(x)
assert out.shape == (1, 10), f'CNN: {out.shape}'
print('OK')

# Test RNN
print('  Testing RNN...', end=' ')
rnn = RNN(input_shape=(10, 8), output_shape=(5,))
x = np.random.randn(1, 10, 8)
out = rnn._forward(x)
assert out.shape == (1, 5), f'RNN: {out.shape}'
print('OK')

# Test LSTM
print('  Testing LSTM...', end=' ')
lstm = LSTM(input_shape=(10, 8), output_shape=(5,))
x = np.random.randn(1, 10, 8)
out = lstm._forward(x)
assert out.shape == (1, 5), f'LSTM: {out.shape}'
print('OK')

# Test GRU
print('  Testing GRU...', end=' ')
gru = GRU(input_shape=(10, 8), output_shape=(5,))
x = np.random.randn(1, 10, 8)
out = gru._forward(x)
assert out.shape == (1, 5), f'GRU: {out.shape}'
print('OK')

# Test Transformer - need to match d_model (default 512)
print('  Testing Transformer...', end=' ')
transformer = Transformer(input_shape=(10, 512), output_shape=(5,))
x = np.random.randn(1, 10, 512)
out = transformer._forward(x)
assert out.shape == (1, 5), f'Transformer: {out.shape}'
print('OK')

# Test MLP
print('  Testing MLP...', end=' ')
mlp = MLP(input_shape=(100,), output_shape=(10,))
x = np.random.randn(1, 100)
out = mlp._forward(x)
assert out.shape == (1, 10), f'MLP: {out.shape}'
print('OK')

# Test VAE
print('  Testing VAE...', end=' ')
vae = VAE(input_shape=(784,), latent_dim=32)
x = np.random.randn(1, 784)
out = vae._forward(x)
assert out.shape == (1, 784), f'VAE: {out.shape}'
print('OK')

# Test GAN
print('  Testing GAN...', end=' ')
gan = GAN(input_shape=(784,), latent_dim=64)
print('OK')

print()
print('All Phase 1 tests passed!')
