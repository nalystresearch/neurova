# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Neurova neural.

NumPy-first deep learning primitives implemented inside Neurova.

This subpackage provides:
- Tensors with autograd
- Modules and parameters
- Basic layers
- Losses and optimizers

It intentionally stays small and dependency-light.
"""

from neurova.neural.tensor import Tensor, tensor
from neurova.neural.module import Module, Parameter
from neurova.neural import layers
from neurova.neural import losses
from neurova.neural import optim
from neurova.neural import conv
from neurova.neural.model_zoo import save_weights, load_weights, ModelZoo

__all__ = [
	"Tensor",
	"tensor",
	"Module",
	"Parameter",
	"layers",
	"losses",
	"optim",
	"conv",
	"save_weights",
	"load_weights",
	"ModelZoo",
]
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.