# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.

"""Dropout, Recurrent, Attention, Embedding, Functional - Placeholders."""
from neurova.nn.layers import Module

class Dropout(Module):
    def __init__(self, p=0.5): super().__init__()
    def forward(self, x): return x
class Dropout2d(Module):
    def __init__(self, p=0.5): super().__init__()
    def forward(self, x): return x
class Dropout3d(Module):
    def __init__(self, p=0.5): super().__init__()
    def forward(self, x): return x
class AlphaDropout(Module):
    def __init__(self, p=0.5): super().__init__()
    def forward(self, x): return x
# copyright (c) 2025 @squid consultancy group (scg)
# all rights reserved.
# licensed under the mit license.