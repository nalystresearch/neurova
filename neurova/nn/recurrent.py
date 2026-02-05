# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Recurrent, Attention, Embedding, Functional - Placeholders."""
from neurova.nn.layers import Module

class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1): super().__init__()
    def forward(self, x): return x
class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1): super().__init__()
    def forward(self, x): return x
class GRU(Module):
    def __init__(self, input_size, hidden_size, num_layers=1): super().__init__()
    def forward(self, x): return x
class RNNCell(Module):
    def __init__(self, input_size, hidden_size): super().__init__()
    def forward(self, x): return x
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size): super().__init__()
    def forward(self, x): return x
class GRUCell(Module):
    def __init__(self, input_size, hidden_size): super().__init__()
    def forward(self, x): return x
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.