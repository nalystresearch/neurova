# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""Embedding, Functional - Placeholders."""
from neurova.nn.layers import Module

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim): super().__init__()
    def forward(self, x): return x
class EmbeddingBag(Module):
    def __init__(self, num_embeddings, embedding_dim): super().__init__()
    def forward(self, x): return x
# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.