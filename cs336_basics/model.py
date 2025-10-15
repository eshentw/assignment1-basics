import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).to(device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        mu = 0.0
        nn.init.trunc_normal_(self.weight, mean=mu, std=std, a=-3*std, b=3*std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            input: (..., in_features)
        Returns:
            output: (..., out_features)
        """
        return einsum(input, self.weight, "... i, o i -> ... o")
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim)).to(device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1
        mu = 0.0
        nn.init.trunc_normal_(self.embedding, mean=mu, std=std, a=-3*std, b=3*std)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        '''
        Parameters:
            input: (...,) long tensor of indices
        Returns:
            output: (..., d)
        '''
        return self.embedding[input]
