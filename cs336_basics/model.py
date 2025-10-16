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


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (..., d_model)
        Returns:
            output: (..., d_model)
        """
        x = x + self.eps
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (self.d_model ** -0.5)
        x_normalized = x / (rms_x)
        return x_normalized * self.scale
    

class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (..., d_model)
        Returns:
            output: (..., d_model)
        """
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.silu = SiLU()
    
    def forward(self, x1, x2: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x1: (..., d_model)
            x2: (..., d_model)
        Returns:
            output: (..., d_ff=8/3*d_model)
        """
        return self.silu(x1) * x2


class FFN(nn.Module):
    def __init__(self, d_model: int, dff: int) -> None:
        super().__init__()
        self.d_model = d_model
        if dff is None:
            dff = int(8 * d_model / 3)
        self.linear1 = Linear(d_model, dff)
        self.linear2 = Linear(dff, d_model)
        self.linear3 = Linear(d_model, dff)
        self.swilu = SwiGLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (..., d_model)
        Returns:
            output: (..., d_model)
        """
        x1 = self.linear1(x)
        x2 = self.linear3(x)
        x = einsum(
            self.swilu(x1, x2), self.linear2, "... d_ff, d_model dff -> ... d_model")
        return x