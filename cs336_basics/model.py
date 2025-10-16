import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, einsum


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    A numerically stable implementation of softmax.
    Parameters:
        x: (..., n)
        dim: dimension to apply softmax over
    Returns:
        output: (..., n) with softmax applied over `dim`
    """
    x_max = rearrange(
        torch.max(x, dim=dim).values, "... -> ... 1")
    # avoid overflow by subtracting max
    # since softmax(x) = softmax(x + c) for any constant c
    x_exp = torch.exp(x - x_max)
    x_exp_sum = rearrange(torch.sum(x_exp, dim=dim), "... -> ... 1")
    return x_exp / x_exp_sum


def scaled_dot_product_attention(
                                q: torch.Tensor, 
                                k: torch.Tensor, 
                                v: torch.Tensor, 
                                mask: Optional[torch.Tensor] = None
                            ) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    Parameters:
        q: (..., seq_len, d_k)
        k: (..., seq_len, d_k)
        v: (..., seq_len, d_v)
        mask: (seq_len, seq_len) with 1s in positions to attend to and 0s elsewhere
    Returns:
        output: (..., seq_len_q, d_v)
    """
    scores = einsum(
        q, k, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k") / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = einsum(
        attn_weights, v, "... seq_len seq_len, ... seq_len d_v -> ... seq_len d_v")
    return output


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
        in_dtype = x.dtype
        x = x.to(torch.float32)
        norm_x = x.norm(2, dim=-1, keepdim=True) + self.eps
        rms_x = norm_x * (self.d_model ** -0.5)
        x_normalized = x / (rms_x)
        return (x_normalized * self.scale).to(in_dtype)
    

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
        x3 = self.swilu(x1, x2)
        x = self.linear2(x3)
        return x


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # Precompute the sinusoidal frequencies
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float()
        freqs = einsum(t, inv_freq, "seq_len, d -> seq_len d")
        cos_emb = freqs.cos()
        sin_emb = freqs.sin()
        self.register_buffer("cos_emb", cos_emb, persistent=False)  # (max_seq_len, d_k/2)
        self.register_buffer("sin_emb", sin_emb, persistent=False)  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (..., seq_len, d_k)
            token_positions: (..., seq_len)
        Returns:
            output: (..., seq_len, d_k)
        """
        cos_emb = self.cos_emb[token_positions]  # (..., d_k/2)
        sin_emb = self.sin_emb[token_positions]  # (..., d_k/2)

        x1, x2 = x[..., ::2], x[..., 1::2]  # (..., d_k/2), (..., d_k/2)
        x_rotated_1 = x1 * cos_emb - x2 * sin_emb  # (..., d_k/2)
        x_rotated_2 = x1 * sin_emb + x2 * cos_emb  # (..., d_k/2)
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_1
        x_rotated[..., 1::2] = x_rotated_2
        return x_rotated
    

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = Linear(self.num_heads * self.d_k, d_model)
        self.k_proj = Linear(self.num_heads * self.d_k, d_model)
        self.v_proj = Linear(self.num_heads * self.d_k, d_model)
        self.o_proj = Linear(d_model, self.num_heads * self.d_k)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        b, s, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        
        casual_mask = torch.triu(torch.ones(s, s, device=x.device), diagonal=1).bool()
        attn_output = scaled_dot_product_attention(q, k, v, mask=~casual_mask)
        attn_output = rearrange(attn_output, "b h s d -> b s (h d)")

        output = self.o_proj(attn_output)
        return output
