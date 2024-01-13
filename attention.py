"""
A few Attention mechanisms. 

All are implemented as self-attention only.
All have a causal and an a-causal version. 
No explicit masking is done, but the causal versions are implemented as such.
This is because I want this to be simple and comparable, and for torch.compile to make it fast.
"""

from typing import Callable

import torch 
from torch import nn 


class Vanilla(nn.Module):
    """Acausal Vanilla Attention"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, dim = X.size()
        assert dim == self.dim
        assert dim % self.heads == 0
        dim_per_head = dim // self.heads

        Q, K, V = self.in_proj(X).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dim_per_head)

        scores = torch.softmax(scores, dim=-1)
        # (batch, heads, seq_len, dim_per_head)
        context = torch.matmul(scores, V)
        # (batch, seq_len, heads, dim_per_head)
        context = context.transpose(1, 2)
        # (batch, seq_len, dim)
        context = context.reshape(batch_size, seq_len, dim)
        context = self.out_proj(context)
        context = context + X
        return context


class VanillaCausal(nn.Module):
    """Causal Vanilla Attention"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim)

    def make_causal(self, scores: torch.Tensor, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
        return scores.masked_fill(mask == 0, -1e9)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, dim = X.size()
        assert dim == self.dim
        assert dim % self.heads == 0
        dim_per_head = dim // self.heads

        Q, K, V = self.in_proj(X).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dim_per_head)
        scores = self.make_causal(scores, seq_len)

        scores = torch.softmax(scores, dim=-1)
        # (batch, heads, seq_len, dim_per_head)
        context = torch.matmul(scores, V)
        # (batch, seq_len, heads, dim_per_head)
        context = context.transpose(1, 2)
        # (batch, seq_len, dim)
        context = context.reshape(batch_size, seq_len, dim)
        context = self.out_proj(context)
        context = context + X
        return context


"""
I'm not sure if Hydra Attention uses an out-projection.
In the code they provide in the paper, they dot not,
but they also provide pre-made Q, K, and V matrices,
so the in-projection is already assumed.

I'm just going to allow experiments with and without.

I will do this for Hydra and Hercules Attention,
but not Zeus, because there is no Q matrix and no skip connection,
and from what I've read, this means that there is no need for an out-projection.
"""


class Hydra(nn.Module):
    """ 
    Acausal Hydra Attention.
    """
    def __init__(
            self, 
            dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        A = torch.sum(feature_map(K) * V, dim=-2)
        Y = A * Q
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class HydraCausal(nn.Module):
    def __init__(
            self, 
            dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        A = torch.cumsum(feature_map(K) * V, dim=-2)  # cumsum means causal
        Y = A * self.feature_map(Q)
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class Hercules(nn.Module):
    """
    Acausal Hercules Attention.
    
    "Hercules" because it is meant to slay the Hydra.
    """

    def __init__(
            self, dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        A = torch.sum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight
        Y = A * Q
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class HerculesCausal(nn.Module):
    """
    Causal Hercules Attention.
    
    "Hercules" because it is meant to slay the Hydra.
    """

    def __init__(
            self, dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(dim, int(dim * 3))
        self.out_proj = nn.Linear(dim, dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        A = torch.cumsum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight
        Y = A * Q
        Y = self.out_proj(Y)
        Z = Y + X
        return Z

    
class Zeus(nn.Module):
    """
    Acausal Zeus Attention.
    
    "Zeus" because it is Hercules' daddy.
    """

    def __init__(
            self, dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(dim, int(dim * 2))

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(X).chunk(2, dim=-1)
        A = torch.sum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight  # =^= residual
        Z = A * X
        return Z


class ZeusCausal(nn.Module):
    """
    Causal Zeus Attention.
    """

    def __init__(
            self, dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(dim, int(dim * 2))

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(X).chunk(2, dim=-1)
        A = torch.cumsum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight  # =^= residual
        Z = A * X
        return Z
