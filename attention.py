"""
A few Attention mechanisms. 

All are implemented as self-attention only.
All have a causal and an a-causal version. 
No explicit masking is done, but the causal versions are implemented as such.
This is because I want this to be simple and comparable, and for torch.compile to make it fast.

They are also written for training, because that's what I'm interested in.
"""

from typing import Callable

import torch 
from torch import nn 
from rotary_embedding_torch import RotaryEmbedding


def embed_rotary(*args: torch.Tensor, dim: int) -> list[torch.Tensor]:
    rot_emb = RotaryEmbedding(dim // 2)  # rotary embedding is half the size of the dim
    return [rot_emb.rotate_queries_or_keys(arg) for arg in args]


class Vanilla(nn.Module):
    """
    Acausal Vanilla Attention.
    
    Doesn't use nn.MultiheadAttention, 
    because I don't have efficient implementations of the other Attention mechanisms
    but still want them to be comparable.
    """
    def __init__(self, feature_dim: int, heads: int):
        assert feature_dim % heads == 0
        assert feature_dim % 2 == 0
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, feature_dim = X.size()
        assert feature_dim == self.feature_dim
        dim_per_head = feature_dim // self.heads
        assert dim_per_head % 2 == 0

        Q, K, V = self.in_proj(X).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q, K = embed_rotary(Q, K, dim=dim_per_head)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dim_per_head)

        scores = torch.softmax(scores, dim=-1)
        # (batch, heads, seq_len, dim_per_head)
        context = torch.matmul(scores, V)
        # (batch, seq_len, heads, dim_per_head)
        context = context.transpose(1, 2)
        # (batch, seq_len, dim)
        context = context.reshape(batch_size, seq_len, feature_dim)
        context = self.out_proj(context)
        context = context + X
        return context


class VanillaCausal(nn.Module):
    """Causal Vanilla Attention"""
    def __init__(self, feature_dim: int, heads: int):
        assert feature_dim % heads == 0
        assert feature_dim % 2 == 0
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = heads
        self.seq_len = 32  # initial sequence length from training --- updated in forward
        self.mask = torch.tril(torch.ones(self.seq_len, self.seq_len), diagonal=0)
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, feature_dim = X.size()
        if seq_len != self.seq_len:
            self.mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0)
            self.seq_len = seq_len

        assert feature_dim == self.feature_dim
        dim_per_head = feature_dim // self.heads
        assert dim_per_head % 2 == 0

        Q, K, V = self.in_proj(X).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q, K = embed_rotary(Q, K, dim=dim_per_head)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(dim_per_head)
        scores = scores.masked_fill(self.mask == 0, -1e9)

        scores = torch.softmax(scores, dim=-1)
        # (batch, heads, seq_len, dim_per_head)
        context = torch.matmul(scores, V)
        # (batch, seq_len, heads, dim_per_head)
        context = context.transpose(1, 2)
        # (batch, seq_len, dim)
        context = context.reshape(batch_size, seq_len, feature_dim)
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
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        Q, K = embed_rotary(Q, K, dim=self.feature_dim)
        A = torch.sum(feature_map(K) * V, dim=-2)
        Y = A * Q
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class HydraCausal(nn.Module):
    def __init__(
            self, 
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        Q, K = embed_rotary(Q, K, dim=self.feature_dim)
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
            self, 
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim)
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
            self, 
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            use_out_proj: bool = True,
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3))
        self.out_proj = nn.Linear(feature_dim, feature_dim) if use_out_proj else torch.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(X).chunk(3, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim)
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
            self, 
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 2))

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(X).chunk(2, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim)
        A = torch.sum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight  # =^= residual
        Z = A * X
        return Z


class ZeusCausal(nn.Module):
    """
    Causal Zeus Attention.
    """

    def __init__(
            self, 
            feature_dim: int, 
            feature_map: Callable[[torch.Tensor], torch.Tensor],
            identity_weight: float = 0.5,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map = feature_map
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 2))

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(X).chunk(2, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim)
        A = torch.cumsum(feature_map(K) * feature_map(V), dim=-2)
        A = (1 - self.identity_weight) * A + self.identity_weight  # =^= residual
        Z = A * X
        return Z
