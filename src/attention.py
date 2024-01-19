"""
A few Attention mechanisms. 

All are implemented as self-attention only.
All have a causal and an a-causal version. 
No explicit masking is done, but the causal versions are implemented as such.
This is because I want this to be simple and comparable, and for torch.compile to make it fast.

They are also written for training, because that's what I'm interested in.
"""

from typing import Callable, Union

import torch 
from torch import nn 
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange

from feature_maps import cos_sim, identity


DEVICE_TYPE = Union[str, int, torch.device]


def embed_rotary(*args: torch.Tensor, dim: int, device="cuda", dtype=torch.bfloat16) -> list[torch.Tensor]:
    rot_emb = RotaryEmbedding(dim // 2)  # rotary embedding is half the size of the dim
    return [rot_emb.rotate_queries_or_keys(arg.cpu()).to(device, dtype) for arg in args]  # TODO: why does this not work on CUDA?


class TorchMHACausal(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            num_heads: int,
            norm: nn.Module,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.mha = nn.MultiheadAttention(feature_dim, num_heads, dropout=0.0, bias=False, device=device, dtype=dtype)
        self.seq_len = 32
        self.mask = self.update_mask(self.seq_len)

    def update_mask(self, seq_len: int) -> torch.Tensor:
        self.mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1).to(self.device)
        self.seq_len = seq_len
        return self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _, seq_len, feature_dim = X.size()
        if seq_len != self.seq_len:
            self.update_mask(seq_len)
        assert feature_dim == self.feature_dim

        residual = X
        X = self.norm(X)
        X = X.permute(1, 0, 2)
        X = self.mha(X, X, X, attn_mask=self.mask, need_weights=False)[0]
        X = X.permute(1, 0, 2)
        X = X + residual
        return X


class Vanilla(nn.Module):
    """
    Acausal Vanilla Attention.
    
    Doesn't use nn.MultiheadAttention, 
    because I don't have efficient implementations of the other Attention mechanisms
    but still want them to be comparable.
    """
    def __init__(
            self, 
            feature_dim: int, 
            num_heads: int, 
            norm: nn.Module,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        assert feature_dim % num_heads == 0
        assert feature_dim % 2 == 0
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.norm = norm
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, feature_dim = X.size()
        assert feature_dim == self.feature_dim
        dim_per_head = feature_dim // self.num_heads
        assert dim_per_head % 2 == 0

        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.num_heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.num_heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.num_heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q, K = embed_rotary(Q, K, dim=dim_per_head, device=self.device, dtype=self.dtype)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim_per_head))

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
    def __init__(
            self, 
            feature_dim: int, 
            num_heads: int,
            norm: nn.Module,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        assert feature_dim % num_heads == 0
        assert feature_dim % 2 == 0
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.norm = norm
        self.seq_len = 32  # initial sequence length from training --- updated in forward
        self.mask = self.update_mask(self.seq_len)
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype)

    def update_mask(self, seq_len: int) -> torch.Tensor:
        self.mask = torch.tril(torch.ones(seq_len, seq_len, dtype=self.dtype), diagonal=0).to(self.device)
        self.seq_len = seq_len
        return self.mask

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Q, K, V: (batch, seq_len, dim)"""
        batch_size, seq_len, feature_dim = X.size()
        if seq_len != self.seq_len:
            self.update_mask(seq_len)

        assert feature_dim == self.feature_dim
        dim_per_head = feature_dim // self.num_heads
        assert dim_per_head % 2 == 0

        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)

        Q = Q.view(batch_size, seq_len, self.num_heads, dim_per_head)
        K = K.view(batch_size, seq_len, self.num_heads, dim_per_head)
        V = V.view(batch_size, seq_len, self.num_heads, dim_per_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        Q, K = embed_rotary(Q, K, dim=dim_per_head, device=self.device, dtype=self.dtype)

        # (batch, heads, seq_len, dim_per_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(dim_per_head))
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


def img_to_qkv(
        X: torch.Tensor, 
        norm: nn.Module, 
        in_proj: nn.Module, 
        heads: int = 1,
        chunks: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = in_proj(norm(X)).chunk(chunks, dim=1)
    qkv = map(
        lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=heads), qkv
    ) 
    return qkv


class VanillaConv(nn.Module):
    def __init__(self, feature_dim, norm, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.norm = norm
        hidden_dim = dim_head * heads
        self.in_proj = nn.Conv2d(feature_dim, hidden_dim * 3, 1, bias=False)
        self.out_proj = nn.Conv2d(hidden_dim, feature_dim, 1)

    def forward(self, X):
        Q, V, v = img_to_qkv(X, self.norm, self.in_proj, heads=self.heads)
        Q = Q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", Q, V)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        _, _, h, w = X.shape
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.out_proj(out) + X
    

class LinearConv(nn.Module):
    def __init__(self, feature_dim, norm, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.norm = norm
        hidden_dim = dim_head * heads
        self.in_proj = nn.Conv2d(feature_dim, hidden_dim * 3, 1, bias=False)

        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, feature_dim, 1), 
            nn.GroupNorm(1, feature_dim)
        )

    def forward(self, X):
        Q, K, V = img_to_qkv(X, self.norm, self.in_proj, heads=self.heads)

        Q = Q.softmax(dim=-2)
        K = K.softmax(dim=-1)

        Q = Q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", K, V)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, Q)
        _, _, h, w = X.shape
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.out_proj(out) + X


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
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)
        Q, K = embed_rotary(Q, K, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.sum(self.feature_map_qkv(K) * V, dim=-2)
        Y = self.feature_map_attn(A) * self.feature_map_qkv(Q)
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class HydraCausal(nn.Module):
    def __init__(
            self, 
            feature_dim: int, 
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)
        Q, K = embed_rotary(Q, K, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.cumsum(self.feature_map_qkv(K) * V, dim=-2)  # cumsum means causal
        Y = self.feature_map_attn(A) * self.feature_map_qkv(Q)
        Y = self.out_proj(Y)
        Z = Y + X
        return Z


class HydraConv(nn.Module):
    """ 
    Acausal Hydra Attention for images.
    """
    def __init__(
            self, 
            feature_dim: int, 
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Conv2d(feature_dim, feature_dim * 3, 1, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = img_to_qkv(X, self.norm, self.in_proj)  # 3x (batch, 1, dim, h*w)
        A = torch.sum(self.feature_map_qkv(K) * V, dim=-1, keepdim=True)  # Attend over the spatial dimension # TODO: is this correct?
        print(f"{A.shape=}, {Q.shape=}, {V.shape=}, {K.shape=}")
        Y = self.feature_map_attn(A) * self.feature_map_qkv(Q)
        # TODO: 
        #  Y = self.feature_map_attn(A) * self.feature_map_qkv(Q)
        #     RuntimeError: The size of tensor a (112) must match the size of tensor b (49) at non-singleton dimension 3
        Y = self.out_proj(Y)
        _, _, h, w = X.shape
        Y = rearrange(Y, "b 1 c (x y) -> b c x y", x=h, y=w)
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
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.sum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-2)
        Y = self.feature_map_attn(A) * Q
        Y = self.out_proj(Y)
        Z = Y + X
        return Z
    

class HerculesConv(nn.Module):
    """
    Acausal Hercules Attention for images.
    
    "Hercules" because it is meant to slay the Hydra.
    """

    def __init__(
            self, 
            feature_dim: int, 
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Conv2d(feature_dim, feature_dim * 3, 1, bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Conv2d(feature_dim, feature_dim, 1, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = img_to_qkv(X, self.norm, self.in_proj)  # 3x (batch, 1, dim, h*w)
        A = torch.sum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-1) # Attend over the spatial dimension # TODO: is this correct?
        Y = self.feature_map_attn(A) * Q
        Y = self.out_proj(Y)
        _, _, h, w = X.shape
        Y = rearrange(Y, "b 1 c (x y) -> b c x y", x=h, y=w)
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
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            use_out_proj: bool = True,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 3), bias=False, device=device, dtype=dtype)
        self.out_proj = nn.Linear(feature_dim, feature_dim, bias=False, device=device, dtype=dtype) if use_out_proj else nn.Identity()

    def forward(self, X: torch.Tensor):
        Q, K, V = self.in_proj(self.norm(X)).chunk(3, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.cumsum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-2)
        Y = self.feature_map_attn(A) * Q
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
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            identity_weight: float = 0.5,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 2), bias=False, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(self.norm(X)).chunk(2, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.sum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-2)
        A = (1 - self.identity_weight) * self.feature_map_attn(A) + self.identity_weight  # =^= residual
        Z = A * X
        return Z
    

class ZeusConv(nn.Module):
    """
    Acausal Zeus Attention for images.
    
    "Zeus" because it is Hercules' daddy.
    """

    def __init__(
            self, 
            feature_dim: int, 
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            identity_weight: float = 0.5,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.identity_weight = identity_weight
        self.in_proj = nn.Conv2d(feature_dim, feature_dim * 3, 1, bias=False, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor):
        K, V = img_to_qkv(X, self.norm, self.in_proj, chunks=2)  # 2x (batch, 1, dim, h*w)
        A = torch.sum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-1, keepdim=True)  # Attend over the spatial dimension # TODO: is this correct?
        A = (1 - self.identity_weight) * self.feature_map_attn(A) + self.identity_weight  # =^= residual
        A = rearrange(A, "b 1 c 1 -> b c 1 1")  # X.shape = (b c x y), A.shape = (b c 1 1)
        Z = A * X
        return Z


class ZeusCausal(nn.Module):
    """
    Causal Zeus Attention.
    """

    def __init__(
            self, 
            feature_dim: int, 
            norm: nn.Module,
            feature_map_qkv: Callable[[torch.Tensor], torch.Tensor] = cos_sim,
            feature_map_attn: Callable[[torch.Tensor], torch.Tensor] = identity,
            identity_weight: float = 0.5,
            device: DEVICE_TYPE = 'cuda',
            dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_map_qkv = feature_map_qkv
        self.feature_map_attn = feature_map_attn
        self.norm = norm
        self.device = device
        self.dtype = dtype
        self.identity_weight = identity_weight
        self.in_proj = nn.Linear(feature_dim, int(feature_dim * 2), bias=False, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor):
        K, V = self.in_proj(self.norm(X)).chunk(2, dim=-1)
        K, V = embed_rotary(K, V, dim=self.feature_dim, device=self.device, dtype=self.dtype)
        A = torch.cumsum(self.feature_map_qkv(K) * self.feature_map_qkv(V), dim=-2)
        A = (1 - self.identity_weight) * self.feature_map_attn(A) + self.identity_weight  # =^= residual
        Z = A * X
        return Z
