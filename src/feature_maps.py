"""A bunch of feature maps (activation functions) for Hydra Attention and equivalents."""
import torch


def cos_sim(X: torch.Tensor) -> torch.Tensor:
    return X / torch.linalg.norm(X, dim=-1, keepdim=True)


def sqrt_dim(X: torch.Tensor) -> torch.Tensor:
    _, _, d = X.shape
    return X / torch.sqrt(torch.tensor(d))


def tanh(X: torch.Tensor) -> torch.Tensor:
    return torch.tanh(X)


def softmax(X: torch.Tensor) -> torch.Tensor:
    return torch.softmax(X, dim=-2)  # softmax over seq_len


def sigmoid(X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(X)


def relu(X: torch.Tensor) -> torch.Tensor:
    return torch.relu(X)


def gelu(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(X)


def identity(X: torch.Tensor) -> torch.Tensor:
    return X


ACTIVATION_NAME_TO_FUNCTION = {
    "cos_sim": cos_sim,
    "sqrt_dim": sqrt_dim,
    "tanh": tanh,
    "softmax": softmax,
    "sigmoid": sigmoid,
    "relu": relu,
    "gelu": gelu,
    "identity": identity,
}
ACTIVATION_FUNCTION_TO_NAME = {v: k for k, v in ACTIVATION_NAME_TO_FUNCTION.items()}
