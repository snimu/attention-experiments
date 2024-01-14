"""A bunch of activation functions for Hydra Attention and equivalents."""
import torch


def cos_sim_activation(X: torch.Tensor) -> torch.Tensor:
    return X / torch.linalg.norm(X, dim=-1, keepdim=True)


def sqrt_dim_activation(X: torch.Tensor) -> torch.Tensor:
    _, _, d = X.shape
    return X / torch.sqrt(d)


def tanh_activation(X: torch.Tensor) -> torch.Tensor:
    return torch.tanh(X)


def softmax_activation(X: torch.Tensor) -> torch.Tensor:
    return torch.softmax(X, dim=-2)  # softmax over seq_len


def sigmoid_activation(X: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(X)


def relu_activation(X: torch.Tensor) -> torch.Tensor:
    return torch.relu(X)


def gelu_activation(X: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(X)


def identity(X: torch.Tensor) -> torch.Tensor:
    return X


ALL_ACTIVATIONS = {
    "cos_sim": cos_sim_activation,
    "sqrt_dim": sqrt_dim_activation,
    "tanh": tanh_activation,
    "softmax": softmax_activation,
    "sigmoid": sigmoid_activation,
    "relu": relu_activation,
    "gelu": gelu_activation,
    "identity": identity,
}
