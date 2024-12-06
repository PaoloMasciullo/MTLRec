import torch.optim as optim
import os
import numpy as np
import torch
from torch import nn
import random


def seed_everything(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    os.environ["PYTHONHASHSEED"] = str(seed)  # Ensure the Python hash is consistent

    # For PyTorch on CPU
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # If you're using MPS backend on MacOS
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # Ensure MPS device gets the same seed
        torch.backends.mps.deterministic = True  # Set MPS to deterministic mode

    # For CUDA, if you're using CUDA backend
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For all CUDA devices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility


def get_loss(loss_name):
    losses = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'bce': nn.BCELoss(),
        'bce_logits': nn.BCEWithLogitsLoss(),
        'nll': nn.NLLLoss(),
        'hinge': nn.HingeEmbeddingLoss(),
        'smooth_l1': nn.SmoothL1Loss()
    }
    if loss_name.lower() in losses:
        return losses[loss_name]
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}.")


def get_optimizer(optimizer_name, model_params, lr=0.001):
    optimizers = {
        'adam': optim.Adam(model_params, lr=lr),
        'sgd': optim.SGD(model_params, lr=lr),
        'adamw': optim.AdamW(model_params, lr=lr),
        'rmsprop': optim.RMSprop(model_params, lr=lr),
        'adagrad': optim.Adagrad(model_params, lr=lr)
    }
    if optimizer_name.lower() in optimizers:
        return optimizers[optimizer_name]
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}.")


def get_activation(activation_name, hidden_units=None):
    activation_name = activation_name.lower()

    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'softmax': nn.Softmax(dim=-1),
        'logsoftmax': nn.LogSoftmax(dim=-1)
    }
    if activation_name in activations:
        return activations[activation_name]
    elif activation_name == "dice":
        assert type(hidden_units) == int
        from src.layers.activations import Dice
        return Dice(hidden_units)
    else:
        raise ValueError(f"Unsupported activation name: {activation_name}.")


def get_device(gpu_idx):
    device = torch.device("cpu")
    if gpu_idx >= 0:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu_idx))
        elif torch.backends.mps.is_available():
            device = torch.device("mps:" + str(gpu_idx))
    return device
