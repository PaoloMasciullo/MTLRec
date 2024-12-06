import torch.nn as nn
import torch.nn.functional as F

from src.utils.torch_utils import get_activation


class MLPBlock(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size=None,
                 hidden_activations="ReLU",
                 output_activation=None,
                 dropout_probs=0.5,
                 use_batchnorm=False,
                 use_bias=True):
        """
        A multi-layer perceptron (MLP) block with customizable hidden layers, activations, dropout, and batch normalization.

        :param input_size: The number of input features.
        :param hidden_sizes: List of sizes for the hidden layers.
        :param output_size: Size of the output layer.
        :param hidden_activations: List of activation functions for hidden layers (can be string or list of strings).
        :param output_activation: Activation function for output layer (default: None).
        :param dropout_probs: Dropout probabilities for each hidden layer (default: 0.5).
        :param use_batchnorm: Whether to apply batch normalization after each hidden layer (default: True).
        :param use_bias: Whether to use bias in linear layers
        """
        super(MLPBlock, self).__init__()

        # Ensure dropout_probs and hidden_activations are lists of the correct length
        if isinstance(dropout_probs, (float, int)):  # If single value, make it a list
            dropout_probs = [dropout_probs] * len(hidden_sizes)
        if isinstance(hidden_activations, str):  # If single activation, make it a list
            hidden_activations = [hidden_activations] * len(hidden_sizes)

        # Ensure that the length of these lists matches the number of hidden layers
        assert len(dropout_probs) == len(hidden_sizes), "Length of dropout_probs must match length of hidden_sizes"
        assert len(hidden_activations) == len(hidden_sizes), "Length of hidden_activations must match length of hidden_sizes"

        # Get activation functions
        hidden_activations = [get_activation(act, units) for act, units in zip(hidden_activations, hidden_sizes)]

        layers = []  # To store layers sequentially
        prev_size = input_size

        # Create hidden layers
        for idx, hidden_size in enumerate(hidden_sizes):
            # Add fully connected layer
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            # Add batch normalization if enabled
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            # Add activation
            if hidden_activations[idx]:
                layers.append(hidden_activations[idx])
            # Add dropout
            if dropout_probs[idx] > 0:
                layers.append(nn.Dropout(p=dropout_probs[idx]))

            prev_size = hidden_size  # Update size for next layer

        # Output layer
        if output_size:
            layers.append(nn.Linear(prev_size, output_size, bias=use_bias))
            if output_activation:
                layers.append(get_activation(output_activation))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
