import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layers.mlp_block import MLPBlock


class TargetAttention(nn.Module):
    def __init__(self,
                 embedding_dim,
                 mlp_hidden_sizes,
                 mlp_hidden_activations="ReLU",
                 mlp_dropout_prob=0.5,
                 mlp_use_batchnorm=False,
                 use_softmax=False):
        """
        :param embedding_dim: Dimensionality of the embedding.
        :param mlp_hidden_sizes: List of hidden units for the MLP block in the energy layer.
        :param mlp_hidden_activations: hidden activation function for the MLP layers.
        :param mlp_dropout_prob: Dropout probability for the MLP layers.
        :param mlp_use_batchnorm: Whether to use batch normalization in the MLP layers.
        :param use_softmax: Whether to compute attention weights using softmax
        """
        super(TargetAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_softmax = use_softmax

        # To compute energy (attention score) using concatenation of interaction types
        combined_dim = embedding_dim * 4  # Concatenation of [q_proj, k_proj, q-k, q*k]

        # MLP block to replace the simple energy layer
        self.energy_mlp = MLPBlock(
            input_size=combined_dim,
            hidden_sizes=mlp_hidden_sizes,
            output_size=1,  # We need a single value as attention score
            hidden_activations=mlp_hidden_activations,
            dropout_probs=mlp_dropout_prob,
            use_batchnorm=mlp_use_batchnorm,
        )

    def forward(self, query, history_keys, mask=None):
        """
        :param query: Tensor of shape (batch_size, embedding_dim), representing the target item's embedding.
        :param history_keys: Tensor of shape (batch_size, seq_len, embedding_dim), representing the embeddings of the user's history.
        :param mask: mask of history_keys, 0 for masked positions
        :return: context_vector: Weighted sum of history keys based on attention scores.
        """
        seq_len = history_keys.size(1)
        query = query.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, embedding_dim)

        # Concatenate the different interactions along the last dimension
        attention_input = torch.cat([query, history_keys, query - history_keys, query * history_keys],
                                    dim=-1)  # (batch_size, seq_len, embedding_dim * 4)

        # Pass through MLP to get attention energy
        attention_weights = self.energy_mlp(attention_input.view(-1, 4 * self.embedding_dim)).view(-1, seq_len)  # (batch_size, seq_len)
        if mask is not None:
            attention_weights = attention_weights * mask.float()
        if self.use_softmax:
            # Compute attention weights using softmax
            if mask is not None:
                attention_weights += -1.e9 * (1 - mask.float())
            attention_weights = F.softmax(attention_weights, dim=1)  # (batch_size, seq_len)

        # Compute the weighted sum of the history keys
        output = torch.bmm(attention_weights.unsqueeze(1), history_keys).squeeze(1)  # (batch_size, embedding_dim)

        return output


class AttentionLayer(nn.Module):
    """attention for info tranfer

    Args:
        dim (int): attention dim

    Shape:
        Input: (batch_size, 2, dim)
        Output: (batch_size, dim)
    """

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs
