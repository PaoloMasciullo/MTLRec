import torch
from torch import nn
from src.layers.representational_layer import RepresentationalLayer
from src.layers.mlp_block import MLPBlock
from src.layers.target_attention import TargetAttention


class MLP(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 dnn_hidden_sizes=[512, 128, 64],
                 dnn_hidden_activations="ReLu",
                 dnn_output_size=1,
                 dnn_output_activation=None,
                 dnn_dropout=0.5,
                 use_batchnorm=True,
                 ):
        """
        :param feature_map: Dictionary of feature names with their types (cardinality or "numerical").
        :param embedding_dim: Dimension for learnable embedding layers.
        :param dnn_hidden_sizes: List of hidden units in the final MLP.
        :param dnn_output_size: Size of the output layer (1 for binary classification).
        """
        super(MLP, self).__init__()
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim

        # Embedding layer
        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )

        # Final MLP Block
        self.dnn = MLPBlock(
            input_size=embedding_dim * sum(1 for info in feature_map["features"].values() if info.get('type') != 'meta'),
            hidden_sizes=dnn_hidden_sizes,
            hidden_activations=dnn_hidden_activations,
            output_size=dnn_output_size,
            output_activation=dnn_output_activation,
            dropout_probs=dnn_dropout,
            use_batchnorm=use_batchnorm
        )
        self.apply(self.init_weights)

    def forward(self, X):
        # Extract embeddings
        feature_emb_dict = self.embedding_layer(X)

        # Combine all features for DNN
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None]
        feature_emb = torch.cat(feature_list, dim=-1)

        # Pass through DNN for final prediction
        y_pred = self.dnn(feature_emb)
        return y_pred

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
