import torch
from torch import nn
from src.layers.representational_layer import RepresentationalLayer
from src.layers.mlp_block import MLPBlock
from src.layers.target_attention import TargetAttention


class DIVAN(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 attention_hidden_sizes=[64],
                 attention_hidden_activations="Dice",
                 dnn_hidden_sizes=[512, 128, 64],
                 dnn_hidden_activations="ReLu",
                 dnn_output_size=1,
                 dnn_output_activation="Sigmoid",
                 dnn_dropout=0.5,
                 gate_hidden_units=[100],
                 gate_hidden_activations="Relu",
                 gate_output_size=1,
                 gate_output_activation="Sigmoid",
                 gate_dropout=0,
                 pop_hidden_units=[512, 256, 128],
                 pop_activations="ReLu",
                 pop_output_size=1,
                 pop_output_activation="Sigmoid",
                 pop_dropout=0.3,
                 use_batchnorm=True,
                 target_features=["item_id", "cate_id"],
                 sequence_features=["click_history", "cate_history"],
                 recency_features=["publish_hours"],
                 attention_dropout=0.5,
                 use_softmax=False,
                 target_pretrained_multimodal_embeddings=None,  # ["item_id_img", "item_id_text"]
                 sequence_pretrained_multimodal_embeddings=None,  # ["hist_id_img", "hist_id_text"]
                 ):
        """
        :param feature_map: Dictionary of feature names with their types (cardinality or "numerical").
        :param embedding_dim: Dimension for learnable embedding layers.
        :param attention_hidden_sizes: List of hidden units in the attention MLP.
        :param attention_hidden_activations: hidden activation function for the attention MLP layers.
        :param dnn_hidden_sizes: List of hidden units in the final MLP.
        :param dnn_output_size: Size of the output layer (1 for binary classification).
        :param target_features: List specifying target features for attention.
        :param sequence_features: List specifying sequence features for attention.
        :param target_pretrained_multimodal_embeddings: List specifying target pretrained multimodal embeddings.
        :param sequence_pretrained_multimodal_embeddings: List specifying sequence pretrained multimodal embeddings.
        """
        super(DIVAN, self).__init__()
        self.feature_map = feature_map
        self.target_features = target_features
        self.sequence_features = sequence_features
        self.target_pretrained_multimodal_embeddings = target_pretrained_multimodal_embeddings
        self.sequence_pretrained_multimodal_embeddings = sequence_pretrained_multimodal_embeddings
        self.recency_features = recency_features
        self.embedding_dim = embedding_dim

        assert len(self.target_features) == len(self.sequence_features), \
            "len(target_features) != len(sequence_features)"

        if self.target_pretrained_multimodal_embeddings:
            assert len(self.target_pretrained_multimodal_embeddings) == len(
                self.sequence_pretrained_multimodal_embeddings), \
                "len(target_pretrained_multimodal_embeddings) != len(sequence_pretrained_multimodal_embeddings)"

        # Embedding layer
        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )

        # Target Attention module
        self.attention_layer = TargetAttention(
            embedding_dim=embedding_dim * len(self.target_features),
            mlp_hidden_sizes=attention_hidden_sizes,
            mlp_hidden_activations=attention_hidden_activations,
            mlp_dropout_prob=attention_dropout,
            use_softmax=use_softmax)

        # Target Attention module for multimodal embeddings
        if self.target_pretrained_multimodal_embeddings:
            self.multimodal_attention_layer = TargetAttention(
                embedding_dim=embedding_dim * len(self.target_pretrained_multimodal_embeddings),
                mlp_hidden_sizes=attention_hidden_sizes,
                mlp_hidden_activations=attention_hidden_activations,
                mlp_dropout_prob=attention_dropout,
                use_softmax=use_softmax)

        self.gate = MLPBlock(
            input_size=embedding_dim * len([*self.sequence_features, *self.sequence_pretrained_multimodal_embeddings]),
            hidden_sizes=gate_hidden_units,
            hidden_activations=gate_hidden_activations,
            output_size=gate_output_size,
            output_activation=gate_output_activation,
            dropout_probs=gate_dropout,
            use_batchnorm=use_batchnorm)

        self.vir_aware_net = MLPBlock(
            input_size=embedding_dim * (len([*self.target_features, *self.target_pretrained_multimodal_embeddings, *self.recency_features])),
            hidden_sizes=pop_hidden_units,
            hidden_activations=pop_activations,
            output_size=pop_output_size,
            output_activation=pop_output_activation,
            dropout_probs=pop_dropout,
            use_batchnorm=use_batchnorm)

        # Final MLP Block
        self.dnn = MLPBlock(
            input_size=embedding_dim * sum(1 for info in feature_map["features"].values() if info.get('type') != 'meta'),
            hidden_sizes=dnn_hidden_sizes,
            hidden_activations=dnn_hidden_activations,
            output_size=dnn_output_size,
            output_activation=dnn_output_activation,
            dropout_probs=dnn_dropout,
            use_batchnorm=use_batchnorm)

        self.apply(self.init_weights)

    def forward(self, X):
        # Extract embeddings
        feature_emb_dict = self.embedding_layer(X)

        # Apply standard attention to target and sequence features
        self._apply_attention(X, self.attention_layer, self.target_features, self.sequence_features, feature_emb_dict)

        # If multimodal embeddings are provided, apply attention for them
        if self.target_pretrained_multimodal_embeddings:
            self._apply_attention(X,
                                  self.multimodal_attention_layer,
                                  self.target_pretrained_multimodal_embeddings,
                                  self.sequence_pretrained_multimodal_embeddings,
                                  feature_emb_dict)

        # Combine all features for DNN
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None]
        feature_emb = torch.cat(feature_list, dim=-1)

        # Get user-specific alpha
        user_feature_list = [feature_emb_dict[field] for field in [*self.sequence_features, *self.sequence_pretrained_multimodal_embeddings] if feature_emb_dict[field] is not None]
        user_emb = torch.cat(user_feature_list, dim=1)
        alpha = self.gate(user_emb)

        # predict news virality
        news_feature_list = [feature_emb_dict[field] for field in [*self.target_features, *self.target_pretrained_multimodal_embeddings, *self.recency_features] if feature_emb_dict[field] is not None]
        new_emb = torch.cat(news_feature_list, dim=1)

        vir_aware_scores = self.vir_aware_net(new_emb)

        # predict din scores
        attention_based_scores = self.dnn(feature_emb)

        # Combine the two prediction scores with user-specific parameters
        y_pred = alpha * attention_based_scores + (1 - alpha) * vir_aware_scores
        return y_pred

    def _apply_attention(self, X, attention_layer, target_fields, sequence_fields, feature_emb_dict):
        """Helper to apply attention and update embedding dictionary."""
        seq_field = sequence_fields[0]  # pick the first sequence field
        mask = X[seq_field].long() != 0  # padding_idx = 0 required

        target_emb = torch.cat([feature_emb_dict[field] for field in target_fields],
                               dim=-1)  # (batch_size, embedding_dim * len(target_fields)
        sequence_emb = torch.cat([feature_emb_dict[field] for field in sequence_fields],
                                 dim=-1)  # (batch_size, seq_len, embedding_dim) * len(target_fields)
        context_vector = attention_layer(target_emb, sequence_emb, mask)

        # Update embeddings in dictionary
        for field, emb in zip(sequence_fields, context_vector.split(self.embedding_dim, dim=-1)):
            feature_emb_dict[field] = emb

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)