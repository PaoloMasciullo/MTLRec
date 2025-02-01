import torch
from torch import nn
from src.layers.representational_layer import RepresentationalLayer
from src.layers.mlp_block import MLPBlock
from src.layers.attention_layers import TargetAttention


class HIFN(nn.Module):
    """Hierarchical Interest Fusion Network"""
    def __init__(self,
                 hifn_input_size=10,
                 dnn_hidden_sizes=[512, 128, 64],
                 dnn_hidden_activations="ReLu",
                 dnn_dropout=0.5,
                 gate_hidden_sizes=[512, 128],
                 gate_hidden_activations="ReLu",
                 gate_dropout=0,
                 use_batchnorm=False,
                 ):
        super(HIFN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hifn_input_size)
        self.gate = MLPBlock(
            input_size=hifn_input_size,
            hidden_sizes=gate_hidden_sizes,
            hidden_activations=gate_hidden_activations,
            output_size=hifn_input_size,
            dropout_probs=gate_dropout,
        )
        self.mlp = MLPBlock(
            input_size=hifn_input_size,
            hidden_sizes=dnn_hidden_sizes,
            hidden_activations=dnn_hidden_activations,
            output_size=hifn_input_size,
            dropout_probs=dnn_dropout,
            use_batchnorm=use_batchnorm
        )

    def forward(self, feature_emb):
        e_context = self.batch_norm(feature_emb)
        e_output = self.mlp(self.gate(e_context) * feature_emb)
        return e_output


class HUIM(nn.Module):
    """Hierarchical User Interest Modeling"""

    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 attention_hidden_sizes=None,
                 attention_hidden_activations="Dice",
                 dnn_hidden_sizes=None,
                 dnn_hidden_activations="ReLu",
                 dnn_output_size=1,
                 dnn_output_activation="Sigmoid",
                 dnn_dropout=0.5,
                 gate_hidden_sizes=None,
                 gate_hidden_activations="ReLu",
                 gate_dropout=0,
                 use_batchnorm=False,
                 target_features=None,
                 sequence_features=None,
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
        super(HUIM, self).__init__()
        if attention_hidden_sizes is None:
            attention_hidden_sizes = [64]
        if dnn_hidden_sizes is None:
            dnn_hidden_sizes = [512, 128, 64]
        if gate_hidden_sizes is None:
            gate_hidden_sizes = [512, 128]
        if sequence_features is None:
            sequence_features = ["click_history", "cate_history"]
        if target_features is None:
            target_features = ["item_id", "cate_id"]
        self.feature_map = feature_map
        self.target_features = target_features
        self.sequence_features = sequence_features
        self.user_instant_interest_features = [feature for feature in feature_map["features"].keys() if
                                               feature not in [*target_features, *sequence_features,
                                                               *target_pretrained_multimodal_embeddings,
                                                               *sequence_pretrained_multimodal_embeddings] and
                                               feature_map["features"][feature].get("type") != 'meta']
        self.target_pretrained_multimodal_embeddings = target_pretrained_multimodal_embeddings
        self.sequence_pretrained_multimodal_embeddings = sequence_pretrained_multimodal_embeddings
        self.embedding_dim = embedding_dim
        self.attention_in_dim = embedding_dim * len(self.target_features)

        # Embedding layer
        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )

        # Target Attention module
        self.attention_layer = TargetAttention(
            embedding_dim=self.attention_in_dim,
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

        # project sequence features to the same dimension as target features if they are not
        if len(sequence_features) != len(target_features):
            self.proj_sequence_features = nn.Linear(embedding_dim * len(sequence_features), self.attention_in_dim)
        if (self.sequence_pretrained_multimodal_embeddings and
                len(sequence_pretrained_multimodal_embeddings) != len(target_pretrained_multimodal_embeddings)):
            self.proj_sequence_pretrained_multimodal_embeddings = nn.Linear(
                embedding_dim * len(sequence_pretrained_multimodal_embeddings),
                embedding_dim * len(target_pretrained_multimodal_embeddings))

        hifn_in_size = embedding_dim * (len([*self.user_instant_interest_features, *self.target_features,
                                             *self.target_pretrained_multimodal_embeddings]) +
                                        len(self.target_pretrained_multimodal_embeddings)) + self.attention_in_dim

        # Hierarchical Interest Fusion Network
        self.hifn = HIFN(
            hifn_input_size=hifn_in_size,
            dnn_hidden_sizes=dnn_hidden_sizes,
            dnn_hidden_activations=dnn_hidden_activations,
            dnn_dropout=dnn_dropout,
            gate_hidden_sizes=gate_hidden_sizes,
            gate_hidden_activations=gate_hidden_activations,
            gate_dropout=gate_dropout,
            use_batchnorm=use_batchnorm
        )

        # Final MLP Block
        self.dnn = MLPBlock(
            input_size=hifn_in_size,
            hidden_sizes=dnn_hidden_sizes,
            hidden_activations=dnn_hidden_activations,
            output_size=dnn_output_size,
            output_activation=dnn_output_activation,
            dropout_probs=dnn_dropout,
            use_batchnorm=use_batchnorm
        )

        self.apply(self.init_weights)

    def forward(self, inputs):
        # Extract embeddings
        feature_emb_dict = self.embedding_layer(inputs)

        # User Invariant Interest Modeling
        # Apply standard attention to target and sequence features
        context_vector = self._apply_attention(inputs,
                                               self.attention_layer,
                                               self.target_features,
                                               self.sequence_features,
                                               feature_emb_dict)

        # Concatenate user_instant_interest_features with user_invariant_interest_features
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None
                        and field not in [*self.sequence_features, *self.sequence_pretrained_multimodal_embeddings]]
        feature_list.append(context_vector)

        # If multimodal embeddings are provided, apply attention for them
        if self.target_pretrained_multimodal_embeddings:
            multimodal_context_vector = self._apply_attention(inputs,
                                                              self.multimodal_attention_layer,
                                                              self.target_pretrained_multimodal_embeddings,
                                                              self.sequence_pretrained_multimodal_embeddings,
                                                              feature_emb_dict)
            feature_list.append(multimodal_context_vector)

        feature_emb = torch.cat(feature_list, dim=-1)

        e_output = self.hifn(feature_emb)  # Hierarchical Interest Fusion Network
        # Pass through DNN for final prediction
        y_pred = self.dnn(e_output)
        return y_pred

    def _apply_attention(self, X, attention_layer, target_fields, sequence_fields, feature_emb_dict):
        """Helper to apply attention"""
        seq_field = sequence_fields[0]  # pick the first sequence field
        mask = X[seq_field].long() != 0  # padding_idx = 0 required

        target_emb = torch.cat([feature_emb_dict[field] for field in target_fields],
                               dim=-1)  # (batch_size, embedding_dim * len(target_fields)
        sequence_emb = torch.cat([feature_emb_dict[field] for field in sequence_fields],
                                 dim=-1)  # (batch_size, seq_len, embedding_dim) * len(sequence_fields)

        # project sequence features to the same dimension as target features if they are not
        if len(sequence_fields) != len(target_fields):
            sequence_emb = self.proj_sequence_features(sequence_emb)

        context_vector = attention_layer(target_emb, sequence_emb, mask)

        return context_vector

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
