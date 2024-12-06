import torch
from torch import nn

from src import TargetAttention
from src.layers.mlp_block import MLPBlock
from src.layers.representational_layer import RepresentationalLayer


class SharedBottom(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 attention_hidden_sizes=[64],
                 attention_hidden_activations="Dice",
                 target_features=["item_id", "cate_id"],
                 sequence_features=["click_history", "cate_history"],
                 target_pretrained_multimodal_embeddings=None,  # ["item_id_img", "item_id_text"]
                 sequence_pretrained_multimodal_embeddings=None,  # ["hist_id_img", "hist_id_text"]
                 attention_dropout=0.5,
                 use_softmax=False,
                 num_tasks=1,
                 bottom_hidden_units=[512, 256, 128],
                 tower_hidden_units=[128, 64],
                 tower_output_sizes=[1],
                 tower_output_activations=["Sigmoid"],
                 hidden_activations="ReLU",
                 dropout_probs=0.5,
                 use_batchnorm=True
                 ):
        super(SharedBottom, self).__init__()
        # Embedding layers
        self.embedding_dim = embedding_dim
        self.feature_map = feature_map
        self.target_features = target_features
        self.sequence_features = sequence_features
        self.target_pretrained_multimodal_embeddings = target_pretrained_multimodal_embeddings
        self.sequence_pretrained_multimodal_embeddings = sequence_pretrained_multimodal_embeddings
        self.num_tasks = num_tasks
        assert len(self.feature_map["targets"]) == self.num_tasks, \
            "The number of targets must be equal to the number of tasks"
        assert len(self.target_features) == len(self.sequence_features), \
            "len(target_features) != len(sequence_features)"

        if self.target_pretrained_multimodal_embeddings:
            assert len(self.target_pretrained_multimodal_embeddings) == len(
                self.sequence_pretrained_multimodal_embeddings), \
                "len(target_pretrained_multimodal_embeddings) != len(sequence_pretrained_multimodal_embeddings)"

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

        self.bottom = MLPBlock(input_size=embedding_dim * sum(
                          1 for info in feature_map["features"].values() if info.get('type') != 'meta'),
                                hidden_sizes=bottom_hidden_units,
                                hidden_activations=hidden_activations,
                                dropout_probs=dropout_probs,
                                use_batchnorm=use_batchnorm)

        # Task-specific towers
        self.towers = nn.ModuleList([
            MLPBlock(
                input_size=bottom_hidden_units[-1],
                hidden_sizes=tower_hidden_units,
                hidden_activations=hidden_activations,
                output_size=tower_output_sizes[i],
                output_activation=tower_output_activations[i],
                dropout_probs=dropout_probs,
                use_batchnorm=use_batchnorm
            )
            for i in range(self.num_tasks)
        ])

        self.apply(self.init_weights)

    def forward(self, inputs):
        feature_emb_dict = self.embedding_layer(inputs)
        # Apply standard attention to target and sequence features
        self._apply_attention(inputs, self.attention_layer, self.target_features, self.sequence_features, feature_emb_dict)

        # If multimodal embeddings are provided, apply attention for them
        if self.target_pretrained_multimodal_embeddings:
            self._apply_attention(inputs,
                                  self.multimodal_attention_layer,
                                  self.target_pretrained_multimodal_embeddings,
                                  self.sequence_pretrained_multimodal_embeddings,
                                  feature_emb_dict)

        # Combine all features for DNN
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None]
        feature_emb = torch.cat(feature_list, dim=-1)

        bottom_output = self.bottom(feature_emb)

        # Pass through each tower for task-specific outputs
        targets = self.feature_map["targets"]
        task_outputs = {f"{targets[i]}_pred": tower(bottom_output) for i, tower in enumerate(self.towers)}
        return task_outputs

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
