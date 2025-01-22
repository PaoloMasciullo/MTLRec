"""
Reference:
    paper: Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41-75.
"""
import torch
from torch import nn

from src.layers.mlp_block import MLPBlock
from src.layers.representational_layer import RepresentationalLayer


class SharedBottom(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim=10,
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
        self.num_tasks = num_tasks
        assert len(self.feature_map["targets"]) == self.num_tasks, \
            "The number of targets must be equal to the number of tasks"

        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )

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
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None]
        feature_emb = torch.cat(feature_list, dim=-1)

        bottom_output = self.bottom(feature_emb)

        # Pass through each tower for task-specific outputs
        targets = self.feature_map["targets"]
        task_outputs = {f"{targets[i]}_pred": tower(bottom_output) for i, tower in enumerate(self.towers)}
        return task_outputs

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
