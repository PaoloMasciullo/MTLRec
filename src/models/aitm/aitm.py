"""
References:
    paper: (KDD'2021) Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising
    url: https://arxiv.org/abs/2105.08489
    code: https://github.com/adtalos/AITM-torch
"""

import torch
from torch import nn

from src.layers import AttentionLayer
from src.layers.mlp_block import MLPBlock
from src.layers.representational_layer import RepresentationalLayer


class AITM(nn.Module):
    """
    Adaptive Information Transfer Multi-task (AITM) framework.
    all the task type must be binary classificatioon.
    """
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
        super(AITM, self).__init__()
        # Embedding layers
        self.embedding_dim = embedding_dim
        self.feature_map = feature_map
        self.num_tasks = num_tasks
        assert len(self.feature_map["targets"]) == self.num_tasks, \
            "The number of targets must be equal to the number of tasks"

        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )
        self.bottoms = nn.ModuleList([
            MLPBlock(
                input_size=embedding_dim * sum(
                    1 for info in feature_map["features"].values() if info.get('type') != 'meta'),
                hidden_sizes=bottom_hidden_units,
                hidden_activations=hidden_activations,
                dropout_probs=dropout_probs,
                use_batchnorm=use_batchnorm
            )
            for _ in range(self.num_tasks)
        ])

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

        self.info_gates = nn.ModuleList([
            MLPBlock(
                input_size=bottom_hidden_units[-1],
                hidden_sizes=[bottom_hidden_units[-1]],
                use_batchnorm=use_batchnorm
            )
            for _ in range(self.num_tasks - 1)
        ])

        self.aits = nn.ModuleList([AttentionLayer(dim=bottom_hidden_units[-1]) for _ in range(self.num_tasks - 1)])

    def forward(self, inputs):
        feature_emb_dict = self.embedding_layer(inputs)
        feature_list = [feature_emb_dict[field] for field in feature_emb_dict if feature_emb_dict[field] is not None]
        feature_emb = torch.cat(feature_list, dim=-1)

        input_towers = [bottom(feature_emb) for bottom in self.bottoms]

        for i in range(1, self.num_tasks):
            info = self.info_gates[i - 1](input_towers[i - 1]).unsqueeze(1)
            ait_input = torch.cat([input_towers[i].unsqueeze(1), info], dim=1)
            input_towers[i] = self.aits[i - 1](ait_input)

        # Pass through each tower for task-specific outputs
        targets = self.feature_map["targets"]
        task_outputs = {f"{targets[i]}_pred": tower(input_tower)
                        for i, (input_tower, tower) in enumerate(zip(input_towers, self.towers))}
        return task_outputs
