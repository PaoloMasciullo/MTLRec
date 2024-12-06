import torch
from torch import nn

from src.layers.mlp_block import MLPBlock
from src.layers.representational_layer import RepresentationalLayer


class CGC_Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 num_tasks=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 dropout_probs=0.5,
                 use_batchnorm=True
                 ):
        """

        :param num_shared_experts:
        :param num_specific_experts:
        :param num_tasks:
        :param input_dim:
        :param expert_hidden_units:
        :param gate_hidden_units:
        :param hidden_activations:
        :param dropout_probs:
        :param use_batchnorm:
        """
        super(CGC_Layer, self).__init__()

        # Shared experts
        self.shared_experts = nn.ModuleList([
            MLPBlock(
                input_size=input_dim,
                hidden_sizes=expert_hidden_units,
                hidden_activations=hidden_activations,
                dropout_probs=dropout_probs,
                use_batchnorm=use_batchnorm
            )
            for _ in range(num_shared_experts)
        ])

        # Task-specific experts
        self.specific_experts = nn.ModuleList([
            nn.ModuleList([
                MLPBlock(
                    input_size=input_dim,
                    hidden_sizes=expert_hidden_units,
                    hidden_activations=hidden_activations,
                    dropout_probs=dropout_probs,
                    use_batchnorm=use_batchnorm
                )
                for _ in range(num_specific_experts)
            ]) for _ in range(num_tasks)
        ])

        # Gating networks
        self.gates = nn.ModuleList([
            MLPBlock(
                input_size=input_dim,
                hidden_sizes=gate_hidden_units,
                hidden_activations=hidden_activations,
                output_size=num_specific_experts + num_shared_experts if i < num_tasks else num_shared_experts,
                output_activation="Softmax",
                dropout_probs=dropout_probs,
                use_batchnorm=use_batchnorm
            )
            for i in range(num_tasks + 1)
        ])

    def forward(self, x):
        # Generate outputs from specific and shared experts
        specific_outputs = [[expert(x[i]) for expert in task_experts] for i, task_experts in
                            enumerate(self.specific_experts)]
        shared_outputs = [expert(x[-1]) for expert in self.shared_experts]

        # Apply gating mechanism
        outputs = []
        for i in range(len(self.gates)):
            if i < len(self.specific_experts):  # Task-specific gating
                gate_input = torch.stack(specific_outputs[i] + shared_outputs, dim=1)
            else:  # Shared gating
                gate_input = torch.stack(shared_outputs, dim=1)

            gate_scores = self.gates[i](x[i if i < len(self.specific_experts) else -1])
            gated_output = torch.sum(gate_scores.unsqueeze(-1) * gate_input, dim=1)
            outputs.append(gated_output)

        return outputs


class PLE(nn.Module):
    def __init__(self,
                 feature_map,
                 embedding_dim=10,
                 num_tasks=1,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 tower_output_sizes=[1],
                 tower_output_activations=["Sigmoid"],
                 hidden_activations="ReLU",
                 dropout_probs=0.5,
                 use_batchnorm=True
                 ):
        """

        :param feature_map:
        :param embedding_dim:
        :param num_tasks:
        :param num_layers:
        :param num_shared_experts:
        :param num_specific_experts:
        :param expert_hidden_units:
        :param gate_hidden_units:
        :param tower_hidden_units:
        :param hidden_activations:
        :param dropout_probs:
        :param use_batchnorm:
        """
        super(PLE, self).__init__()
        # Embedding layers
        self.feature_map = feature_map
        self.num_tasks = num_tasks
        assert len(self.feature_map["targets"]) == self.num_tasks, \
            "The number of targets must be equal to the number of tasks"

        self.embedding_layer = RepresentationalLayer(
            feature_map, embedding_dim
        )
        # CGC Layers
        self.cgc_layers = nn.ModuleList([
            CGC_Layer(num_shared_experts=num_shared_experts,
                      num_specific_experts=num_specific_experts,
                      num_tasks=self.num_tasks,
                      input_dim=embedding_dim * sum(
                          1 for info in feature_map["features"].values() if info.get('type') != 'meta') if i == 0 else
                      expert_hidden_units[-1],
                      expert_hidden_units=expert_hidden_units,
                      gate_hidden_units=gate_hidden_units,
                      hidden_activations=hidden_activations,
                      dropout_probs=dropout_probs,
                      use_batchnorm=use_batchnorm)
            for i in range(num_layers)
        ])

        # Task-specific towers
        self.towers = nn.ModuleList([
            MLPBlock(
                input_size=expert_hidden_units[-1],
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

        # Pass through each CGC layer
        cgc_outputs = [feature_emb for _ in range(self.num_tasks + 1)]  # Shared input for all tasks
        for cgc_layer in self.cgc_layers:
            cgc_outputs = cgc_layer(cgc_outputs)

        # Pass through each tower for task-specific outputs
        targets = self.feature_map["targets"]
        task_outputs = {f"{targets[i]}_pred": tower(cgc_outputs[i]) for i, tower in enumerate(self.towers)}
        return task_outputs

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
