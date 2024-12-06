import io
import json

import numpy as np
import torch
import torch.nn as nn


class PretrainedEmbeddings(nn.Module):
    def __init__(self, feature_map, info, feature, embedding_dim):
        super(PretrainedEmbeddings, self).__init__()
        # load pretrained embeddings
        npz = np.load(info["pretrained_emb"])
        embeddings, keys = npz["value"], npz["key"]

        # load feature vocab
        with io.open(feature_map["data_dir"] + "vocab.json", "r", encoding="utf-8") as fd:
            vocab = json.load(fd)[feature]
            vocab_type = type(list(vocab.items())[1][0])  # get key dtype

        if info["freeze_emb"]:
            embedding_matrix = np.zeros((info["vocab_size"], info["pretrain_dim"]))
        else:
            embedding_matrix = np.random.normal(loc=0, scale=1.e-4, size=(info["vocab_size"], info["pretrain_dim"]))
            if info.get("pad_index", None):
                embedding_matrix[info.get("pad_index", None), :] = np.zeros(info["pretrain_dim"])  # set as zero for PAD

        keys = keys.astype(vocab_type)  # ensure the same dtype between pretrained keys and vocab keys
        for idx, word in enumerate(keys):
            if word in vocab:
                embedding_matrix[vocab[word]] = embeddings[idx]

        self.embedding_layer = nn.Embedding(info["vocab_size"],
                                                      info["pretrain_dim"],
                                                      padding_idx=info.get("pad_index", None))

        self.embedding_layer.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())
        if info["freeze_emb"]:
            self.embedding_layer.weight.requires_grad = False

        self.project_embeddings = nn.Linear(info["pretrain_dim"], embedding_dim)

        self.apply(self.init_weights)

    def forward(self, feature):
        emb = self.embedding_layer(feature)
        #emb = self.project_embeddings(emb)
        return emb

    @staticmethod
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv1d, nn.Conv2d]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)


class RepresentationalLayer(nn.Module):
    def __init__(self, feature_map, embedding_dim):
        super(RepresentationalLayer, self).__init__()
        self.feature_map = feature_map
        self.embedding_layers = nn.ModuleDict()
        self.pooling_info = {}

        # Initialize embeddings and pooling configurations
        for feature, info in self.feature_map["features"].items():
            # Shared embeddings
            if "share_embedding" in info and info["share_embedding"] in self.embedding_layers:
                self.embedding_layers[feature] = self.embedding_layers[info["share_embedding"]]
                continue

            # Pretrained embeddings
            if "pretrained_emb" in info:
                self.embedding_layers[feature] = PretrainedEmbeddings(feature_map, info, feature, embedding_dim)
                continue

            # Categorical embeddings
            if info["type"] == "categorical":
                self.embedding_layers[feature] = nn.Embedding(info["vocab_size"],
                                                              embedding_dim,
                                                              padding_idx=info.get("pad_index", None))

            # Sequential features with pooling
            elif info["type"] == "sequence":
                self.embedding_layers[feature] = nn.Embedding(info["vocab_size"],
                                                              embedding_dim,
                                                              padding_idx=info.get("pad_index", None))
                if info.get("pooling") is not None:
                    self.pooling_info[feature] = info["pooling"]

            # Numerical features
            elif info["type"] == "numerical":
                self.embedding_layers[feature] = nn.Linear(1, embedding_dim, bias=False)

        self.init_weights()

    def forward(self, X):
        embeddings = {}

        for feature, emb_layer in self.embedding_layers.items():
            emb = emb_layer(X[feature])

            # Pooling only for sequential features
            if feature in self.pooling_info:
                if self.pooling_info[feature] == "sum":
                    embeddings[feature] = emb.sum(dim=1)
                elif self.pooling_info[feature] == "average":
                    mask = emb.sum(dim=-1) != 0  # zeros at padding tokens
                    embeddings[feature] = emb.sum(dim=1) / (mask.float().sum(-1, keepdim=True) + 1e-12)
            else:
                embeddings[feature] = emb

        return embeddings

    def init_weights(self):
        for feature, emb_layer in self.embedding_layers.items():
            if "share_embedding" in self.feature_map["features"][feature]:
                continue
            if "pretrained_emb" in self.feature_map["features"][feature]:
                continue
            elif type(emb_layer) == nn.Embedding:
                if emb_layer.padding_idx is not None:
                    nn.init.normal_(emb_layer.weight[1:, :], std=1e-4)
                else:
                    nn.init.normal_(emb_layer.weight, std=1e-4)

