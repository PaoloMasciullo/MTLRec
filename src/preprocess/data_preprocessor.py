import gc
import json
import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .feature_encoder import LabelEncoder
import polars as pl
import pickle


class DataProcessor:
    def __init__(self, feature_types, target_cols, group_id=None, save_dir='./saved_data/'):
        self.save_dir = save_dir

        self.feature_types = feature_types
        if isinstance(target_cols, list):  # for multitask training
            self.target_cols = target_cols
        else:
            self.target_cols = [target_cols]
        self.group_id = group_id
        self.feature_map = {"features": {}, "targets": [target["name"] for target in self.target_cols]}  # Maps feature names to metadata
        if self.group_id:
            self.feature_map["group_id"] = self.group_id
        self.encoders = {}
        self.scalers = {}

    def process_from_files(self, train_file, valid_file, test_file):
        print("Preprocessing data...")
        for split, split_file in zip(["train", "valid", "test"], [train_file, valid_file, test_file]):
            df = self.preprocess(pl.read_csv(split_file))
            if split == "train":
                data = self.fit_transform(df)
            else:
                data = self.transform(df)
            del df
            gc.collect()
            self._save_data(split, data)
            del data
            gc.collect()

    def preprocess(self, df: pl.DataFrame):
        cols = [target["name"] for target in self.target_cols]
        for feature in self.feature_types:
            cols.append(feature['name'])
            if feature['name'] in df.columns:
                if feature["type"] == "categorical":
                    df = df.with_columns(pl.col(feature['name']).cast(pl.String).fill_null(""))
                elif feature["type"] == "numerical":
                    df = df.with_columns(pl.col(feature['name']).fill_null(0))
            if feature.get('mapped_feature') in df.columns:
                df = df.with_columns(pl.col(feature['mapped_feature']).alias(feature['name']))
                df = df.with_columns(pl.col(feature['name']).cast(pl.String).fill_null(""))
        df = df.select(cols)
        return df

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit(self, df: pl.DataFrame):
        print("Fit data processor...")
        for feature in self.feature_types:
            print(f"    - Fitting feature: {feature}")
            if feature["type"] == "categorical":
                le = self._fit_categorical(df, feature, min_freq=feature["min_freq"])
                if feature.get("pretrained_emb"):
                    self._fit_pretrained(feature, le)
                if feature.get("share_embedding"):
                    self._handle_shared_embedding(feature, le)
            elif feature["type"] == "numerical":
                self._fit_numerical(df, feature)
                if feature.get("share_embedding"):
                    self.feature_map["features"][feature["name"]]["share_embedding"] = feature["share_embedding"]
            elif feature["type"] == "sequence":
                le = self._fit_sequence(df, feature, min_freq=feature["min_freq"])
                if feature.get("pretrained_emb"):
                    self._fit_pretrained(feature, le)
                if feature.get("share_embedding"):
                    self._handle_shared_embedding(feature, le)
            elif feature["type"] == "meta":
                self._fit_meta(feature)
            else:
                raise NotImplementedError("feature type={}".format(feature["type"]))
        self._save_processor()
        self._save_vocab()

    def transform(self, df: pl.DataFrame):
        print("Transform features...")
        transformed_data = {}
        for feature in self.feature_types:
            print(f"    - Transforming feature: {feature}")
            if feature["type"] == "categorical":
                transformed_data[feature["name"]] = self._transform_categorical(df, feature)
            elif feature["type"] == "numerical":
                transformed_data[feature["name"]] = self._transform_numerical(df, feature)
            elif feature["type"] == "sequence":
                transformed_data[feature["name"]] = self._transform_sequence(df, feature)
            elif feature["type"] == "meta":
                transformed_data[feature["name"]] = torch.tensor(df[feature["name"]], dtype=torch.long)
            else:
                raise NotImplementedError("feature type={}".format(feature["type"]))
        for target in self.target_cols:
            if target["type"] == "binary":  # 0 or 1
                transformed_data[target["name"]] = torch.tensor(df[target["name"]].to_numpy(), dtype=torch.float32)
            elif target["type"] == "multiclass":  # 0, 1, 2, 3, ..., n
                transformed_data[target["name"]] = torch.tensor(df[target["name"]], dtype=torch.long)
            elif target["type"] == "regression":  # continuous values
                transformed_data[target["name"]] = torch.tensor(df[target["name"]], dtype=torch.float32)
            elif target["type"] == "binary-vector":  # [1, 1, 0, 1]
                sequences = df[target["name"]].fill_null("").str.split(target["splitter"]).to_list()
                sequences = [[int(el) for el in seq] for seq in sequences]
                transformed_data[target["name"]] = torch.tensor(sequences)
            else:
                raise NotImplementedError("target type={}".format(target["type"]))
        return transformed_data

    def _fit_categorical(self, df, feature, min_freq):
        le = LabelEncoder(min_freq=min_freq)
        self.encoders[feature["name"]] = le.fit(df[feature["name"]])
        self.feature_map["features"][feature["name"]] = {
            "type": feature["type"],
            "oov_index": le.oov_index,
            "pad_index": le.pad_index,
            "vocab_size": len(le.index_to_class)
        }
        return le

    def _fit_numerical(self, df, feature):
        scaler = StandardScaler()
        scaler.fit(df[[feature["name"]]])
        self.scalers[feature["name"]] = scaler
        self.feature_map["features"][feature["name"]] = {"type": feature["type"]}
        return scaler

    def _fit_sequence(self, df, feature, min_freq):
        le = LabelEncoder(min_freq=min_freq)
        # Extract unique values after splitting each entry by the specified splitter
        all_values = (df[feature["name"]]
                      .drop_nulls()  # Remove missing values
                      .str.split(feature["splitter"])  # Split strings by the specified splitter
                      .explode())  # Flatten the list of lists
        self.encoders[feature["name"]] = le.fit(all_values)
        self.feature_map["features"][feature["name"]] = {
            "type": feature["type"],
            "pooling": feature['pooling'] if 'pooling' in feature else None,
            "oov_index": le.oov_index,
            "pad_index": le.pad_index,
            "vocab_size": len(le.index_to_class)
        }
        return le

    def _fit_meta(self, feature):
        self.feature_map["features"][feature["name"]] = {"type": feature['type']}

    def _fit_pretrained(self, feature, le):
        keys = np.load(feature["pretrained_emb"])['key']
        self.encoders[feature["name"]] = le.merge_vocabulary(keys)
        # Store information in the feature map
        self.feature_map["features"][feature["name"]].update({
            "type": feature["type"],
            "pretrained_emb": feature["pretrained_emb"],
            "freeze_emb": feature["freeze_emb"],
            "pretrain_dim": feature["pretrain_dim"],
            "oov_index": le.oov_index,
            "pad_index": le.pad_index,
            "vocab_size": len(le.index_to_class),
        })

    def _handle_shared_embedding(self, feature, current_encoder):
        """
        Handles shared embeddings logic to create a unified vocabulary for the source and target features.
        """
        shared_feature_name = feature["share_embedding"]
        self.feature_map["features"][feature["name"]]["share_embedding"] = shared_feature_name

        # Check if the shared feature has already been processed
        if shared_feature_name not in self.encoders:
            raise ValueError(
                f"Shared embedding source '{shared_feature_name}' for feature '{feature['name']}' "
                "is missing or not processed. Ensure it is defined earlier in the feature list."
            )

        # Get the existing label encoder for the shared feature
        shared_encoder = self.encoders[shared_feature_name]

        # Merge the current encoder's vocabulary into the shared encoder
        # Update both the target feature and the shared feature to use the same encoder
        self.encoders[feature["name"]] = shared_encoder.merge_vocabulary(current_encoder.index_to_class)
        self.encoders[shared_feature_name] = shared_encoder
        self.feature_map["features"][feature["name"]]["vocab_size"] = len(shared_encoder.index_to_class)
        self.feature_map["features"][shared_feature_name]["vocab_size"] = len(shared_encoder.index_to_class)

    def _transform_categorical(self, df, feature):
        le = self.encoders[feature["name"]]
        return torch.tensor(le.transform(df[feature["name"]]))

    def _transform_sequence(self, df, feature):
        le = self.encoders[feature["name"]]
        pad_index = le.pad_index  # Get the pad index from the LabelEncoder
        sequences = df[feature["name"]].fill_null("").str.split(feature["splitter"]).to_list()
        transformed_sequences = [le.transform(seq) for seq in sequences]

        # Initialize the padded sequences with the pad index
        padded_sequences = np.full((len(transformed_sequences), feature["max_len"]), pad_index, dtype=int)

        for i, seq in enumerate(transformed_sequences):
            trunc_seq = seq[-feature["max_len"]:]  # Truncate to max_len
            padded_sequences[i, :len(trunc_seq)] = trunc_seq  # Fill the sequence

        return torch.tensor(padded_sequences)

    def _transform_numerical(self, df, feature):
        scaler = self.scalers[feature["name"]]
        return torch.tensor(scaler.transform(df[[feature["name"]]]), dtype=torch.float32)

    def _save_vocab(self):
        os.makedirs(self.save_dir, exist_ok=True)
        vocab_file = self.save_dir + 'vocab.json'
        vocab = dict()
        for feature in self.feature_types:
            if feature["type"] in ["categorical", "sequence"]:
                vocab[feature["name"]] = self.encoders[feature["name"]].class_to_index
        with open(vocab_file, "w") as fd:
            fd.write(json.dumps(vocab, indent=4))
        print(f"Vocabulary and processed data saved to {vocab_file}")

    def _save_processor(self):
        """Save the DataProcessor to a file."""
        os.makedirs(self.save_dir, exist_ok=True)
        processor_save_path = self.save_dir + 'data_processor.pkl'
        self.feature_map["data_dir"] = self.save_dir

        save_data = {
            "feature_types": self.feature_types,
            "target_col": self.target_cols,
            "group_id": self.group_id,
            "feature_map": self.feature_map,
            "label_encoders": self.encoders,
            "scalers": self.scalers,
            "save_dir": self.save_dir
        }
        with open(processor_save_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"DataProcessor saved to {processor_save_path}")

    def _save_data(self, split, transformed_data):
        """Save the transformed_data to a file"""
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = self.save_dir + split + ".pkl"
        with open(save_file, "wb") as f:
            pickle.dump(transformed_data, f)

    def load_data(self, split):
        save_file = self.save_dir + split + ".pkl"

        with open(save_file, "rb") as f:
            save_data = pickle.load(f)
        return save_data

    @classmethod
    def load_processor(cls, dir_path):
        """
        Load the DataProcessor and optionally processed data from a file.
        :param dir_path: Directory to load the DataProcessor.
        :return: DataProcessor instance.
        """
        processor_save_path = dir_path + 'data_processor.pkl'

        with open(processor_save_path, "rb") as f:
            save_data = pickle.load(f)

        # Recreate the DataProcessor instance
        processor = cls(
            feature_types=save_data["feature_types"],
            target_cols=save_data["target_col"],
            group_id=save_data["group_id"],
            save_dir=dir_path
        )
        processor.feature_map = save_data["feature_map"]
        processor.encoders = save_data["label_encoders"]
        processor.scalers = save_data["scalers"]

        # Return the processor
        return processor
