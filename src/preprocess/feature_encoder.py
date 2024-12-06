import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelEncoder:
    def __init__(self, null_value="", min_freq=1, oov_token="__OOV__", pad_token="__PAD__"):
        self.null_value = null_value
        self.min_freq = min_freq
        self.oov_token = oov_token
        self.oov_index = None
        self.pad_token = pad_token
        self.pad_index = None
        self.class_to_index = {}  # Mapping from class labels to indices
        self.index_to_class = []  # Set of classes ordered by their indices

    def fit(self, y: pl.Series):
        # Count occurrences efficiently using Polars
        counts = y.value_counts(sort=True)

        # Filter by frequency and exclude null_value
        valid_classes = counts.filter(
            (counts[counts.columns[1]] >= self.min_freq) & (counts[counts.columns[0]] != self.null_value)
        )[counts.columns[0]].to_list()

        # Build initial vocabulary with unique valid classes
        self.index_to_class = valid_classes
        self.build_vocab()
        return self

    def transform(self, y):
        # Map each label to its index, using pad_index for null_value and oov_index for unknowns
        return np.array([self.class_to_index.get(label, self.oov_index) if label != self.null_value else self.pad_index
                         for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def merge_vocabulary(self, other_vocabulary):
        # Merge with existing vocabulary
        new_words = 0
        for word in other_vocabulary:
            if word not in self.index_to_class:
                self.class_to_index[word] = self.class_to_index[self.oov_token] + new_words
                new_words += 1
        self.class_to_index[self.oov_token] = max(self.class_to_index.values()) + 1
        self.index_to_class = [label for label in self.class_to_index.values()]
        self.oov_index = self.class_to_index[self.oov_token]
        return self

    def build_vocab(self):
        if self.pad_token in self.index_to_class:
            self.index_to_class.remove(self.pad_token)
        if self.oov_token in self.index_to_class:
            self.index_to_class.remove(self.oov_token)

        self.index_to_class = [self.pad_token] + self.index_to_class  # Ensure pad token is the first index
        self.index_to_class.append(self.oov_token)  # Ensure oov token is the last index

        # Build the dictionaries
        self.class_to_index = {label: idx for idx, label in enumerate(self.index_to_class)}
        self.pad_index = self.class_to_index[self.pad_token]
        self.oov_index = self.class_to_index[self.oov_token]

    def get_vocab(self):
        return self.class_to_index
