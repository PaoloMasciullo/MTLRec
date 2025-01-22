from torch.utils.data import Dataset


class SingleTaskDataset(Dataset):
    def __init__(self, feature_map, data):
        """
        Initialize the dataset with processed data.
        :param feature_map: Dictionary containing `features` (keys of input features)
                            and `targets` (key for target label).
        :param data: Dictionary of tensors containing features and target labels.
        """
        target = feature_map['targets'][0]
        if target not in data:
            raise KeyError(f"Target '{target}' is missing from the dataset!")

        self.features = {k: v for k, v in data.items() if k in feature_map["features"]}
        self.target = data[target]

        # Assert all lengths match
        lengths = {k: len(v) for k, v in data.items()}
        assert all(length == len(self.target) for length in lengths.values()), \
            f"Feature lengths {lengths} do not match target length {len(self.target)}!"

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        features = {k: v[idx] for k, v in self.features.items()}
        target = self.target[idx]
        return features, target


class MultiTaskDataset(Dataset):
    def __init__(self, feature_map, data):
        """
        Initialize the dataset with processed data.
        :param feature_map: Dictionary containing `features` (keys of input features)
                            and `targets` (list of keys for target labels).
        :param data: Dictionary of tensors containing features and target labels.
        """
        targets = [target for target in feature_map['targets']]
        if not isinstance(targets, list):
            raise TypeError("`feature_map['targets']` must be a list of target keys.")

        # Check if all target keys exist in the data
        missing_targets = [target for target in targets if target not in data]
        if missing_targets:
            raise KeyError(f"Targets {missing_targets} are missing from the dataset!")

        self.features = {k: v for k, v in data.items() if k in feature_map["features"]}
        self.targets = {k: v for k, v in data.items() if k in targets}

        # Assert all targets and features have the same length
        target_lengths = {k: len(v) for k, v in self.targets.items()}
        feature_lengths = {k: len(v) for k, v in self.features.items()}
        all_lengths = {**target_lengths, **feature_lengths}
        expected_length = next(iter(target_lengths.values()))  # Length of the first target

        assert all(length == expected_length for length in all_lengths.values()), \
            f"Dataset length mismatch: {all_lengths}. All must match length {expected_length}."

    def __len__(self):
        return len(next(iter(self.targets.values())))

    def __getitem__(self, idx):
        features = {k: v[idx] for k, v in self.features.items()}
        targets = {k: v[idx] for k, v in self.targets.items()}
        return features, targets
