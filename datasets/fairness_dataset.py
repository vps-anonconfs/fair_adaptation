import numpy as np
import random


class FairnessDataset:
    """
    Base dataset for implementing any other dataset with fairness specifications
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def get_subset(self, split, frac=1):
        """
        :param frac: data fraction to retain
        :param split: data split to return
        :return: torch dataset corresponding to the provided split
        """
        if split not in self.split_dict:
            raise ValueError(f"Split {split} not found in dataset's split_dict.")

        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]
        rng = random.Random(self.random_state)
        rng.shuffle(split_idx)
        ln = int(len(split_idx)*frac)
        split_idx = split_idx[:ln]

        return FairnessSubset(self, split_idx)

    @property
    def sensitive_features_names(self):
        raise NotImplementedError()

    @property
    def sensitive_features_indices(self):
        """
        Returns indices of the sensitive features in the input
                return none if sensitive features are not present in the input
        """
        raise NotImplementedError()

    @property
    def n_classes(self):
        raise NotImplementedError()

    @property
    def y_array(self):
        raise NotImplementedError()


class FairnessSubset(FairnessDataset):
    @property
    def n_classes(self):
        pass

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, y

    def __len__(self):
        return len(self.indices)

    @property
    def sensitive_features_names(self):
        return self.dataset.sensitive_feature_names

    @property
    def sensitive_features_indices(self):
        return self.dataset.sensitive_feature_indices

    @property
    def y_array(self):
        return self.dataset.y_array
