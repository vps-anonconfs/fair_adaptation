from .tabular_dataset_utils import TabularDataset, load_tabular_dataset

from typing import List
import numpy as np
import pandas as pd
import random
import torch
VAL_SIZE = 0.2


class AdultDataset(TabularDataset):
    def __init__(self, path: List[str], sensitive_features: List[str], drop_columns: List[str], use_sens,
                 shift: float = 0, label_shift: float = 0, random_state: int = 42, override_ys=[], override_xs=[]):
        self.sensitive_features = sensitive_features
        self.sensitive_feature_indices = None

        assert len(path) == 2, "This routine expects two files provided: train and test"
        train_path, test_path = path
        exclude_columns = drop_columns
        if not use_sens:
            exclude_columns = drop_columns + self.sensitive_features

        self.train_x_df, self.train_y_df = load_tabular_dataset(train_path, target_column_name='target',
                                                                drop_columns=exclude_columns)
        self.test_x_df, self.test_y_df = load_tabular_dataset(test_path, target_column_name='target',
                                                              drop_columns=exclude_columns)
        self.x_df = pd.concat([self.train_x_df, self.test_x_df])
        self.y_df = pd.concat([self.train_y_df, self.test_y_df])

        if use_sens:
            self.sensitive_feature_indices = [self.x_df.columns.tolist().index(c) for c in self.sensitive_features]

        self.train_x, self.train_y = self.train_x_df.to_numpy(), self.train_y_df.to_numpy()
        self.test_x, self.test_y = self.test_x_df.to_numpy(), self.test_y_df.to_numpy()
        self.train_x, self.train_y = \
            torch.Tensor(self.train_x).float(), torch.squeeze(torch.Tensor(self.train_y).to(torch.int64))
        self.test_x, self.test_y = \
            torch.Tensor(self.test_x).float(), torch.squeeze(torch.Tensor(self.test_y).to(torch.int64))

        self.random_state = random_state
        if shift > 0:
            self.train_x = torch.Tensor(self._shift(self.train_x.numpy(), factor=shift)).float()
            self.test_x = torch.Tensor(self._shift(self.test_x.numpy(), factor=shift)).float()
        if label_shift > 0:
            # todo: assuming that the order of points is not disturbed
            self.x, self.y = self._label_shift(torch.cat([self.train_x, self.test_x]),
                                               torch.cat([self.train_y, self.test_y]), label_shift)
            self.train_x, self.train_y = self.x[:len(self.train_x)], self.y[:len(self.train_y)]
            self.test_x, self.test_y = self.x[len(self.train_x):], self.y[len(self.train_y):]

        rng = random.Random(random_state)
        self.split_array = np.array(rng.choices([0, 1], weights=[1 - VAL_SIZE, VAL_SIZE],
                                                k=len(self.train_x)))
        self.split_array = np.concatenate([self.split_array, 2*np.ones([len(self.test_x)])])
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
            
        self.x = torch.cat([self.train_x, self.test_x], dim=0)
        self.y = torch.cat([self.train_y, self.test_y], dim=0)
        
        if len(override_xs) != 0:
            split_mask = self.split_array == self.split_dict['train']
            split_idx = np.where(split_mask)[0]
            new_x = torch.Tensor(override_xs[split_idx]).float()
            self.x[split_idx] = new_x

        if len(override_ys) != 0:
            split_mask = self.split_array == self.split_dict['train']
            split_idx = np.where(split_mask)[0]
            new_y = torch.argmax(torch.Tensor(override_ys[split_idx]), axis=1)
            self.y[split_idx] = torch.squeeze(new_y).to(torch.int64)

if __name__ == '__main__':
    # 34014 of class 0, 11208 of class 1
    # constant classifier (0): 0.7521
    dataset_kwargs = {
        'path': ['cache/adult.data.csv', 'cache/adult.test.csv'],
        'sensitive_features': ['sex_Male', 'sex_Female'],
        'drop_columns': ['native-country'],
        'use_sens': False,
        'label_shift': 0.5
    }
    dat = AdultDataset(**dataset_kwargs)
    print(f"""
    Number of columns: {dat[0][0].size}
    Column names: {dat.x_df.columns.tolist()}
    Total size: {len(dat)}
    Size of splits: {len(dat.get_subset('train'))}, {len(dat.get_subset('val'))}, {len(dat.get_subset('test'))}
    Sensitive features: {dat.sensitive_feature_indices, dat.sensitive_features_names}
    Labels: {np.unique(dat.y_array, return_counts=True)}
    """)