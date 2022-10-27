from .tabular_dataset_utils import TabularDataset, encode

from typing import List
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import random
import torch
VAL_SIZE, TEST_SIZE = 0.2, 0.2


class CreditDataset(TabularDataset):
    def __init__(self, path: List[str], sensitive_features: List[str], drop_columns: List[str], use_sens,
                 shift: float = 0,  label_shift: float = 0, random_state: int = 42, override_ys=[], override_xs=[]):
        credit_data = fetch_openml(data_id=42477, as_frame=True, data_home=path)

        # Force categorical data do be dtype: category
        features = credit_data.data

        # todo
        """
        warnings in the line: features[cf] = features[cf].astype(str).astype('category')
        /homes/vp421/repos/FairnessAdaptation/datasets/credit.py:20: SettingWithCopyWarning: 
        A value is trying to be set on a copy of a slice from a DataFrame.
        Try using .loc[row_indexer,col_indexer] = value instead
        """
        categorical_features = ['x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
        for cf in categorical_features:
            features[cf] = features[cf].astype(str).astype('category')

        # Encode output
        target = (credit_data.target == "1") * 1
        target = pd.DataFrame({'target': target})

        self.x_df, self.y_df = encode(features, target)

        self.sensitive_features = sensitive_features
        excluded_columns = drop_columns
        if not use_sens:
            excluded_columns = drop_columns + self.sensitive_features
        excluded_columns = [c for dc in excluded_columns for c in self.x_df.columns.tolist() if c.startswith(dc)]
        self.x_df = self.x_df.drop(columns=excluded_columns)

        self.sensitive_feature_indices = None
        if use_sens:
            self.sensitive_feature_indices = [self.x_df.columns.tolist().index(c) for c in self.sensitive_features]
        self.x, self.y = self.x_df.to_numpy(), self.y_df.to_numpy()
        self.x, self.y = torch.Tensor(self.x).float(), torch.squeeze(torch.Tensor(self.y).to(torch.int64))

        self.random_state = random_state
        if shift > 0:
            self.x = torch.Tensor(self._shift(self.x.numpy(), factor=shift)).float()
        if label_shift > 0:
            self.x, self.y = self._label_shift(self.x, self.y, label_shift)

        rng = random.Random(random_state)
        self.split_array = np.array(rng.choices([0, 1, 2], weights=[1 - VAL_SIZE - TEST_SIZE, VAL_SIZE, TEST_SIZE],
                                                k=len(self.x)))
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}
        
        if(len(override_xs) != 0):
            split_mask = self.split_array == self.split_dict['train']
            split_idx = np.where(split_mask)[0]
            new_x = torch.Tensor(override_xs[split_idx]).float()
            self.x[split_idx] = new_x
            
        if(len(override_ys) != 0):
            split_mask = self.split_array == self.split_dict['train']
            split_idx = np.where(split_mask)[0]
            new_y = torch.argmax(torch.Tensor(override_ys[split_idx]), axis=1)
            self.y[split_idx] = torch.squeeze(new_y).to(torch.int64)



        


if __name__ == '__main__':
    # 23364 of class 0, and 6636 of class 1
    # constant classifier (0): 0.7788
    dataset_kwargs = {
        'path': 'cache/credit.csv',
        'sensitive_features': ['x2_1.0', 'x2_2.0'],
        'drop_columns': [],
        'use_sens': False,
    }
    dat = CreditDataset(**dataset_kwargs)
    print(f"""
    Number of columns: {dat[0][0].size}
    Column names: {dat.x_df.columns.tolist()}
    Total size: {len(dat)}
    Size of splits: {len(dat.get_subset('train'))}, {len(dat.get_subset('val'))}, {len(dat.get_subset('test'))}
    Sensitive features: {dat.sensitive_feature_indices, dat.sensitive_features_names}
    Labels: {np.unique(dat.y_array, return_counts=True)}
    """)
