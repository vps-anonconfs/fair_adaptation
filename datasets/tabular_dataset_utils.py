import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List
import random
import numpy as np
import torch
from typing import Union
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

from .fairness_dataset import FairnessDataset

VAL_SIZE, TEST_SIZE = 0.2, 0.2


class TabularDataset(FairnessDataset):
    def __init__(self, path: str, sensitive_features: List[str], drop_columns: List[str], use_sens, shift: float = 0,
                 label_shift: float = 0, random_state: int = 42, override_ys=[], override_xs=[]):
        """
        :param path: path to the local file
        :param sensitive_features: list of sensitive features
        :param drop_columns: feature names that are to be dropped when loading
        :param use_sens: when set to true, retains the sensitive columns in the data
        :param shift: a value between 0 and 1, where higher value indicated greater shift
        :param label_shift: a value between 0 and 1, where higher value indicated greater label shift
        :param random_state: random state for train-val-test splitting, etc.
        """
        self.sensitive_features = sensitive_features
        self.sensitive_feature_indices = None

        exclude_columns = drop_columns
        if not use_sens:
            exclude_columns = drop_columns + self.sensitive_features
        self.x_df, self.y_df = load_tabular_dataset(path, target_column_name='target', drop_columns=exclude_columns)
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

    def _shift(self, input_arr: np.ndarray, factor: float):
        """
        Simulates covariate shift by removing support of input distribution along certain dimensions
        :param input_arr: inputs to be shifted
        :param factor: A value that controls the number of dimensions that are projected out
        returns the shifted distribution
        """
        assert 1 >= factor >= 0
        return shift_by_projecting_out(input_arr, factor, self.random_state)

    def _label_shift(self, input_arr: Union[torch.Tensor, np.array], output_arr: Union[torch.Tensor, np.array],
                     factor: float):
        """
        simulates label shift by re-assigning labels
        :param output_arr: categorical class variable
        :param factor: the amount by which the major class probability changes
               for eg. if class_prob=[0.9, 0.1] and shift is 0.5, then new class_prob=[0.4, 0.6]
               if factor is >0.9 above, then it is clipped to the maximum possible value of 0.9
        returns new input and output array
        """
        assert 0 <= factor <= 1

        def npify(x):
            if type(x) == torch.Tensor:
                np_x = x.detach().cpu().numpy().copy()
            else:
                np_x = x.copy()
            return np_x

        np_input, np_output = npify(input_arr), npify(output_arr)

        rng = np.random.default_rng(self.random_state)
        classes, class_count = np.unique(np_output, return_counts=True)
        num_classes = len(classes)
        assert num_classes == 2, "Supports only binary classification!"
        class_count = class_count.astype(np.float32)
        class_count /= class_count.sum()
        maj_cls_idx = np.argmax(class_count)
        factor = min(class_count[maj_cls_idx], factor)
        new_class_count = class_count.copy()
        new_class_count[maj_cls_idx] -= factor
        num_classes = len(new_class_count)
        for idx in range(num_classes):
            if idx != maj_cls_idx:
                new_class_count[idx] += factor / (num_classes - 1)

        scaler = preprocessing.StandardScaler().fit(np_input)
        x_scaled = scaler.transform(np_input)
        clf = LogisticRegression(random_state=self.random_state).fit(x_scaled, np_output)
        probs = clf.predict_proba(np_input)[:, 0]
        _ln = int(new_class_count[0]*len(np_output))
        thresh = probs[np.argpartition(probs, _ln)[_ln]]

        new_output = (np.array(probs) > thresh).astype(np.int32)
        if type(output_arr) == torch.Tensor:
            output_arr = torch.ones_like(output_arr)*new_output
        else:
            output_arr = new_output

        _, class_count2 = np.unique(npify(output_arr), return_counts=True)
        class_count2 = class_count2.astype(np.float32)
        class_count2 /= class_count2.sum()
        print(f"Simulating label shift, old class proportions: {class_count}, new {class_count2}")

        return input_arr, output_arr

    def _label_shift2(self, input_arr: Union[torch.Tensor, np.array], output_arr: Union[torch.Tensor, np.array],
                     factor: float):
        """
        simulates label shift by re-assigning labels
        :param output_arr: categorical class variable
        :param factor: the amount by which the major class probability changes
               for eg. if class_prob=[0.9, 0.1] and shift is 0.5, then new class_prob=[0.4, 0.6]
               if factor is >0.9 above, then it is clipped to the maximum possible value of 0.9
        returns new input and output array
        """
        assert 0 <= factor <= 1

        if type(output_arr) == torch.Tensor:
            np_output = output_arr.detach().cpu().numpy()
        else:
            np_output = output_arr
        rng = np.random.default_rng(self.random_state)
        classes, class_count = np.unique(np_output, return_counts=True)
        class_count = class_count.astype(np.float32)
        class_count /= class_count.sum()
        maj_cls_idx = np.argmax(class_count)
        factor = min(class_count[maj_cls_idx], factor)
        num_classes = len(class_count)
        idxs_per_cls = [rng.permutation(np.where(np_output == cls_idx)[0]) for cls_idx in range(num_classes)]
        reassign_ln = int(factor*len(output_arr))
        all_but_major_class = np.array([_i for _i in range(num_classes) if _i != maj_cls_idx])
        for idx in idxs_per_cls[maj_cls_idx][:reassign_ln]:
            output_arr[idx] = rng.choice(all_but_major_class)

        if type(output_arr) == torch.Tensor:
            np_output2 = output_arr.detach().cpu().numpy()
        else:
            np_output2 = output_arr
        _, class_count2 = np.unique(np_output2, return_counts=True)
        class_count2 = class_count2.astype(np.float32)
        class_count2 /= class_count2.sum()
        print(f"Simulating label shift, old class proportions: {class_count}, new {class_count2}")

        return input_arr, output_arr

    def _label_shift3(self, input_arr: Union[torch.Tensor, np.array], output_arr: Union[torch.Tensor, np.array],
                     factor: float):
        """
        simulates label shift by resampling
        :param output_arr: categorical class variable
        :param factor: the amount by which the major class probability changes
               for eg. if class_prob=[0.9, 0.1] and shift is 0.5, then new class_prob=[0.4, 0.6]
               if factor is >0.9 above, then it is clipped to the maximum possible value of 0.9

        returns new input and output array
        """
        assert 0 <= factor <= 1

        if type(output_arr) == torch.Tensor:
            np_output = output_arr.detach().cpu().numpy()
        else:
            np_output = output_arr
        rng = np.random.default_rng(self.random_state)
        classes, class_count = np.unique(np_output, return_counts=True)
        class_count = class_count.astype(np.float32)
        class_count /= class_count.sum()
        maj_cls_idx = np.argmax(class_count)
        factor = min(class_count[maj_cls_idx], factor)
        new_class_count = class_count.copy()
        new_class_count[maj_cls_idx] -= factor
        num_classes = len(new_class_count)
        for idx in range(num_classes):
            if idx != maj_cls_idx:
                new_class_count[idx] += factor / (num_classes - 1)
        print(f"Simulating label shift, old class proportions: {class_count}, new {new_class_count}")
        idxs_per_cls = [np.where(np_output == cls_idx)[0] for cls_idx in range(num_classes)]
        new_cls_lns = [int(new_class_count[cls_idx]*len(output_arr)) for cls_idx in range(num_classes)]
        new_idxs = np.concatenate([
            rng.choice(idxs_per_cls[cls_idx], new_cls_lns[cls_idx], replace=True)
            for cls_idx in range(num_classes)
        ])
        return input_arr[new_idxs], output_arr[new_idxs]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    @property
    def sensitive_features_names(self):
        return self.sensitive_features

    @property
    def sensitive_features_indices(self):
        return self.sensitive_feature_indices

    @property
    def n_classes(self):
        return 2

    @property
    def y_array(self):
        return self.y.numpy()


def shift_by_projecting_out(input_arr: np.ndarray, factor: float, random_state: int):
    """
    Simulates covariate shift by removing support of input distribution along certain dimensions
    :param input_arr: inputs to be shifted
    :param factor: A value that controls the number of dimensions that are projected out
    :param random_state: Random state for reproducibility
    returns the shifted distribution
    """
    n, m = input_arr.shape
    k = int(0.75 * m)
    # draw random vectors
    rng = np.random.default_rng(random_state)
    # n x m
    new_input = input_arr.copy()
    # k x m
    U = rng.uniform(-1, 1, size=[k, m])
    U /= np.expand_dims(np.linalg.norm(U, axis=-1), axis=-1)
    for ui in range(len(U)):
        # m x 1
        u = np.reshape(U[ui], [-1, 1])
        new_input = new_input - np.matmul(np.matmul(new_input, u), np.transpose(u))

    # also drift the mean
    new_input = new_input - np.expand_dims(input_arr.mean(axis=0), axis=0)

    final = (1 - factor) * input_arr + factor * new_input
    return final

def load_tabular_dataset(csv_fname, target_column_name, drop_columns=[]):
    data_df = pd.read_csv(csv_fname, na_values='?').dropna()

    x_raw = data_df.drop(columns=[target_column_name])
    y_raw = data_df[[target_column_name]]

    x_df, y_df = encode(x_raw, y_raw)
    # TODO: for legacy reasons, sensitive_features are names after encoding and drop_columns are before encoding
    #       prefix search column names to avoid missing specified column error
    excluded_columns = [c for dc in drop_columns for c in x_df.columns.tolist() if c.startswith(dc)]
    x_df = x_df.drop(columns=excluded_columns)
    return x_df, y_df


def encode(x_raw, y_raw, drop_first=False, drop_first_labels=True):
    """
    encodes tabular dataset by converting all categorical columns in to numerical and min-max scales all features
    :param x_raw: features dataframe
    :param y_raw: labels dataframe or column label

    TODO: what's the drop_first business and why is it set to true by default for labels.
          These have no effect and should be removed.
    :param drop_first: whether to drop first when one-hot encoding features
    :param drop_first_labels: whether to drop first when one-hot encoding labels

    :returns: tuple of processed x and y as numpy arrays
    """

    x_df_all, y_df = _prepare_dataset(x_raw, y_raw, drop_first=drop_first, drop_first_labels=drop_first_labels)
    return x_df_all, y_df


def _scale_num_cols(x_df, num_cols):
    """
    :param x_df: features dataframe
    :param num_cols: name of all numerical columns to be scaled
    returns: feature dataframe with scaled numerical features
    """
    x_df_scaled = x_df.copy()
    scaler = MinMaxScaler()
    x_num = scaler.fit_transform(x_df_scaled[num_cols])
    # if (COV_SHIFT):
    #     cov_shift = np.random.uniform(-COV_SHIFT_RATE, COV_SHIFT_RATE, size=(len(x_num[0]), len(x_num[0])))
    #     I = np.eye(len(x_num[0]))
    #     cov_shift = I - cov_shift
    #     x_num = np.matmul(x_num, cov_shift)

    for i, c in enumerate(num_cols):
        x_df_scaled[c] = x_num[:, i]

    return x_df_scaled


def _process_num_cat_columns(x_df, drop_first):
    """
    :param x_df: features dataframe
    returns: feature dataframe with scaled numerical features and one-hot encoded categorical features
    """
    num_cols = []
    cat_cols = []

    for c in x_df.columns:
        if x_df[c].dtype == 'object' or x_df[c].dtype.name == 'category':
            cat_cols.append(c)
        else:
            num_cols.append(c)

    # TODO: need to think about this drop_first
    # if (COV_SHIFT):
    #     print(cat_cols)
    #     for ind in range(len(cat_cols)):
    #         # if(cat_cols[ind] == 'x4' or cat_cols[ind] == 'x3'):
    #         #    continue
    #         unique_elms = len(set(X_df[cat_cols[ind]]))
    #         elements = set(X_df[cat_cols[ind]])
    #         frequency = X_df[cat_cols[ind]].value_counts()
    #         # print(elements, frequency)
    #         frequency /= sum(X_df[cat_cols[ind]].value_counts())
    #         # New relative frequency weighting
    #         frequency += np.random.uniform(-1 * COV_SHIFT_RATE, COV_SHIFT_RATE, len(frequency))
    #         frequency = np.clip(frequency, 0.05, 1)
    #         frequency /= sum(frequency)
    #         for index, row in X_df.iterrows():
    #             if (np.random.uniform() < 3 * COV_SHIFT_RATE):
    #                 mod_val = np.random.choice(frequency.index, p=frequency)
    #                 X_df.at[index, cat_cols[ind]] = mod_val
    #             else:
    #                 continue

    x_df_encoded = pd.get_dummies(x_df, columns=cat_cols, drop_first=drop_first)

    cat_cols = list(set(x_df_encoded.columns) - set(num_cols))
    num_cols.sort()
    cat_cols.sort()

    x_df_encoded_scaled = _scale_num_cols(x_df_encoded, num_cols)

    return x_df_encoded_scaled[num_cols + cat_cols]


def _process_labels(y_df, drop_first):
    y_processed = pd.get_dummies(y_df, drop_first=drop_first)
    return y_processed


def _prepare_dataset(x_df_original, y_df_original, drop_first, drop_first_labels):
    """
    :param x_df_original: features dataframe
    :param y_df_original: labels dataframe
    returns:
        - feature dataframe with scaled numerical features and one-hot encoded categorical features
        - one hot encoded labels, with drop_first option
    """
    x_df = x_df_original.copy()

    x_processed = _process_num_cat_columns(x_df, drop_first)
    y_processed = _process_labels(y_df_original, drop_first_labels)

    return x_processed, y_processed

