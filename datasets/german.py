from .tabular_dataset_utils import TabularDataset
import numpy as np


class GermanDataset(TabularDataset):
    def __init__(self, override_ys=[], override_xs=[], *args, **kwargs):
        super(GermanDataset, self).__init__(override_ys=override_ys, override_xs=[], *args, **kwargs)


if __name__ == '__main__':
    # 300 of class 0 and 700 of class 1
    dataset_kwargs = {
        'path': 'cache/german.csv',
        'sensitive_features': ['status_sex_A91', 'status_sex_A93', 'status_sex_A94', 'status_sex_A92'],
        'drop_columns': [],
        'use_sens': False
    }
    dat = GermanDataset(**dataset_kwargs)

    print(f"""
    Number of columns: {dat[0][0].size}
    Column names: {dat.x_df.columns.tolist()}
    Total size: {len(dat)}
    Size of splits: {len(dat.get_subset('train'))}, {len(dat.get_subset('val'))}, {len(dat.get_subset('test'))}
    Sensitive features: {dat.sensitive_feature_indices, dat.sensitive_features_names}
    Labels: {np.unique(dat.y_array, return_counts=True)}
    """)
