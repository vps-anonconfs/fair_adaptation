import argparse
import json
import copy
import pickle

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
sys.path.append('.')

import FairCertModule as FCModule
from datasets import get_dataset
from configs import dataset_defaults
from datasets import tabular_dataset_utils
import models
import data_utils

"""
------------------------------------
tests for fairness metrics
"""

def test_fairness_metrics2():
    dset_name = 'german'
    config = dataset_defaults[dset_name]
    dataset_kwargs = config['dataset_kwargs']
    dataset_kwargs['use_sens'] = True
    dataset = get_dataset(dset_name, **dataset_kwargs)
    x_arr = np.array([x.numpy() for (x, y) in dataset])
    sensr_bounds = FCModule.get_fairness_intervals(x_arr, dataset.sensitive_feature_indices,
                                                   metric='SENSR', use_sens=True, eps=0.1)
    lp_bounds = FCModule.get_fairness_intervals(x_arr, dataset.sensitive_feature_indices,
                                                metric='LP', use_sens=True, eps=0.1)
    # m x m
    _corr = np.corrcoef(np.transpose(x_arr))
    stacked_corr = np.stack([_corr[:, sfi] for sfi in dataset.sensitive_feature_indices], axis=-1)
    cumm_corr = stacked_corr.sum(axis=-1)
    closest_by_corr = np.argsort(-cumm_corr)

    closest_by_sensr = np.argsort(-sensr_bounds)
    closest_by_lp = np.argsort(-lp_bounds)
    print(f"""
    Indices closest to sensitive features by
    sensitive feature indices: {dataset.sensitive_feature_indices}
    correlation: {closest_by_corr[:20]}
    sensr      : {closest_by_sensr[:20]}
    lp         : {closest_by_lp[:20]}
    They are expected to be close!
    all bounds : {sensr_bounds}
    bounds     : {[sensr_bounds[sfi] for sfi in dataset.sensitive_feature_indices]}
    """)


# initially copied from:
# https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/datasets/_samples_generator.py#L1386
def make_spd_matrix(n_dim, cnum, random_state=None):
    generator = np.random.default_rng(random_state)

    A = generator.uniform(size=(n_dim, n_dim))
    U, _, Vt = np.linalg.svd(np.dot(A.T, A))
    evals = generator.uniform(size=n_dim)
    evals = np.sort(evals)
    evals /= evals.max()
    evals[0] *= cnum
    X = np.dot(np.dot(U, np.diag(evals)), Vt)

    return X


def test_fairness_metrics():
    M = np.array([[1.5, 0.5], [0.5, 1.5]])
    interval = FCModule.get_bounds_from_mahalanobis(M)
    print("Check:", interval)
    # tightness will get worse with increasing number of dimension or due to poor condition number of M
    num_tests = 10
    tightness = {}
    dim_vals, cnum_vals = np.arange(2, 30, 5), [2**_i for _i in range(6)]
    xx, xy = np.meshgrid(cnum_vals, dim_vals)
    xx, xy = xx.flatten(), xy.flatten()
    tightness = []
    for xi in range(len(xx)):
        cnum, dim = xx[xi], xy[xi]
        ts = []
        for _rnd in range(10):
            M = make_spd_matrix(n_dim=dim, cnum=cnum, random_state=_rnd)
            interval = FCModule.get_bounds_from_mahalanobis(M)
            print(interval)
            # check tightness of the approximation through MC sampling
            num_samples = 1000
            inside = 0
            for _ in range(num_samples):
                sampled_x = np.random.uniform(low=-interval, high=interval)
                mh_dist = np.matmul(np.matmul(sampled_x.T, M), sampled_x)
                inside += int(mh_dist < 1)
            _t = inside/num_samples
            ts.append(_t)
            # print(f"matrix: {M}")
        _t1, _t2 = np.mean(ts), np.std(ts)
        print(f"Dim {dim: 02d} Cond number {cnum: 02d} Tightness: {np.mean(ts): 0.3f}")
        tightness.append(_t1)
    xx, xy = list(map(int, xx)), list(map(int, xy))
    # print(list(xx.astype(np.int32)), list(xy), tightness)
    with open('checkpoints/mh_tightness.json', 'w') as f:
        json.dump({'cond_num': xx, 'dim': xy, 'tightness': tightness}, f)
    print(xx, xy, tightness)


def test_shifts():
    input_arr = np.random.multivariate_normal(mean=[0, 0], cov=[[1.5, 0.5], [0.5, 1.5]], size=[1000])
    print(input_arr.shape)
    shifted_arr = tabular_dataset_utils.shift_by_projecting_out(input_arr=input_arr, factor=0.5, random_state=21)
    print('Make sure the second plot (in red) has lower variance and zero variance in one of the directions of '
          'non-zero variance of the original (in blue)')
    fix, axis = plt.subplots(1, 2, sharex=True, sharey=True)
    axis[0].plot(input_arr[:, 0], input_arr[:, 1], 'b*')
    axis[1].plot(shifted_arr[:, 0], shifted_arr[:, 1], 'r*')
    plt.show()


def test_mh_rank():
    dat_configs = ['german_expt101', 'adult_expt101', 'credit_expt101']
    for dat_config in dat_configs:
        with open(f'checkpoints/{dat_config}/hparams.json') as f:
            args = argparse.Namespace(**json.load(f))
        sp_dataset_kwargs = copy.deepcopy(args.dataset_kwargs)
        sp_dataset_kwargs['use_sens'] = True
        dataset_with_sensitive_features = get_dataset(args.dataset, **sp_dataset_kwargs)
        use_sens, sens_inds = args.dataset_kwargs['use_sens'], dataset_with_sensitive_features.sensitive_feature_indices
        x_arr = np.array([x.numpy() for (x, y) in dataset_with_sensitive_features.get_subset('train')])
        M = FCModule.compute_SenSR_matrix_source(x_arr, sens_inds, keep_protected_idxs=use_sens)
        l, U = np.linalg.eigh(np.linalg.pinv(M))
        l /= np.max(l)
        print("Numerical rank", (l > 1e-2).sum())


def test_pretrained_models():
    base_std_configs = ['german_expt0', 'german_expt102', 'adult_expt0', 'adult_expt102']
    adapt_configs = ['german_expt114', 'german_expt114', 'adult_expt111', 'adult_expt111']
    results = []
    for adapt_config, base_config in zip(adapt_configs, base_std_configs):
        with open(f'checkpoints/{adapt_config}/hparams.json') as f:
            args = argparse.Namespace(**json.load(f))
        dataset_kwargs = copy.deepcopy(args.dataset_kwargs)
        dataset_kwargs['use_sens'] = args.use_sens
        dataset = get_dataset(args.dataset, **dataset_kwargs)

        train_dataset = dataset.get_subset('train')
        dm = data_utils.CustomDataModule(train_dataset, dataset.get_subset('val'), dataset.get_subset('test'))

        trainer_kwargs = copy.deepcopy(args.trainer_kwargs)
        model_kwargs = copy.deepcopy(args.model_kwargs)
        model_kwargs['num_cls'] = dataset.n_classes
        dim = len(train_dataset[0][0])

        def load_model(model_ckpt):
            base_model = models.get_network(args.model, **model_kwargs)
            pretrained_model = models.get_trainer(base_model, args.trainer, **trainer_kwargs)
            if model_ckpt:
                assert os.path.exists(model_ckpt)
                # because we load from pl module model ckpt
                pretrained_model.load_state_dict(torch.load(model_ckpt)['state_dict'])
            return pretrained_model

        rng = np.random.default_rng()
        loss_diffs = []
        n_samples = 100
        vec_and_diffs = []
        for _ in range(n_samples):
            pseudo_fair_vec = rng.lognormal(0, 0.25, [dim])
            model_ckpt = f'checkpoints/{base_config}/last.ckpt'
            print(f'Loading model from {model_ckpt}')
            model = load_model(model_ckpt)
            model.set_fair_interval(pseudo_fair_vec)

            pl_trainer = pl.Trainer(max_steps=5, accelerator="cpu", devices=1)

            model.ALPHA = 0
            train_result = pl_trainer.validate(model, dataloaders=[dm.train_dataloader()])
            prev_loss = train_result[0]["val_loss"]

            model.ALPHA = 1
            pl_trainer.fit(model, datamodule=dm)

            model.ALPHA = 0
            train_result = pl_trainer.validate(model, dataloaders=[dm.train_dataloader()])
            new_loss = train_result[0]["val_loss"]
            loss_diff = new_loss - prev_loss
            loss_diffs.append(loss_diff)

            vec_and_diffs.append((pseudo_fair_vec, loss_diff))
        print(loss_diffs, np.mean(loss_diffs))
        results.append({'base_config': base_config, 'adapt_config': adapt_config, 'vec_and_diffs': vec_and_diffs})
    with open("checkpoints/sharpness_results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    # test_fairness_metrics()
    # test_shifts()
    # test_mh_rank()
    test_pretrained_models()

