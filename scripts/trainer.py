#!/usr/bin/env python
# coding: utf-8

"""
Load the model from ckpt, (small) shifted data and fine-tune
"""
import copy
import sys
import os

# Deep Learning Imports
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import argparse
import json
from argparse import Namespace

sys.path.append('.')
import FairCertModule as FCModule
import data_utils
import utils
import configs
import datasets
import models


def train_or_evaluate(args, evaluate_only):
    
    out_dir = f"{args.out_dir}/{args.name}"
    # Dumping first to avoid weird error: quick fix but come back and look at saving error
    if not evaluate_only:
        # record about the specifics of the file name
        hparams = vars(args)
        print(hparams)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(f"{out_dir}/hparams.json", "w") as f:
            json.dump(hparams, f)

    dataset_kwargs = copy.deepcopy(args.dataset_kwargs)
    dataset_kwargs['use_sens'] = args.use_sens
    dataset = datasets.get_dataset(args.dataset, **dataset_kwargs)

    _ln = len(dataset.get_subset('train'))
    if int(_ln*args.data_frac) < 50:
        raise AssertionError(f"Number of data points is too small (fewer than 50), set data fraction to at least: "
                             f"{50/_ln: 0.4f}!!")
    print(f"Training data size: {int(args.data_frac*_ln)}")

    sp_dataset_kwargs = copy.deepcopy(args.dataset_kwargs)
    sp_dataset_kwargs['use_sens'] = True
    dataset_with_sensitive_features = datasets.get_dataset(args.dataset, **sp_dataset_kwargs)
    # load full dataset for metric estimation, this is to remove confounding with training data starvation in model
    # training
    sp_train_dataset = dataset_with_sensitive_features.get_subset('train')
    x_arr = np.array([x.numpy() for (x, y) in sp_train_dataset])
    fair_vec = FCModule.get_fairness_intervals(x_arr, dataset_with_sensitive_features.sensitive_feature_indices,
                                               metric=args.metric, use_sens=args.use_sens)

    """
    =========================================================
    Prepare data for pytorch lightning model
    =========================================================
    """
    train_dataset = dataset.get_subset('train', frac=args.data_frac)
    dm = data_utils.CustomDataModule(train_dataset, dataset.get_subset('val'), dataset.get_subset('test'))

    class_wts = np.ones(dataset.n_classes)
    trainer_kwargs = copy.deepcopy(args.trainer_kwargs)
    if args.use_weighted_ce:
        counts = np.unique(train_dataset.y_array, return_counts=True)[1]
        class_wts = ((counts.sum() - counts) / counts.sum()) * (dataset.n_classes / (dataset.n_classes - 1))
    trainer_kwargs['class_weights'] = torch.Tensor(class_wts)
    model_kwargs = copy.deepcopy(args.model_kwargs)
    model_kwargs['num_cls'] = dataset.n_classes
    assert "max_epochs" not in trainer_kwargs, "Set max_steps instead!"
    print("Length:", len(dm.train_dataloader()))
    num_epochs = int(args.max_steps/len(dm.train_dataloader()))
    if num_epochs > 300:
        num_epochs = 300
        args.max_steps = num_epochs*len(dm.train_dataloader())
    trainer_kwargs["max_epochs"] = num_epochs

    def load_model(model_ckpt):
        base_model = models.get_network(args.model, **model_kwargs)
        pretrained_model = models.get_trainer(base_model, args.trainer, **trainer_kwargs)
        if model_ckpt:
            assert os.path.exists(model_ckpt)
            # because we load from pl module model ckpt
            pretrained_model.load_state_dict(torch.load(model_ckpt)['state_dict'])
        return pretrained_model

    if evaluate_only:
        print(f"Loading model from {out_dir}")
        model_ckpt = utils.get_best_ckpt(out_dir)
    else:
        print(f"Loading model from {args.model_ckpt}")
        model_ckpt = args.model_ckpt
    model = load_model(model_ckpt)

    # todo (unlabeled and dataless): class_wts are not used, i.e. args.use_weighted_ce is not operational
    # todo: test and valid splits should not be modified
    # todo: these are best implemented as new dataset types rather than new trainer.
    if args.trainer == 'unlabeled':
        X_train = np.asarray([x.numpy() for x in dataset.x])
        new_labels = model(torch.Tensor(X_train))
        dataset = datasets.get_dataset(args.dataset, override_ys=new_labels, **dataset_kwargs)
        dm = data_utils.CustomDataModule(dataset.get_subset('train'), dataset.get_subset('val'), dataset.get_subset('test'))
            
    elif args.trainer == 'dataless':
        print("We are in the dataless regime")
        X_noise = copy.deepcopy(np.asarray([x.numpy() for x in dataset.x]))
        mins = np.min(X_noise, axis=0)
        maxs = np.max(X_noise, axis=0)
        X_noise = np.random.uniform(0, 1, size=X_noise.shape)
        X_noise = ((mins - maxs)*X_noise) + maxs
        new_labels = model(torch.Tensor(X_noise))
        print("New labels: ", new_labels.shape)
        dataset = datasets.get_dataset(args.dataset, override_ys=new_labels, override_xs=X_noise, **dataset_kwargs)
        print("done with dataless dataset!")
        dm = data_utils.CustomDataModule(dataset.get_subset('train'), dataset.get_subset('val'), dataset.get_subset('test'))
        class_wts = np.ones(dataset.n_classes)

    if args.trainer in ['unlabeled', 'dataless']:
        if args.use_weighted_ce:
            counts = np.unique(train_dataset.y_array, return_counts=True)[1]
            class_wts = ((counts.sum() - counts) / counts.sum()) * (dataset.n_classes / (dataset.n_classes - 1))
        trainer_kwargs['class_weights'] = torch.Tensor(class_wts)

        model = load_model(model_ckpt)

    """
    =========================================================
    Set up re-train parameters for the model
    =========================================================
    """
    model.set_fair_interval(fair_vec)

    # Fix the number of layers you want to retrain
    ind = 0
    params = [t for t in model.parameters()]
    # tune all params otherwise
    if args.num_adapt > 0:
        for t in model.parameters():
            if ind < (len(params) - 2 * args.num_adapt):
                print(t.shape, " - FROZEN")
                t.requires_grad = False
                t.trainable = False
            else:
                print(t.shape, "- LEARNABLE")
            ind += 1

    """
    =========================================================
    Run the training loop for only a few epochs
    =========================================================
    """
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_quality",
        mode="max",
        dirpath=f"{out_dir}",
        filename=args.dataset + "-{epoch:02d}-{val_quality:.2f}",
        save_last=True,
    )
    # stop training if no improvement beyond tolerance
    earlystopping_callback = EarlyStopping(
        monitor='val_quality',
        mode="max",
        min_delta=0,
        patience=10000,
    )
    #[print("Max steps: ", args.max_steps) for i in range(1000)]
    # args.max_steps = 15000
    pl_trainer = pl.Trainer(max_steps=args.max_steps, accelerator="cpu", devices=1,
                            callbacks=[checkpoint_callback])

    if not evaluate_only:
        pl_trainer.fit(model, datamodule=dm)
        # Error here
        
        val_result = pl_trainer.test(ckpt_path=os.path.join(out_dir, 'last.ckpt'), dataloaders=[dm.val_dataloader()])
        test_result = pl_trainer.test(ckpt_path=os.path.join(out_dir, 'last.ckpt'), dataloaders=[dm.test_dataloader()])
    else:
        val_result = pl_trainer.test(model, dataloaders=[dm.val_dataloader()])
        test_result = pl_trainer.test(model, dataloaders=[dm.test_dataloader()])

    """
    =========================================================
    Write results and hparams 
    =========================================================
    """

    if not evaluate_only:
        # record about the specifics of the file name
        hparams = vars(args)
        print(hparams)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(f"{out_dir}/hparams.json", "w") as f:
            json.dump(hparams, f)

        with open(f"{out_dir}/test_results.json", "w") as f:
            json.dump(test_result, f)

    return val_result, test_result

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument("--model-ckpt", default=None, type=str, help="Full path to pretrained model checkpoint")
    # action=argparse.BooleanOptionalAction for python<3.9
    parser.add_argument("--use-sens", action='store_true', default=False, help="Use sensitive features if set to true")
    parser.add_argument("--use-weighted-ce", action='store_true', default=False,
                        help="When set, balances the classes in the CE loss")

    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--model", type=str, help="Name of the model")
    parser.add_argument("--trainer", type=str, help="Name of the trainer")
    parser.add_argument("--metric", type=str, help="Distance metric to use for individual fairness")
    parser.add_argument('--name', required=True, type=str, help='name this run')
    parser.add_argument('--evaluate-only', action='store_true', default=False,
                        help="If set, only evaluates while loading the model from the checkpoint")

    # args for loading dataset, model and trainer
    parser.add_argument("--dataset_kwargs", help="Dataset loading args")
    parser.add_argument("--model_kwargs", help="Model loading args")
    parser.add_argument("--trainer_kwargs", help="Trainer loading args")

    # control parameters
    parser.add_argument("--data-frac", default=1, type=float,
                        help="Proportion of loaded data to use, should be a value in [0, 1]")
    parser.add_argument("--num-adapt", default=-1, type=int,
                        help="Number of layers to retrain, set to -1 if to tune all parameters.")
    parser.add_argument('--max-steps', type=int, help='Number of training steps')

    parser.add_argument("--out-dir", type=str,
                        help="Output folder where checkpoint, test results will be stored")

    # redundant args
    # parser.add_argument("--metric", default="LP")
    # parser.add_argument("--eps", default=0)
    # parser.add_argument("--dataset", required=True, type=str, help="")

    cli_args = parser.parse_args()
    config_args = configs.populate_defaults(Namespace(**{'name': cli_args.name}))
    args = configs.merge(cli_args, config_args)
    print(vars(args))

    out_dir = f"{args.out_dir}/{args.name}"
    if not args.evaluate_only:
        # clearing out the folder to avoid multiple rewrites
        if os.path.exists(out_dir):
            os.system(f'rm {out_dir}/*')
    utils.set_seed(args.seed)

    train_or_evaluate(args, args.evaluate_only)


if __name__ == '__main__':
    run()