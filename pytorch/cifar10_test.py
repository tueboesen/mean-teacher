# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Train CIFAR-10 with 1000 or 4000 labels and all training images. Evaluate against test set."""

import sys
import logging

import torch

import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext


LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 2,
        'checkpoint_epochs': 20,
        # Pseudo labels
        'ssl_train_iter': 20,
        'class_imbalance': 1,
        'entropy_weight': True,
        'reg_input': 1,
        'mutable_known_labels': False,
        'ANN_method': 2,
        'laplace_mode': 2,
        'SSL_ADMM': 0,
        'deterministic': False,

        'save_autoencoder': True,
        'use_autoencoder': True,
        # 'load_autoencoder':
        # Save/Load
        # 'resume': ".\\results\\cifar10_test\\2019-09-25_16_58_52\\4000_10\\transient\\checkpoint.2.ckpt",
         'save_pretrain': True,
         # 'load_pretrain': ".\\results\\cifar10_test\\2019-09-30_17_03_38\\4000_10\\transient\\pretrain.2.ckpt",
        # 'load_pretrain': ".\\results\\cifar10_test\\2019-10-04_09_14_27\\100_10\\transient\\pretrain.120.ckpt",

        # Data
        'dataset': 'cifar10',
        'train_subdir': 'train+val',
        'eval_subdir': 'test',

        # Data sampling
        'base_batch_size': 128,
        'base_labeled_batch_size': 31,

        # Architecture
        'arch': 'cifar_smalltest',
        'ae_arch': 'auto_encoder',
        # 'arch': 'cifar_shakeshake26',

        # Costs
        'consistency_type': 'mse',
        'consistency_rampup': 5,
        'consistency': 100.0,
        'logit_distance_cost': .01,
        'weight_decay': 2e-4,

        # Optimization
        'lr_rampup': 0,
        'base_lr': 0.05,
        'nesterov': True,
    }

    # # 4000 labels:
    # for data_seed in range(10, 20):
    #     yield {
    #         **defaults,
    #         'title': '4000-label cifar-10',
    #         'n_labels': 4000,
    #         'data_seed': data_seed,
    #         'epochs': 300,
    #         'lr_rampdown_epochs': 350,
    #         'ema_decay': 0.99,
    #     }

    # 100 labels:
    # for data_seed in range(10, 11):
    #     yield {
    #         **defaults,
    #         'title': '100-label cifar-10',
    #         'pre_train_epochs': 60,
    #         'n_labels': 100,
    #         'data_seed': data_seed,
    #         'epochs': 60,
    #         'lr_rampdown_epochs': 80,
    #         'ema_decay': 0.95,
    #         'base_batch_size': 100,
    #         'base_labeled_batch_size': 11,
    #     }
    #
    # for data_seed in range(10, 11):
    #     yield {
    #         **defaults,
    #         'title': '250-label cifar-10',
    #         'pre_train_epochs': 120,
    #         'n_labels': 250,
    #         'data_seed': data_seed,
    #         'epochs': 120,
    #         'lr_rampdown_epochs': 150,
    #         'ema_decay': 0.95,
    #         'base_batch_size': 256,
    #         'base_labeled_batch_size': 11,
    #     }
    #
    # for data_seed in range(10, 11):
    #     yield {
    #         **defaults,
    #         'title': '500-label cifar-10',
    #         'pre_train_epochs': 180,
    #         'n_labels': 500,
    #         'data_seed': data_seed,
    #         'epochs': 180,
    #         'lr_rampdown_epochs': 210,
    #         'ema_decay': 0.97,
    #         'base_batch_size': 100,
    #         'base_labeled_batch_size': 31,
    #     }
    #
    # # 1000 labels:
    # for data_seed in range(10, 11):
    #     yield {
    #         **defaults,
    #         'title': '1000-label cifar-10',
    #         'pre_train_epochs': 180,
    #         'n_labels': 1000,
    #         'data_seed': data_seed,
    #         'epochs': 180,
    #         'lr_rampdown_epochs': 210,
    #         'ema_decay': 0.97,
    #         'base_batch_size': 128,
    #         'base_labeled_batch_size': 31,
    #     }

    # 1000 labels: different batch size
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '1000-label cifar-10',
            'pre_train_epochs': 2,
            'n_labels': 1000,
            'data_seed': data_seed,
            'epochs': 2,
            'lr_rampdown_epochs': 210,
            'ema_decay': 0.97,
            'base_batch_size': 100,
            'base_labeled_batch_size': 31,
        }

    # 2000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '2000-label cifar-10',
            'pre_train_epochs': 2,
            'n_labels': 2000,
            'data_seed': data_seed,
            'epochs': 3,
            'lr_rampdown_epochs': 210,
            'ema_decay': 0.97,
            'base_batch_size': 100,
            'base_labeled_batch_size': 31,
        }
    # 4000 labels:
    for data_seed in range(10, 11):
        yield {
            **defaults,
            'title': '4000-label cifar-10',
            'pre_train_epochs': 180,
            'n_labels': 4000,
            'data_seed': data_seed,
            'epochs': 180,
            'lr_rampdown_epochs': 210,
            'ema_decay': 0.97,
            'base_batch_size': 100,
            'base_labeled_batch_size': 31,
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    LOG.info('run title: %s, data seed: %d', title, data_seed)

    ngpu = torch.cuda.device_count()
    assert ngpu > 0, "Expecting at least one GPU, found none."

    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
        'data_seed': data_seed,
    }
    context = RunContext(__file__, "{}_{}".format(n_labels, data_seed))
    logfile = "{}/{}.log".format(context.result_dir, 'output')
    fh = logging.FileHandler(logfile)
    LOG.addHandler(fh)
    LOG.info('run title: %s, data seed: %d', title, data_seed)
    main.args = parse_dict_args(LOG,**adapted_args,**kwargs)
    main.main(context,LOG)

    LOG.info('Run finished, closing logfile.')
    LOG.removeHandler(fh)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
