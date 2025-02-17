# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.`

import re
import argparse
import logging

from . import architectures, datasets


__all__ = ['parse_cmd_args', 'parse_dict_args']


def create_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                        choices=datasets.__all__,
                        help='dataset: ' +
                            ' | '.join(datasets.__all__) +
                            ' (default: imagenet)')
    parser.add_argument('--train-subdir', type=str, default='train',
                        help='the subdirectory inside the data directory that contains the training data')
    parser.add_argument('--eval-subdir', type=str, default='val',
                        help='the subdirectory inside the data directory that contains the evaluation data')
    parser.add_argument('--labels', default=None, type=str, metavar='FILE',
                        help='list of image labels (default: based on directory structure)')
    parser.add_argument('--exclude-unlabeled', default=False, type=str2bool, metavar='BOOL',
                        help='exclude unlabeled examples from the training set')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        choices=architectures.__all__,
                        help='model architecture: ' +
                            ' | '.join(architectures.__all__))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--labeled-batch-size', default=None, type=int,
                        metavar='N', help="labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='use nesterov momentum', metavar='BOOL')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=None, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-type', default="mse", type=str, metavar='TYPE',
                        choices=['mse', 'kl'],
                        help='consistency loss type to use')
    parser.add_argument('--consistency-rampup', default=30, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--logit-distance-cost', default=-1, type=float, metavar='WEIGHT',
                        help='let the student model have two outputs and use an MSE loss between the logits with the given weight (default: only have one output)')
    parser.add_argument('--checkpoint-epochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--evaluation-epochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pre-train-epochs', default=0, type=int,
                        metavar='PTE', help='pre training epochs (default 0)')
    parser.add_argument('--ssl-train-iter', default=0, type=int,
                        metavar='SSL', help='SSL training iterations (default 0)')
    parser.add_argument('--class-imbalance', default=1, type=int,
                        metavar='c_weight', help='enable class imbalance weight (default 1)')
    parser.add_argument('--entropy-weight', default=True, type=str2bool,
                        metavar='e_weight', help='enable entropy weight and one shot labels (default True)')
    parser.add_argument('--reg-input', default=1, type=int,
                        metavar='RI', help='Regularization input type, 0 = nn input, 1 = nn output, 2 = hybrid (default 1)')
    parser.add_argument('--laplace-mode', default=2, type=int,
                        metavar='LM', help='Determines how the laplacian is created, 0 = euclidian metric, 1 = dot product, 2 = angular (default 2)')
    parser.add_argument('--SSL-ADMM', default=0, type=int,
                        metavar='ADMM', help='Determines how we solve the SSL linear system, 0 = ICELs closed solution, 1 = ADMM gradient descent')
    parser.add_argument('--mutable-known-labels', default=False, type=str2bool,
                        metavar='MKL', help='Enables the known labels to change (default False)')
    parser.add_argument('--ANN-method', default=2, type=int,
                        metavar='ANN', help='ANN method, 0 = exact, 1 = Annoy, 2 = HNSW (default 2)')
    parser.add_argument('--save-pretrain', default=False, type=str2bool,
                        metavar='SP', help='Save pretraining')
    parser.add_argument('--load-pretrain', default='', type=str, metavar='PATH',
                        help='path to pretrained network (default: none)')
    parser.add_argument('--deterministic', default=False, type=str2bool, metavar='Deter',
                        help='Determines whether the network is reproductible, will run slower (default: False)')
    parser.add_argument('--data-seed', default=0, type=int,
                        metavar='DS', help='Sets the data_seed, if deterministic is True')
    parser.add_argument('--use-autoencoder', default=False, type=str2bool,
                        metavar='eae', help='Use an autoencoder for determining the distance function, or possible to transform away from the input images')
    parser.add_argument('--ae-arch', default='', type=str,
                        metavar='ARCH', help='autoencoder architechture')
    parser.add_argument('--load-autoencoder', default='', type=str, metavar='PATH',
                        help='path to pretrained autoencoder')
    parser.add_argument('--save-autoencoder', default=False, type=str2bool, metavar='sae',
                        help='Save the autoencoder after training')
    # parser.add_argument('--laplace-metric', default=1, type=int,
    #                     metavar='RI', help='Regularization input type, 0 = nn input, 1 = nn output, 2 = hybrid (default 1)')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()


def parse_dict_args(LOG,**kwargs):
    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))

    LOG.info("Using these command line args:")
    for key, value in kwargs.items():
        LOG.info("{:30s} : {}".format(key,value))
    return create_parser().parse_args(cmdline_args)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2epochs(v):
    try:
        if len(v) == 0:
            epochs = []
        else:
            epochs = [int(string) for string in v.split(",")]
    except:
        raise argparse.ArgumentTypeError(
            'Expected comma-separated list of integers, got "{}"'.format(v))
    if not all(0 < epoch1 < epoch2 for epoch1, epoch2 in zip(epochs[:-1], epochs[1:])):
        raise argparse.ArgumentTypeError(
            'Expected the epochs to be listed in increasing order')
    return epochs
