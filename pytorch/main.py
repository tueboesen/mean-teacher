# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import argparse
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.losses import class_loss_calculation
from mean_teacher.optimization import train, SSL, train_ADMM, create_model, train_ae
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL, ImageFolderWithIndex, entropy_weights, create_cardinal_Weight, save_checkpoint, \
    create_data_loaders, save_pretraining
from mean_teacher.utils import *

# LOG = logging.getLogger('main')
# create logger with 'spam_application'
# LOG.setLevel(logging.DEBUG)
# create file handler which logs even debug messages



args = None
best_prec1 = 0

import sys


#
# class Logger(object):
#     def __init__(self,directory,name):
#         self.log_file_path = "{}/{}.log".format(directory, name)
#         self.terminal = sys.stdout
#         self.log = open(self.log_file_path, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         #this flush method is needed for python 3 compatibility.
#         #this handles the flush command by doing nothing.
#         #you might want to specify some extra behavior here.
#         pass




def main(context,LOG):
    global best_prec1
    global_step = 0
    pre_train_epoch_start = 0
    skip_pretrain = False

    if args.deterministic:
        fix_seed(args.data_seed)
    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    # sys.stdout = Logger(context.result_dir,"output")

    #Data loaders
    train_loader, eval_loader, pretrain_loader,ssl_loader,ae_loader = create_data_loaders(**dataset_config, args=args)
    if args.entropy_weight:
        target_truth = np.copy(train_loader.dataset.targets)
    else:
        target_truth = np.argmax(train_loader.dataset.targets,axis=1)
    idx_unlabelled = train_loader.batch_sampler.primary_indices
    for idx in idx_unlabelled:
        train_loader.dataset.targets[idx] = NO_LABEL

    #Use Autoencoder?
    if args.use_autoencoder:
        autoencoder = create_model(LOG=LOG,args=args,nc=num_classes,ae=True)
        if args.load_autoencoder: #Load ae?
            autoencoder.load_state_dict(torch.load(args.load_autoencoder))
        else: #Train ae
            optimizer = optim.Adam(autoencoder.parameters())
            for epoch in range(100):
                train_ae(ae_loader, autoencoder, optimizer, epoch, LOG)
            if args.save_autoencoder:
                torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")


    #Model creater
    model = create_model(LOG=LOG,args=args,nc=num_classes)
    ema_model = create_model(ema=True,LOG=LOG,args=args,nc=num_classes)

    LOG.info(parameters_string(model))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        U = checkpoint['U']
        V = checkpoint['V']
        target_truth = checkpoint['target_truth']
        lambd = checkpoint['lambd']
        label = checkpoint['label']
        target_labelled = checkpoint['target_labelled']
        idx_labelled = checkpoint['idx_labelled']
        cp = checkpoint['cp']
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        if args.entropy_weight:
            if not args.mutable_known_labels:
                label[idx_labelled] = target_labelled
            train_loader.dataset.targets[:] = label[:]
        else:
            if not args.mutable_known_labels:
                cp[idx_labelled, :] = target_labelled
            train_loader.dataset.targets[:, :] = cp[:, :]
    elif args.load_pretrain:
        assert os.path.isfile(args.load_pretrain), "=> no pretrain found at '{}'".format(args.load_pretrain)
        LOG.info("=> loading pretrained network '{}'".format(args.load_pretrain))
        pretrained = torch.load(args.load_pretrain)
        args.start_epoch = 0
        global_step = pretrained['global_step']
        model.load_state_dict(pretrained['state_dict'])
        ema_model.load_state_dict(pretrained['ema_state_dict'])
        optimizer.load_state_dict(pretrained['optimizer'])
        lambd = pretrained['lambd']
        pre_train_epoch_start = pretrained['epoch']
        LOG.info("=> loaded pretrained network '{}' (epoch {})".format(args.load_pretrain, pretrained['epoch']))
        if pre_train_epoch_start >= args.pre_train_epochs:
            skip_pretrain = True

    if not args.deterministic:
        cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch,LOG,args)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch,LOG,args)
        return

    nx = len(train_loader.dataset.targets)
    nc = train_loader.dataset.nclasses
    if not skip_pretrain:
        lambd = np.zeros((nc, nx))
        for epoch in range(pre_train_epoch_start,args.pre_train_epochs):
            start_time = time.time()
            # train for one epoch
            global_step = train(train_loader, model, ema_model, optimizer, epoch, training_log,args,global_step,LOG)
            # train(pretrain_loader, model, ema_model, optimizer, epoch, training_log,args,global_step,LOG)
            LOG.info("--- pretraining epoch in %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1,LOG,args)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1,LOG,args)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                if args.save_pretrain:
                    save_pretraining({
                        'epoch': epoch+1,
                        'global_step': global_step,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'ema_state_dict': ema_model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer': optimizer.state_dict(),
                        'lambd': lambd,
                    }, checkpoint_path, epoch + 1)

        if args.save_pretrain:
            save_pretraining({
                'epoch': epoch+1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                'lambd': lambd,
            }, checkpoint_path, epoch + 1)

    args.start_epoch += 60
    pseudo_iterations = np.round(np.linspace(args.start_epoch, args.epochs,args.ssl_train_iter+1)[0:-1])
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in pseudo_iterations:
            # Generate Pseudo labels
            idx_labelled = train_loader.batch_sampler.secondary_indices
            target_labelled = [train_loader.dataset.targets[idx] for idx in idx_labelled]
            cp, cp_old = SSL(ssl_loader, ema_model, optimizer, epoch, LOG, idx_labelled, target_labelled, lambd,
                           target_truth, args)

            label = np.argmax(cp, axis=1)

            if args.entropy_weight:
                if not args.mutable_known_labels:
                    label[idx_labelled] = target_labelled
                train_loader.dataset.targets[:] = label[:]
            else:
                if not args.mutable_known_labels:
                    cp[idx_labelled,:] = target_labelled
                train_loader.dataset.targets[:,:] = cp[:,:]


        #Update Lambda
        lambd += cp.T - cp_old.T
        print("Lambda mean value = {}".format(np.mean(np.abs(lambd))))


        start_time = time.time()
        lambd = torch.from_numpy(lambd.T).float().cuda()
        train_ADMM(train_loader, model, ema_model, optimizer, epoch, training_log, lambd, idx_labelled, args,global_step, LOG)
        lambd = lambd.t().cpu().numpy()
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))


        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1,LOG,args)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1,LOG,args)
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
                # 'U': U,
                # 'V': V,
                'lambd': lambd,
                'target_truth': target_truth,
                'label': label,
                'target_labelled': target_labelled,
                'idx_labelled': idx_labelled,
                'cp': cp,

            }, is_best, checkpoint_path, epoch + 1)









if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))