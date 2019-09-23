# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.losses import class_loss_calculation
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL, ImageFolderWithIndex, entropy_weights, create_cardinal_Weight
from mean_teacher.utils import *
from mean_teacher.regularization import ANN_annoy, GraphLaplacian, SSL_ADMM, SSL_ADMM_dummy, ANN_W, SSL_Icen

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')

    train_loader, eval_loader, pretrain_loader,ssl_loader = create_data_loaders(**dataset_config, args=args)
    if args.entropy_weight:
        target_truth = np.copy(train_loader.dataset.targets)
    else:
        target_truth = np.argmax(train_loader.dataset.targets,axis=1)
    idx_unlabelled = train_loader.batch_sampler.primary_indices
    for idx in idx_unlabelled:
        train_loader.dataset.targets[idx] = NO_LABEL



    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

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
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    if args.dual_train:
        for epoch in range(3):
            start_time = time.time()
            # train for one epoch
            train(train_loader, model, ema_model, optimizer, epoch, training_log)
            # train(pretrain_loader, model, ema_model, optimizer, epoch, training_log)
            LOG.info("--- pretraining epoch in %s seconds ---" % (time.time() - start_time))

        nx = len(train_loader.dataset.targets)
        nc = train_loader.dataset.nclasses
        lambd = np.zeros((nc,nx))

    for epoch in range(args.start_epoch+3, args.epochs):
        #SSL
        idx_labelled = train_loader.batch_sampler.secondary_indices
        target_labelled = [train_loader.dataset.targets[idx] for idx in idx_labelled]
        U, V, cp = SSL(ssl_loader, model, optimizer, epoch, training_log, idx_labelled,target_labelled,lambd,args.laplace_mode,target_truth)

        # Test accuracy of SSL
        C = np.argmax(cp, axis=1)
        hits_SSL = np.zeros(nc)
        acc_SSL = np.zeros(nc)
        acc_SSL_tot = np.sum(C == target_truth) / len(C) * 100
        for i in range(nc):
            hits_SSL[i] = np.sum(C == i) / len(C) * 100
            acc_SSL[i] = np.sum(C[C == i] == target_truth[C == i]) / np.sum(target_truth==i)*100

        print("SSL hits (%): \n {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f} \n".format(hits_SSL[0],hits_SSL[1],hits_SSL[2],hits_SSL[3],hits_SSL[4],hits_SSL[5],hits_SSL[6],hits_SSL[7],hits_SSL[8],hits_SSL[9]))
        print(
            "SSL acc (%): \n {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f} \n".format(
                acc_SSL[0], acc_SSL[1], acc_SSL[2], acc_SSL[3], acc_SSL[4], acc_SSL[5], acc_SSL[6], acc_SSL[7],
                acc_SSL[8], acc_SSL[9]))
        print("SSL Acc tot = {:3.2f} \n".format(acc_SSL_tot))

        if args.entropy_weight:
            if not args.mutable_known_labels:
                C[idx_labelled] = target_labelled
            train_loader.dataset.targets[:] = C[:]
        else:
            if not args.mutable_known_labels:
                cp[idx_labelled,:] = target_labelled
            train_loader.dataset.targets[:,:] = cp[:,:]


        #Update Lambda
        lambd += U - V


        start_time = time.time()
        #Supervised training for one epoch
        lambd = torch.from_numpy(lambd.T).float().cuda()

        train_ADMM(train_loader, model, ema_model, optimizer, epoch, training_log,lambd,idx_labelled)
        lambd = lambd.t().cpu().numpy()
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))


        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1)
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
            }, is_best, checkpoint_path, epoch + 1)


def parse_dict_args(**kwargs):
    global args

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
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = ImageFolderWithIndex(traindir, transform=train_transformation, label_probability=(not args.entropy_weight))
    # target_transform = torch.eye(self.nclasses)
    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)
        # for idx in unlabeled_idxs:
        #     dataset.targets[idx] = NO_LABEL
    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)

    if args.dual_train:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
        pretrain_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    else:
        pretrain_loader = None
    if args.ssl_train:
        ssl_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(traindir, eval_transformation),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 * args.workers,  # Needs images twice as fast
            pin_memory=True,
            drop_last=False)
    else:
        ssl_loader = None


    return train_loader, eval_loader, pretrain_loader,ssl_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(train_loader, model, ema_model, optimizer, epoch, log):
    global global_step

    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((input, ema_input), target, idxs) in enumerate(train_loader):
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ema_model_out = ema_model(ema_input_var)
        model_out = model(input_var)

        if isinstance(model_out, Variable):
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
        else:
            assert len(model_out) == 3
            assert len(ema_model_out) == 3
            logit1, logit2,_ = model_out
            ema_logit, _,_ = ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.item())
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0
        if args.entropy_weight:
            target_max = target_var
        else:
            target_max = torch.argmax(target_var,dim=1)

        class_loss = class_criterion(class_logit, target_max) / labeled_minibatch_size
        meters.update('class_loss', class_loss.item())

        ema_class_loss = class_criterion(ema_logit, target_max) / labeled_minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        loss = class_loss + consistency_loss + res_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_max.data, topk=(1, 5))
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)
        meters.update('top5', prec5, labeled_minibatch_size)
        meters.update('error5', 100. - prec5, labeled_minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_max.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1, labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1, labeled_minibatch_size)
        meters.update('ema_top5', ema_prec5, labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5, labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })

def train_ADMM(train_loader, model, ema_model, optimizer, epoch, log,lambd,tmp):
    global global_step
    labels = train_loader.dataset.targets[:]
    if args.entropy_weight:
        cardinal_weights = create_cardinal_Weight(labels)
        class_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=NO_LABEL,weight=torch.from_numpy(cardinal_weights)).cuda() #Note the none!, we need to manually sum it up
    else:
        cardinal_weights = create_cardinal_Weight(np.argmax(labels,axis=1))
        class_criterion = nn.MSELoss(reduction='none').cuda()
    CW = torch.from_numpy(cardinal_weights).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss_no_reduction #losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss
    labeled_minibatch_size = args.labeled_batch_size

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    for i, ((input, ema_input), target, idxs) in enumerate(train_loader):
        idxs = idxs.long()
        lambd_select = lambd[idxs, :]
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input)
        with torch.no_grad():
            ema_input_var = torch.autograd.Variable(ema_input)
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

        minibatch_size = len(target_var)
        unlabeled_minibatch_size = minibatch_size - labeled_minibatch_size
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        ema_model_out = ema_model(ema_input_var)
        model_out = model(input_var)

        if isinstance(model_out, Variable):
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
        else:
            assert len(model_out) == 3
            assert len(ema_model_out) == 3
            logit1, logit2, _ = model_out
            ema_logit, _, _ = ema_model_out

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
        else:
            class_logit, cons_logit = logit1, logit1
        softmax1, softmax2 = F.softmax(logit1, dim=1), F.softmax(ema_logit, dim=1)
        iu = list(range(0, unlabeled_minibatch_size))
        ik = list(range(unlabeled_minibatch_size, minibatch_size))

        class_loss = class_loss_calculation(class_logit+lambd_select, target_var, iu, ik, class_criterion, CW,softx=softmax1)
        ema_class_loss = class_loss_calculation(ema_logit+lambd_select, target_var, iu, ik, class_criterion, CW, softx=softmax2)

        meters.update('class_loss', class_loss.item())
        meters.update('ema_class_loss', ema_class_loss.item())
        if args.entropy_weight:
            target_max = target_var
        else:
            target_max = torch.argmax(target_var,dim=1)

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * torch.sum(CW[target_max]*torch.sum(consistency_criterion(cons_logit, ema_logit),dim=1)) / minibatch_size
            # consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)

        if args.logit_distance_cost >= 0:
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit, weight=CW[target_max]) / minibatch_size
            meters.update('res_loss', res_loss.item())
        else:
            res_loss = 0


        loss = class_loss + consistency_loss + res_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_max.data, topk=(1, 5))
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)
        meters.update('top5', prec5, labeled_minibatch_size)
        meters.update('error5', 100. - prec5, labeled_minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_max.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1, labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1, labeled_minibatch_size)
        meters.update('ema_top5', ema_prec5, labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5, labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })


def SSL(SSL_loader, model, log, global_step, epoch, idx,C,lambd,laplace_mode,target_truth):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    descriptorcpu = []
    input_list = []
    outputcpu = []
    with torch.no_grad():
        for i, (input, target ) in enumerate(SSL_loader):
            meters.update('data_time', time.time() - end)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

            # compute output
            output,_,descriptor = model(input_var)
            descriptorcpu.append(descriptor.cpu().numpy())
            input_list.append(input_var.cpu().numpy())
            outputcpu.append(output.cpu().numpy())
        descriptor_flat = np.asarray([item for sublist in descriptorcpu for item in sublist])
        input_array = np.asarray([item for sublist in input_list for item in sublist])
        U = np.asarray([item for sublist in outputcpu for item in sublist]).transpose()

        #Test accuracy of U
        Cc = np.argmax(U.T, axis=1)
        nc, nx = U.shape
        hits_pretrain = np.zeros(nc)
        acc_pretrain = np.zeros(nc)
        acc_SSL_tot = np.sum(Cc == target_truth) / len(Cc) * 100
        for i in range(nc):
            hits_pretrain[i] = np.sum(Cc == i) / len(Cc) * 100
            acc_pretrain[i] = np.sum(Cc[Cc == i] == target_truth[Cc == i]) / np.sum(target_truth == i)*100

        print("SSL hits (%): \n {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f} \n".format(hits_pretrain[0],hits_pretrain[1],hits_pretrain[2],hits_pretrain[3],hits_pretrain[4],hits_pretrain[5],hits_pretrain[6],hits_pretrain[7],hits_pretrain[8],hits_pretrain[9]))
        print(
            "SSL acc (%): \n {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f}   {:3.2f} \n".format(
                acc_pretrain[0], acc_pretrain[1], acc_pretrain[2], acc_pretrain[3], acc_pretrain[4], acc_pretrain[5], acc_pretrain[6], acc_pretrain[7],
                acc_pretrain[8], acc_pretrain[9]))
        print("SSL Acc tot = {:3.2f} \n".format(acc_SSL_tot))


        V = np.copy(U)
        if laplace_mode == 0:
            graph_feats =input_array.reshape(input_array.shape[0],-1)  #Probably needs to be flattened
        elif laplace_mode == 1:
            graph_feats = descriptor_flat
        elif laplace_mode == 2:
            alpha = 1-min(0.01*global_step,1)
            graph_feats = alpha*input_var + (1-alpha)*descriptor_flat
        A,d = ANN_annoy(graph_feats)
        # L = GraphLaplacian(graph_feats, A, d)
        #TODO Decide whether this is only the labelled or the full dataset (used for misfit calc in SSL)
        beta = 1e-3
        rho = 1e-3
        maxIter = 20
        nc = len(np.unique(C))
        nk = len(C)
        Cpk = np.zeros((nc,nk))
        alpha = 100
        # U, cp = SSL_ADMM(U, idx, Cpk, L, alpha, beta, rho, lambd, V, maxIter)
        Y = np.zeros((nx,nc))
        for (i,val) in zip(idx,C):
            Y[i,val] = 1
        alpha = 0.99
        L = ANN_W(graph_feats, A, alpha)
        cp = SSL_Icen(L,Y)

    return U, V, cp.T



def validate(eval_loader, model, log, global_step, epoch):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_loader):
            meters.update('data_time', time.time() - end)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda(non_blocking=True))

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().item()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            output1, _,_ = model(input_var)
            # softmax1, softmax2 = F.softmax(output1, dim=1), F.softmax(output2, dim=1)
            class_loss = class_criterion(output1, target_var) / minibatch_size

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1, labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1, labeled_minibatch_size)
            meters.update('top5', prec5, labeled_minibatch_size)
            meters.update('error5', 100.0 - prec5, labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                LOG.info(
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        i, len(eval_loader), meters=meters))

    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
        .format(top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })

    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size.float()).item())
    return res


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))