import time

import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from mean_teacher import losses, architectures
from mean_teacher.data import NO_LABEL, create_cardinal_Weight
from mean_teacher.losses import class_loss_calculation
from mean_teacher.regularization import ANN_hnsw, SSL_Icen, ANN_annoy, Laplacian_Euclidian, \
    Laplacian_ICEL, Laplacian_angular, SSL_ADMM
from mean_teacher.utils import AverageMeterSet, accuracy, update_ema_variables, adjust_learning_rate, \
    get_current_consistency_weight, accuracy_SSL
from torch.autograd import Variable
from scipy.special import softmax

def train(train_loader, model, ema_model, optimizer, epoch, log,args,global_step,LOG):

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

        adjust_learning_rate(optimizer, epoch, i, len(train_loader),args)
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

        class_loss = class_criterion(class_logit, target_max) / minibatch_size
        # class_loss = class_criterion(class_logit, target_max) / labeled_minibatch_size
        meters.update('class_loss', class_loss.item())

        ema_class_loss = class_criterion(ema_logit, target_max) / minibatch_size
        # ema_class_loss = class_criterion(ema_logit, target_max) / labeled_minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch,args)
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
    return global_step


def train_ae(train_loader, model, optimizer, epoch, LOG):

    # criterion = nn.BCELoss()
    criterion = nn.MSELoss(reduction='mean').cuda()
    model.train()
    running_loss = 0.0
    for i, (input, _) in enumerate(train_loader):
        input_var = torch.autograd.Variable(input.cuda(non_blocking=True))

        # ============ Forward ============
        encoded, outputs = model(input_var)
        loss = criterion(outputs, input_var)
        # ============ Backward ============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data

        # ============ Logging ============
    LOG.info("Epoch {}, loss: {:.3f}".format(epoch, running_loss / (i+1)))

def train_ADMM(train_loader, model, ema_model, optimizer, epoch, log,lambd,tmp,args,global_step,LOG):
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

        adjust_learning_rate(optimizer, epoch, i, len(train_loader),args)
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

        class_loss = class_loss_calculation(class_logit, target_var, iu, ik, class_criterion, CW,softx=softmax1)
        ema_class_loss = class_loss_calculation(ema_logit, target_var, iu, ik, class_criterion, CW, softx=softmax2)
        # class_loss = class_loss_calculation(class_logit+lambd_select, target_var, iu, ik, class_criterion, CW,softx=softmax1)
        # ema_class_loss = class_loss_calculation(ema_logit+lambd_select, target_var, iu, ik, class_criterion, CW, softx=softmax2)

        meters.update('class_loss', class_loss.item())
        meters.update('ema_class_loss', ema_class_loss.item())
        if args.entropy_weight:
            target_max = target_var
        else:
            target_max = torch.argmax(target_var,dim=1)

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch,args)
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
        if (np.isnan(loss.item()) or loss.item() > 1e5):
           print("class_loss= {}".format(class_loss))
           print("consistency_loss= {}".format(consistency_loss))
           print("res_loss= {}".format(res_loss))

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
        tt =1


def SSL(SSL_loader, model, log, global_step, epoch, idx,C,lambd,target_truth,args):
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
        cp = softmax(U.T, axis=1)
        V = np.copy(U)
        Cc = np.argmax(U.T, axis=1)
        nc, nx = U.shape

        accuracy_SSL(Cc,target_truth,nc,log)


        if args.ANN_method == -1: #Debug mode, try everything:
            graph_feats = descriptor_flat
            # t0 = time.time()
            # A1, d1 = ANN_annoy(graph_feats)
            # t1 = time.time()
            # print("ANN annoy {}".format(t1-t0))
            A2, d2 = ANN_hnsw(graph_feats)
            # t2 = time.time()
            # print("ANN hnsw {}".format(t2-t1))
            # Le1 = Laplacian_Euclidian(graph_feats, A1, d1)
            # Le2 = Laplacian_Euclidian(graph_feats, A2, d2)
            # t3 = time.time()
            # print("Lap Euclidian {}".format(t3-t2))
            alpha = 0.99
            # Li1 = Laplacian_ICEL(graph_feats, A1, alpha)
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            # t4 = time.time()
            # print("Lap ICEL {}".format(t4-t3))
            # La1 = Laplacian_angular(graph_feats, A1, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            # t5 = time.time()
            # print("Lap angular {}".format(t5-t4))

            # Y = np.zeros((nx, nc))
            # for (i, val) in zip(idx, C):
            #     Y[i, val] = 1
            # cp_new = SSL_Icen(Le1, Y)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)
            #
            # Y = np.zeros((nx, nc))
            # for (i, val) in zip(idx, C):
            #     Y[i, val] = 1
            # cp_new = SSL_Icen(Le2, Y)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)

            alpha = 0.95
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)


            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.98
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.99
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.995
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.999
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.9995
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)

            alpha = 0.9999
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc)

            alpha = 0.99995
            Li2 = Laplacian_ICEL(graph_feats, A2, alpha)
            La2 = Laplacian_angular(graph_feats, A2, alpha)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(Li2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)
            Y = np.zeros((nx, nc))
            for (i, val) in zip(idx, C):
                Y[i, val] = 1
            cp_new = SSL_Icen(La2, Y)
            Cc = np.argmax(cp_new.T, axis=1)
            accuracy_SSL(Cc, target_truth, nc,log)



            # nc = len(np.unique(C))
            # nk = len(C)
            # Cpk = np.zeros((nc, nk))
            # tt = range(len(idx))
            # ta = target_truth[idx]
            # Cpk[ta, tt] = 1
            # beta = 1e-3
            # rho = 1e-6
            # maxIter = 100
            # alpha = 200

            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, Le1, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)

            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, Le2, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)

            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, Li1, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)
            #
            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, Li2, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)
            #
            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, La1, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)
            #
            # U = np.copy(V)
            # _, cp_new = SSL_ADMM(U, idx, Cpk, La2, alpha, beta, rho, lambd, cp.T, maxIter)
            # Cc = np.argmax(cp_new.T, axis=1)
            # accuracy_SSL(Cc, target_truth, nc)

        else:
            if args.reg_input == 0:
                graph_feats = input_array.reshape(input_array.shape[0], -1)
            elif args.reg_input == 1:
                graph_feats = descriptor_flat
            elif args.reg_input == 2:
                alpha = 1-min(0.01*global_step,1)
                graph_feats = alpha*input_var + (1-alpha)*descriptor_flat

            if args.ANN_method == 0:
                raise("Method not made yet")
            elif args.ANN_method == 1:
                A,d = ANN_annoy(graph_feats)
            elif args.ANN_method == 2:
                A, d = ANN_hnsw(graph_feats)
            else:
                raise("Not valid ANN method selected")

            if args.laplace_mode == 0:
                L = Laplacian_Euclidian(graph_feats, A, d)
            elif args.laplace_mode == 1:
                alpha = 0.99
                L = Laplacian_ICEL(graph_feats, A, alpha)
            elif args.laplace_mode == 2:
                alpha = 0.99
                L = Laplacian_angular(graph_feats, A, alpha)
            else:
                raise("An invalid Laplace_mode was selected")

            if args.SSL_ADMM == 0:
                Y = np.zeros((nx, nc))
                for (i, val) in zip(idx, C):
                    Y[i, val] = 1
                cp_new = SSL_Icen(L, Y)
            elif args.SSL_ADMM == 1:
                beta = 1e-3
                rho = 1e-3
                maxIter = 20
                nc = len(np.unique(C))
                nk = len(C)
                Cpk = np.zeros((nc, nk))
                tt = range(len(idx))
                ta = target_truth[idx]
                Cpk[ta, tt] = 1
                alpha = 100
                U, cp_new = SSL_ADMM(U, idx, Cpk, L, alpha, beta, rho, lambd, V, maxIter)
            else:
                raise("An invalid SSL_ADMM value was selected")


        #Test acc of ANNs
        Cc = np.argmax(cp_new.T, axis=1)
        accuracy_SSL(Cc, target_truth, nc,log)


    return cp_new.T, cp


def create_model(ema=False,LOG=None,args=None,nc=None,ae=False):
    if ae:
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.ae_arch))
    else:
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))
    if ae:
        model_factory = architectures.__dict__[args.ae_arch]
    else:
        model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=nc)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()

    return model