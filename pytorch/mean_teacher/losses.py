# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""

import torch
from torch.nn import functional as F
from torch.autograd import Variable

from mean_teacher.data import entropy_weights


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='sum') / num_classes


def softmax_mse_loss_no_reduction(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='none') / num_classes



def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')


def symmetric_mse_loss(input1, input2, weight=None):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    if weight is None:
        result = torch.sum((input1 - input2) ** 2) / num_classes
    else:
        result = torch.sum(weight*torch.sum((input1 - input2) ** 2,dim=1)) / num_classes
    return result

def class_loss_calculation(x,target,iu,ik,lossfnc,CW,softx=None):
    """Calculates the misfit """
    nu = len(iu)
    nk = len(ik)
    nx, nc = x.shape

    if isinstance(lossfnc,torch.nn.modules.loss.CrossEntropyLoss):
        if softx is None:
            softx = F.softmax(x, dim=1)
        entropy_weight = entropy_weights(softx[iu].detach())
        class_loss_unlabelled = torch.sum(entropy_weight * lossfnc(x[iu],target[iu])) / nu
        class_loss_labelled = torch.sum(lossfnc(x[ik], target[ik])) / nk
        class_loss = class_loss_unlabelled + class_loss_labelled
    elif isinstance(lossfnc, torch.nn.modules.loss.MSELoss):
        target_max = torch.argmax(target, dim=1)
        class_loss_unlabelled = torch.sum(CW[target_max[iu]] * torch.sum(lossfnc(x[iu], target[iu]), dim=1)) / (nu*nc)
        class_loss_labelled = torch.sum(CW[target_max[ik]] * torch.sum(lossfnc(x[ik], target[ik]), dim=1)) / (nk*nc)
        class_loss = class_loss_unlabelled + class_loss_labelled
    else:
        raise ValueError("lossfnc not defined")
    return class_loss