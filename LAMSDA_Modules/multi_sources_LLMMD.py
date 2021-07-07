#!/usr/bin/env python
# encoding: utf-8

import torch
from LAMSDA_Modules.multi_sources_LocalWeight import LocalWeight

# the module of Stage 2 (LLMMD)

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

# Stage 2 LLMMD

def LLMMD(source, target, s_label, t_label_pseudo, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = source.size()[0]

    loc_weight_ss, loc_weight_tt, loc_weight_st = LocalWeight.loc_weight(s_label, t_label_pseudo, type='visual')
    loc_ss = torch.from_numpy(loc_weight_ss).cuda()
    loc_weight_tt = torch.from_numpy(loc_weight_tt).cuda()
    loc_weight_st = torch.from_numpy(loc_weight_st).cuda()

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loc_loss = torch.Tensor([0]).cuda()

    if torch.sum(torch.isnan(sum(kernels))):
        return loc_loss

    SSN = kernels[:batch_size, :batch_size]
    TTN = kernels[batch_size:, batch_size:]
    STN = kernels[:batch_size, batch_size:]

    loc_loss += torch.sum(loc_weight_ss * SSN + loc_weight_tt * TTN - 2 * loc_weight_st * STN)

    return loc_loss

# MMD

def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)

    return loss
