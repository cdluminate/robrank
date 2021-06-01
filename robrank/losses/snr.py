'''
Copyright (C) 2019-2021, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import os
import torch as th
import numpy as np
from .. import configs
import torch
import torch.nn as nn
from .miner import miner
import functools as ft
import itertools as it
import pytest
import torch.nn.functional as F
import rich
c = rich.get_console()


def fn_psnr_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor, *,
                   margin: float = configs.snr.margin,
                   reg_lambda: float = configs.snr.reg_lambda):
    '''
    Raw functional version of SPC-2 SNR.
    https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch
    '''
    pos_snr = th.var(repA - repP, dim=1) / th.var(repA, dim=1)
    neg_snr = th.var(repA - repN, dim=1) / th.var(repA, dim=1)
    reg_los = th.mean(th.abs(th.sum(repA, dim=1)))
    snr_los = (pos_snr - neg_snr + margin).relu()
    snr_los = th.sum(snr_los) / th.sum(snr_los > 0)
    loss = snr_los + reg_lambda * reg_los
    return loss


def fn_psnr(repres: th.Tensor, labels: th.Tensor,
            *, metric: str, minermethod: str, p_switch: float = -1.0):
    '''
    SNR Loss function for DML
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2, dim=-1)
    # Sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=configs.snr.margin, p_switch=p_switch)
    # Calculate Loss
    loss = fn_psnr_kernel(repres[anc, :], repres[pos, :], repres[neg, :])
    return loss


class psnr(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        if hasattr(self, '_minermethod'):
            return ft.partial(fn_psnr, metric=self._metric,
                              minermethod=self._minermethod)(*args, **kwargs)
        else:
            return ft.partial(fn_psnr, metric=self._metric)(
                *args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN):
        return fn_psnr_kernel(repA, repP, repN)


class psnrC(psnr):
    _metric = 'C'


class psnrE(psnr):
    _metric = 'E'


class psnrN(psnr):
    _metric = 'N'


@pytest.mark.parametrize('metric, minermethod', it.product('NEC',
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_psnr(metric: str, minermethod: str):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_psnr(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('func', [psnrC, psnrE, psnrN])
def test_psnr(func: object):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
