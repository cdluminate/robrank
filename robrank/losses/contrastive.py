'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

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

import torch as th
import torch.nn.functional as F
import numpy as np
from .. import configs
import functools as ft
from .miner import miner
import pytest
import itertools as it


def fn_pcontrast_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor,
                        *, metric: str, margin: float):
    '''
    <functional> the core computation for spc-2 contrastive loss.
    '''
    if metric in ('C',):
        targets = th.ones(repA.size(0)).to(repA.device)
        lap = F.cosine_embedding_loss(repA, repP, targets, margin=margin)
        lan = F.cosine_embedding_loss(repA, repN, -targets, margin=margin)
        loss = lap + lan
    elif metric in ('E', 'N'):
        __pd = ft.partial(th.nn.functional.pairwise_distance, p=2)
        lap = __pd(repA, repP).mean()
        lap = th.tensor(0.).to(repA.device) if th.isnan(lap) else lap
        lan = margin - __pd(repA, repN)
        lan = th.masked_select(lan, lan > 0.).mean()
        lan = th.tensor(0.).to(repA.device) if th.isnan(lan) else lan
        loss = lap + lan
    return loss


def fn_pcontrast(repres: th.Tensor, labels: th.Tensor, *,
                 metric: str, minermethod: str = 'spc2-random', p_switch: float = -1.0):
    '''
    Functional version of contrastive loss function with cosine distance
    as the distance metric. Metric is either 'C' (for cosine) or 'E' for
    euclidean.
    Dataset type should be SPC-2 (according to ICML20 reference)
    '''
    # determine the margin
    if metric in ('C', 'N'):
        margin = configs.contrastive.margin_cosine
    elif metric in ('E', ):
        margin = configs.contrastive.margin_euclidean
    # normalize representation on demand
    if metric in ('C', 'N'):
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    # sampling triplets
    ancs, poss, negs = miner(
        repres, labels, method=minermethod, metric=metric, margin=margin, p_switch=p_switch)
    # loss
    loss = fn_pcontrast_kernel(repres[ancs, :], repres[poss, :],
                               repres[negs, :], metric=metric, margin=margin)
    return loss


class pcontrastC(th.nn.Module):
    _metric = 'C'
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pcontrast, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN):
        if self._metric in ('C', 'N'):
            margin = configs.contrastive.margin_cosine
        elif self._metric in ('E', ):
            margin = configs.contrastive.margin_euclidean
        loss = fn_pcontrast_kernel(repA, repP, repN,
                                   metric=self._metric, margin=margin)
        return loss


class pcontrastE(pcontrastC):
    _metric = 'E'


class pcontrastN(pcontrastC):
    _metric = 'N'


class pdcontrastN(th.nn.Module):
    _metric = 'N'
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pcontrast, metric=self._metric,
                          minermethod='spc2-distance')(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pDcontrastN(th.nn.Module):
    _metric = 'N'
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pcontrast, metric=self._metric,
                          minermethod='spc2-distance', p_switch=0.15)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


@pytest.mark.parametrize('metric, minermethod',
                         it.product(('C', 'E', 'N'), ('spc2-random', 'spc2-distance')))
def test_pcontrast(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pcontrast(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('metric', 'CEN')
def test_pcontrast_raw(metric: str):
    rA, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    rP = th.rand(10, 32, requires_grad=True)
    rN = th.rand(10, 32, requires_grad=True)
    lossfunc = {'C': pcontrastC, 'N': pcontrastN, 'E': pcontrastE}[metric]()
    if metric in ('C', 'N'):
        _N = ft.partial(F.normalize, dim=-1)
        rA, rP, rN = _N(rA), _N(rP), _N(rN)
    loss = lossfunc.raw(rA, rP, rN)
    loss.backward()
