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

import os
import torch as th
import numpy as np
from .. import configs
import torch
import torch.nn as nn
from torch.autograd import Variable
from .miner import miner
import functools as ft
import itertools as it
import pytest
import torch.nn.functional as F
import rich
c = rich.get_console()


def fn_pquad(repres: th.Tensor, labels: th.Tensor, *, metric: str,
             minermethod: str, p_switch: float = -1.0):
    '''
    Quadruplet Loss Function
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = configs.triplet.margin_cosine
        margin2 = configs.quadruplet.margin2_cosine
        repres = F.normalize(repres, p=2, dim=-1)
    elif metric in ('E',):
        margin = configs.triplet.margin_euclidean
        margin2 = configs.quadruplet.margin2_euclidean
    # Sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=margin, p_switch=p_switch)
    mask2 = th.logical_and(neg != neg.view(-1, 1),
                           labels.view(-1)[neg] != labels.view(-1)[neg].view(-1, 1))
    neg2 = [np.random.choice(th.where(mask)[0].cpu()) if any(th.where(mask)[0])
            else np.random.choice(repres.size(0)) for mask in mask2]
    neg2 = th.tensor(neg2).to(repres.device)
    # Calculate Triplet Loss: tloss
    __cos = ft.partial(F.cosine_similarity, dim=-1)
    __euc = ft.partial(F.pairwise_distance, p=2)
    if metric == 'C':
        dap = 1 - __cos(repres[anc, :], repres[pos, :])
        dan = 1 - __cos(repres[anc, :], repres[neg, :])
        tloss = (dap - dan + margin).clamp(min=0.).mean()
    elif metric in ('E', 'N'):
        __triplet = ft.partial(F.triplet_margin_loss, p=2, margin=margin)
        tloss = __triplet(repres[anc, :], repres[pos, :], repres[neg, :])
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    # Calculate Quadruplet Loss: qloss
    if metric in ('E', 'N'):
        dap = __euc(repres[anc, :], repres[pos, :])
        dnn = __euc(repres[neg, :], repres[neg2, :])
        qloss = (dap - dnn + margin2).relu().mean()
    elif metric == 'C':
        dap = 1 - __cos(repres[anc, :], repres[pos, :])
        dnn = 1 - __cos(repres[neg, :], repres[neg2, :])
        qloss = (dap - dnn + margin2).relu().mean()
    # sum and return
    return tloss + qloss


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_pquad(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pquad(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


class pquad(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pquad, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pquadC(pquad):
    _metric = 'C'


class pquadE(pquad):
    _metric = 'E'


class pquadN(pquad):
    _metric = 'N'


class pdquadN(pquad):
    _metric = 'N'
    _minermethod = 'spc2-distance'


@pytest.mark.parametrize('func', (pquadC, pquadE, pquadN, pdquadN))
def test_pquad(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()


def fn_rhomboid(repres: th.Tensor, labels: th.Tensor, *,
                metric: str, minermethod: str, p_switch: float = -1.0):
    '''
    my private rhomboid loss implementation (for SPC-2 batch)
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = configs.triplet.margin_cosine
        repres = F.normalize(repres, p=2, dim=-1)
    elif metric in ('E',):
        margin = configs.triplet.margin_euclidean
    # Sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric, margin=margin, p_switch=p_switch)
    ne2 = (neg - th.sign((neg % 2) - 0.5)).long()
    # Calculate Loss
    if metric == 'C':
        def __dist(x, y): return 1 - F.cosine_similarity(x, y)
    elif metric in ('E', 'N'):
        __dist = F.pairwise_distance
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    # a, p, n
    dap = __dist(repres[anc, :], repres[pos, :])
    dan = __dist(repres[anc, :], repres[neg, :])
    loss = (dap - dan + margin).relu().mean()
    # n, n2, a
    xdap = __dist(repres[neg, :], repres[ne2, :])
    xdan = __dist(repres[neg, :], repres[anc, :])
    xloss = (xdap - xdan + margin).relu().mean()
    return loss + xloss


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_rhomboid(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_rhomboid(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


class prhom(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_rhomboid, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class prhomC(prhom):
    _metric = 'C'


class prhomE(prhom):
    _metric = 'E'


class prhomN(prhom):
    _metric = 'N'


class pdrhomN(prhom):
    _metric = 'N'
    _minermethod = 'spc2-distance'


@pytest.mark.parametrize('func', (prhomC, prhomE, prhomN, pdrhomN))
def test_prhom(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()


def fn_pgil(repres: th.Tensor, labels: th.Tensor,
            *, metric: str, minermethod: str):
    '''
    GIL for Deep Metric Learning
    '''
    # sample the triplets
    anc, pos, neg = miner(repres, labels, method=minermethod,
                          metric=metric)
    # normalize
    if metric in ('C', 'N'):
        repres = F.normalize(repres, p=2)
    # loss function
    rA, rP, rN = repres[anc, :], repres[pos, :], repres[neg, :]
    if metric == 'C':
        margin = configs.triplet.margin_cosine
        dap = 1 - F.cosine_similarity(rA, rP, dim=-1)
        dan = 1 - F.cosine_similarity(rA, rN, dim=-1)
        dpn = 1 - F.cosine_similarity(rP, rN, dim=-1)
    elif metric in ('E', 'N'):
        margin = configs.triplet.margin_euclidean
        dap = F.pairwise_distance(rA, rP, p=2)
        dan = F.pairwise_distance(rA, rN, p=2)
        dpn = F.pairwise_distance(rP, rN, p=2)
    else:
        raise NotImplementedError
    if metric == 'N':
        margin = configs.triplet.margin_cosine
    # [method 1: move anchor]
    #mask_repulse = (dap > dan).view(-1)
    ##loss_repulse = ((repres[anc, :] * (repres[anc, :] - repres[neg, :])).sum(-1) + 1.0) / (dan ** 2)
    # loss_repulse = ((repres[anc, :] * (repres[neg, :] - repres[anc, :])).sum(-1) + 1.0) #/ (dan ** 2)
    #mask_attract = (dap <= dan).view(-1)
    ##loss_attract = ((repres[anc, :] * (repres[pos, :] - repres[anc, :])).sum(-1) + 1.0) / (dap ** 2)
    # loss_attract = ((repres[anc, :] * (repres[anc, :] - repres[pos, :])).sum(-1) + 1.0) #/ (dap ** 2)
    #lrep = th.masked_select(loss_repulse, mask_repulse)
    #latt = th.masked_select(loss_attract, mask_attract)
    #loss = th.cat([lrep, latt]).mean()
    # [method 2: move pos and neg] : working
    #loss_attract = (th.mul(repres[pos, :], repres[pos, :] - repres[anc, :]).sum(-1) + 1.0) * (dap ** 2)
    #loss_repulse = (th.mul(repres[neg, :], repres[anc, :] - repres[neg, :]).sum(-1) + 1.0) / (dan ** 2)
    #loss = th.cat([loss_attract, loss_repulse]).mean()
    # [method 3: ring all]
    # loss = th.cat([
    #    # anchor: attract and repulse
    #    (th.mul(repres[anc, :], repres[pos, :] - repres[anc, :]).sum(-1) + 1.0) * (dap ** 2),
    #    #(th.mul(repres[anc, :], repres[anc, :] - repres[neg, :]).sum(-1) + 1.0) / (dan ** 2),
    #    # positive: attract and repulse
    #    (th.mul(repres[pos, :], repres[anc, :] - repres[pos, :]).sum(-1) + 1.0) * (dap ** 2),
    #    #(th.mul(repres[pos, :], repres[pos, :] - repres[neg, :]).sum(-1) + 1.0) / (dpn ** 2),
    #    # negative: repulse
    #    (th.mul(repres[neg, :], repres[neg, :] - repres[anc, :]).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(repres[neg, :], repres[neg, :] - repres[pos, :]).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean()
    # [method 4: all / no weight]
    # loss = th.cat([
    #    (th.mul(rA, rN - rP).sum(-1) + 1.0),
    #    (th.mul(rP, rN - rA).sum(-1) + 1.0),
    #    (th.mul(rN, rA + rP - rN).sum(-1) + 1.0),
    #    ]).mean()
    # [method 5: all / has weight]
    # loss = th.cat([
    #    (th.mul(rA, rA - rP).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rA, rN - rA).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rP, rP - rA).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rP, rN - rP).sum(-1) + 1.0) / (dpn ** 2),
    #    (th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean()
    # [static weight + triplet mask]
    # loss = th.stack([
    #    (th.mul(rA, rA/2. - rP).sum(-1) + 1.0) * (dap/dan),
    #    (th.mul(rA, rN - rA/2.).sum(-1) + 1.0) / (dap/dan),
    #    (th.mul(rP, rP/2. - rA).sum(-1) + 1.0) * (dap/dan),
    #    (th.mul(rP, rN - rP/2.).sum(-1) + 1.0) / (dap/dpn),
    #    (th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dap/dan),
    #    (th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dap/dpn),
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [static weight + pair mask]
    # loss = th.cat([
    #    th.masked_select((th.mul(rA, rA - rP).sum(-1) + 1.0) * (dap ** 2).detach(), dap > margin),
    #    th.masked_select((th.mul(rA, rN - rA).sum(-1) + 1.0) / (dan ** 2).detach(), dan < margin),
    #    th.masked_select((th.mul(rP, rP - rA).sum(-1) + 1.0) * (dap ** 2).detach(), dap > margin),
    #    th.masked_select((th.mul(rP, rN - rP).sum(-1) + 1.0) / (dpn ** 2).detach(), dpn < margin),
    #    th.masked_select((th.mul(rN, rA - rN/2.).sum(-1) + 1.0) / (dan ** 2).detach(), dan < margin),
    #    th.masked_select((th.mul(rN, rP - rN/2.).sum(-1) + 1.0) / (dpn ** 2).detach(), dpn < margin),
    #    ]).mean()
    # [no weight + triplet mask]
    # loss = th.stack([
    #    th.mul(rA, rA/2. - rP).sum(-1) + 1.0,
    #    th.mul(rA, rN - rA/2.).sum(-1) + 1.0,
    #    th.mul(rP, rP/2. - rA).sum(-1) + 1.0,
    #    th.mul(rP, rN - rP/2.).sum(-1) + 1.0,
    #    th.mul(rN, rA - rN/2.).sum(-1) + 1.0,
    #    th.mul(rN, rP - rN/2.).sum(-1) + 1.0,
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [normalized]
    # loss = th.stack([
    #    (th.mul(rA, F.normalize(rA/2. - rP)).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rA, F.normalize(rN - rA/2.)).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rP, F.normalize(rP/2. - rA)).sum(-1) + 1.0) * (dap ** 2),
    #    (th.mul(rP, F.normalize(rN - rP/2.)).sum(-1) + 1.0) / (dpn ** 2),
    #    (th.mul(rN, F.normalize(rA - rN/2.)).sum(-1) + 1.0) / (dan ** 2),
    #    (th.mul(rN, F.normalize(rP - rN/2.)).sum(-1) + 1.0) / (dpn ** 2),
    #    ]).mean(0)
    #mask = (dap - dan + margin >= 0.).view(-1)
    #loss = th.masked_select(loss, mask).mean()
    # [ simple ]
    # 1. should not use mask
    #l1 = (th.mul(rP, rP/2 - rA).sum(-1) + 1.0) * (dap ** 2)
    #l2 = (th.mul(rN, rA - rN/2).sum(-1) + 1.0) / (dan ** 2)
    # [ simple: direction + norm . pow(1)
    #l1 = (th.mul(rP, rP/2 - rA).sum(-1) + 1.0)
    #l2 = (th.mul(rN, rA - rN/2).sum(-1) + 1.0) / (dan**2).detach()
    # [simple: direction + norm . pow(2)
    l1 = (th.mul(rP, rP / 2 - rA).sum(-1) + 1.0) * dap.detach()
    l2 = (th.mul(rN, rA - rN / 2).sum(-1) + 1.0) / (dan**3).detach()
    loss = th.cat([l1, l2]).mean()
    return loss


class pgil(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        if hasattr(self, '_minermethod'):
            return ft.partial(fn_pgil, metric=self._metric,
                              minermethod=self._minermethod)(*args, **kwargs)
        else:
            return ft.partial(fn_pgil, metric=self._metric)(
                *args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pgilC(pgil):
    _metric = 'C'


class pgilE(pgil):
    _metric = 'E'


class pgilN(pgil):
    _metric = 'N'


@pytest.mark.parametrize('func', (pgilC, pgilE, pgilN))
def test_pgil(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
