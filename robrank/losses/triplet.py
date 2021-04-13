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
from torch.autograd import Variable
from .miner import miner
import functools as ft
import itertools as it
import pytest
import torch.nn.functional as F
import rich
c = rich.get_console()


def __ctriplet(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    Variant of triplet loss that accepts classification batch.
    Metrics: C = cosine, E = Euclidean, N = Normlization + Euclidean
    '''
    anchor, positive, negative = tuple(
        zip(*miner(repres, labels, method='random-triplet')))
    if metric == 'C':
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
        __cos = ft.partial(th.nn.functional.cosine_similarity, dim=-1)
        dap = 1 - __cos(repres[anchor, :], repres[positive, :])
        dan = 1 - __cos(repres[anchor, :], repres[negative, :])
        loss = (dap - dan + configs.triplet.margin_cosine).clamp(min=0.).mean()
    elif metric in ('E', 'N'):
        margin = configs.triplet.margin_euclidean
        if metric == 'N':
            repres = th.nn.functional.normalize(repres, p=2, dim=-1)
            margin = configs.triplet.margin_cosine
        __triplet = ft.partial(th.nn.functional.triplet_margin_loss,
                               p=2, margin=margin)
        loss = __triplet(repres[anchor, :],
                         repres[positive, :], repres[negative, :])
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return loss


ctripletC = ft.partial(__ctriplet, metric='C')


def ctripletC_determine_metric(): return 'C'


def ctripletC_datasetspec(): return 'classification'


ctripletE = ft.partial(__ctriplet, metric='E')


def ctripletE_determine_metric(): return 'E'


def ctripletE_datasetspec(): return 'classification'


ctripletN = ft.partial(__ctriplet, metric='N')


def ctripletN_determine_metric(): return 'N'


def ctripletN_datasetspec(): return 'classification'


def __ttriplet(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    Variant of triplet loss that accetps [a,p,n; a,p,n] batch.
    metrics: C = cosine, E = euclidean, N = normalization + euclidean
    '''
    if metric == 'C':
        __cos = ft.partial(th.nn.functional.cosine_similarity, dim=-1)
        pdistAP = 1 - __cos(repres[0::3], repres[1::3])
        pdistAN = 1 - __cos(repres[0::3], repres[2::3])
        # compute loss: triplet margin loss, cosine version
        loss = (pdistAP - pdistAN +
                configs.triplet.margin_cosine).clamp(min=0.).mean()
    elif metric in ('E', 'N'):
        margin = configs.triplet.margin_euclidean
        if metric == 'N':
            repres = th.nn.functional.normalize(repres, dim=-1)
            margin = configs.triplet.margin_cosine
        __triplet = ft.partial(th.nn.functional.triplet_margin_loss,
                               p=2, margin=margin)
        loss = __triplet(repres[0::3], repres[1::3], repres[2::3])
    else:
        raise ValueError('illegal metric!')
    return loss


ttripletC = ft.partial(__ttriplet, metric='C')


def ttripletC_determine_metric(): return 'C'


def ttripletC_datasetspec(): return 'triplet'


ttripletE = ft.partial(__ttriplet, metric='E')


def ttripletE_determine_metric(): return 'E'


def ttripletE_datasetspec(): return 'triplet'


ttripletN = ft.partial(__ttriplet, metric='N')


def ttripletN_determine_metric(): return 'N'


def ttripletN_datasetspec(): return 'triplet'


def fn__ptriplet(repres: th.Tensor, labels: th.Tensor,
                 *, metric: str, minermethod: str, p_switch: float = -1.0):
    '''
    Variant of triplet loss that accetps [cls=1,cls=1,cls=2,cls=2] batch.
    This corresponds to the SPC-2 setting in the ICML20 paper.

    metrics: C = cosine, E = euclidean, N = normalization + euclidean
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
    # Calculate Loss
    if metric == 'C':
        __cos = ft.partial(F.cosine_similarity, dim=-1)
        dap = 1 - __cos(repres[anc, :], repres[pos, :])
        dan = 1 - __cos(repres[anc, :], repres[neg, :])
        loss = (dap - dan + margin).clamp(min=0.).mean()
    elif metric in ('E', 'N'):
        #__euc = ft.partial(F.pairwise_distance, p=2)
        #dap = __euc(repres[anc, :], repres[pos, :])
        #dan = __euc(repres[anc, :], repres[neg, :])
        #loss = (dap - dan + margin).relu().mean()
        #c.log('dap', dap.mean().item(), 'dan', dan.mean().item())
        __triplet = ft.partial(F.triplet_margin_loss, p=2, margin=margin)
        loss = __triplet(repres[anc, :], repres[pos, :], repres[neg, :])
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return loss


class ptriplet(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        if hasattr(self, '_minermethod'):
            return ft.partial(fn__ptriplet, metric=self._metric,
                              minermethod=self._minermethod)(*args, **kwargs)
        else:
            return ft.partial(fn__ptriplet, metric=self._metric)(
                *args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class ptripletC(ptriplet):
    _metric = 'C'


class ptripletE(ptriplet):
    _metric = 'E'


class ptripletN(ptriplet):
    _metric = 'N'


class pmtriplet(ptriplet):
    _minermethod = 'spc2-semihard'


class pmtripletC(pmtriplet):
    _metric = 'C'


class pmtripletE(pmtriplet):
    _metric = 'E'


class pmtripletN(pmtriplet):
    _metric = 'N'


class phtriplet(ptriplet):
    _minermethod = 'spc2-hard'


class phtripletC(phtriplet):
    _metric = 'C'


class phtripletE(phtriplet):
    _metric = 'E'


class phtripletN(phtriplet):
    _metric = 'N'


class pstriplet(ptriplet):
    _minermethod = 'spc2-softhard'


class pstripletC(pstriplet):
    _metric = 'C'


class pstripletE(pstriplet):
    _metric = 'E'


class pstripletN(pstriplet):
    _metric = 'N'


class pdtriplet(ptriplet):
    _minermethod = 'spc2-distance'


class pdtripletC(pdtriplet):
    _metric = 'C'


class pdtripletE(pdtriplet):
    _metric = 'E'


class pdtripletN(pdtriplet):
    _metric = 'N'


class pDtripletN(pdtripletN):
    def __call__(self, *args, **kwargs):
        return ft.partial(fn__ptriplet, metric=self._metric,
                          minermethod=self._minermethod,
                          p_switch=0.15)(*args, **kwargs)


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_triplet(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn__ptriplet(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('func', (ptripletC, ptripletN, ptripletE,
                                  pmtripletN, phtripletN, pstripletN, pdtripletN, pDtripletN))
def test_triplet(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()


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
