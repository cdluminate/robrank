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
import random
import rich
c = rich.get_console()


def fn_ptriplet_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor,
                       *, metric: str, margin: float):
    '''
    <functional> the core computation for spc-2 triplet loss.
    '''
    if metric == 'C':
        dap = 1 - F.cosine_similarity(repA, repP, dim=-1)
        dan = 1 - F.cosine_similarity(repA, repN, dim=-1)
        loss = (dap - dan + margin).relu().mean()
    elif metric in ('E', 'N'):
        #dap = F.pairwise_distance(repA, repP, p=2)
        #dan = F.pairwise_distance(repA, repN, p=2)
        #loss = (dap - dan + margin).relu().mean()
        loss = F.triplet_margin_loss(repA, repP, repN, margin=margin)
    else:
        raise ValueError(f'Illegal metric type {metric}!')
    return loss


def fn_ptriplet(repres: th.Tensor, labels: th.Tensor,
                *, metric: str, minermethod: str, p_switch: float = -1.0, xa: bool = False):
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
    if xa:
        return fn_ptriplet_kernel(repres[anc, :].detach(), repres[pos, :],
                                  repres[neg, :], metric=metric, margin=margin)
    loss = fn_ptriplet_kernel(repres[anc, :], repres[pos, :], repres[neg, :],
                              metric=metric, margin=margin)
    return loss


class ptriplet(th.nn.Module):
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'
    _xa = False

    def __call__(self, *args, **kwargs):
        if hasattr(self, '_minermethod'):
            return ft.partial(fn_ptriplet,
                              metric=self._metric,
                              minermethod=self._minermethod,
                              xa=self._xa)(*args, **kwargs)
        else:
            return ft.partial(fn_ptriplet,
                              metric=self._metric,
                              xa=self._xa)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec

    def raw(self, repA, repP, repN, *, override_margin: float = None):
        if self._metric in ('C', 'N'):
            margin = configs.triplet.margin_cosine
        elif self._metric in ('E',):
            margin = configs.triplet.margin_euclidean
        if override_margin is not None:
            margin = override_margin
        loss = fn_ptriplet_kernel(repA, repP, repN,
                                  metric=self._metric, margin=margin)
        return loss

    def cosine_only(self, repres, repres_orig, labels):
        '''
        pami figure 2
        '''
        assert self._metric == 'N', 'this function only supports metric N'
        assert self._datasetspec == 'SPC-2', 'this function only supports SPC-2'
        repres = F.normalize(repres, p=2, dim=-1)
        # spc-2 random sampling
        ianc = th.arange(0, len(labels), 2)
        ipos = th.arange(1, len(labels), 2)
        ineg = []
        for i in range(labels.nelement() // 2):
            mask_neg = (labels != labels[2*i])
            if mask_neg.sum() > 0:
                ineg.append(random.choice(th.where(mask_neg)[0]).item())
            else:
                ineg.append(np.random.choice(len(labels)))
        ineg = th.tensor(ineg, dtype=th.long, device=repres.device)
        # calculate grad direction
        def ndiff__(a__, b__):
            tmp__ = a__ - b__
            norm__ = tmp__.norm()
            return tmp__ / norm__

        benign_vq = repres_orig[ianc]
        advers_vq = repres[ianc]
        benign_vp = repres_orig[ipos]
        advers_vp = repres[ipos]
        benign_vn = repres_orig[ineg]
        advers_vn = repres[ineg]

        benign_pl_pvq = ndiff__(benign_vq, benign_vp) - ndiff__(benign_vq, benign_vn)
        advers_pl_pvq = ndiff__(advers_vq, advers_vp) - ndiff__(advers_vq, advers_vn)
        cos1 = th.nn.functional.cosine_similarity(advers_pl_pvq, benign_pl_pvq).view(-1)
        cos1 = cos1.cpu().detach().tolist()

        benign_pl_pvp = ndiff__(benign_vp, benign_vq)
        advers_pl_pvp = ndiff__(advers_vp, advers_vq)
        cos2 = th.nn.functional.cosine_similarity(advers_pl_pvp, benign_pl_pvp).view(-1)
        cos2 = cos2.cpu().detach().tolist()

        benign_pl_pvn = ndiff__(benign_vq, benign_vn)
        advers_pl_pvn = ndiff__(advers_vq, advers_vn)
        cos3 = th.nn.functional.cosine_similarity(advers_pl_pvn, benign_pl_pvn).view(-1)
        cos3 = cos3.cpu().detach().tolist()

        return cos1 + cos2 + cos3


class ptripletC(ptriplet):
    _metric = 'C'


class ptripletE(ptriplet):
    _metric = 'E'


class ptripletN(ptriplet):
    _metric = 'N'


class ptripxaN(ptriplet):
    _metric = 'N'
    _xa = True


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
        return ft.partial(fn_ptriplet, metric=self._metric,
                          minermethod=self._minermethod,
                          p_switch=0.15)(*args, **kwargs)


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance', 'spc2-hard', 'spc2-softhard', 'spc2-semihard')))
def test_fn_ptriplet(metric: str, minermethod: str):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_ptriplet(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('func', (ptripletN, ptripletE, ptripletC,
                                  pmtripletN, pmtripletE, pmtripletC,
                                  pstripletN, pstripletE, pstripletC,
                                  pdtripletN, pdtripletE, pdtripletC,
                                  phtripletN, phtripletE, phtripletC))
def test_ptriplet(func: object):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
