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
import os
import numpy as np
import functools as ft
from .miner import miner
from .. import configs
import torch.nn.functional as F
import itertools as it
import pytest


def fn_pmargin_kernel(repA: th.Tensor, repP: th.Tensor, repN: th.Tensor,
                      *, metric: str, margin: float, beta: th.Tensor):
    '''
    <functional> the core computation for spc-2 margin loss.
    see ICML20 "revisiting deep metric learing ..."
    '''
    if metric in ('E', 'N'):
        dap = F.pairwise_distance(repA, repP)
        dan = F.pairwise_distance(repA, repN)
    elif metric in ('C',):
        dap = 1 - F.cosine_similarity(repA, repP)
        dan = 1 - F.cosine_similarity(repA, repN)
    else:
        raise ValueError
    lap = (dap - beta + margin).relu()
    lan = (beta - dan + margin).relu()
    lap = th.masked_select(lap, lap > 0.).mean()
    lap = th.tensor(0.).to(repA.device) if th.isnan(lap) else lap
    lan = th.masked_select(lan, lan > 0.).mean()
    lan = th.tensor(0.).to(repA.device) if th.isnan(lan) else lan
    loss = lap + lan
    return loss


def fn_pmargin(repres: th.Tensor, labels: th.Tensor, *,
               beta: float = configs.margin.beta,
               margin: float = configs.margin.margin,
               metric: str, minermethod: str = 'spc2-random'):
    '''
    Margin loss, functional version.
    '''
    # normalize representations on demand
    if metric in ('C', 'N'):
        repres = F.normalize(repres, dim=-1)
    # select triplets
    ancs, poss, negs = miner(repres, labels, method=minermethod, metric=metric)
    # loss
    loss = fn_pmargin_kernel(repres[ancs, :], repres[poss, :], repres[negs, :],
                             metric=metric, margin=margin, beta=beta)
    return loss


class pmarginC(th.nn.Module):
    _metric = 'C'
    _margin: float = configs.margin.margin
    _minermethod = 'spc2-random'

    def __init__(self):
        super(pmarginC, self).__init__()
        self.beta = th.nn.Parameter(th.tensor(configs.margin.beta))

    def raw(self, repA, repP, repN):
        '''
        raw mode used by robrank/defenses/pnp
        '''
        loss = fn_pmargin_kernel(repA, repP, repN, metric=self._metric,
                                 margin=self._margin, beta=self.beta)
        return loss

    def forward(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if int(os.getenv('DEBUG', -1)) > 0:
            print('* margin: current beta = ', self.beta.data)
        return ft.partial(fn_pmargin, metric=self._metric,
                          minermethod=self._minermethod,
                          beta=self.beta, margin=self._margin)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return 'SPC-2'

    def getOptim(self):
        optim = th.optim.SGD(self.parameters(), lr=configs.margin.lr_beta)
        return optim


class pmarginE(pmarginC):
    _metric = 'E'


class pmarginN(pmarginC):
    _metric = 'N'


class pdmarginN(pmarginN):
    _minermethod = 'spc2-distance'


@pytest.mark.parametrize('metric, minermethod', it.product(('C', 'E', 'N'),
                                                           ('spc2-random', 'spc2-distance')))
def test_fn_pmargin(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pmargin(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()


@pytest.mark.parametrize('func', (pmarginC, pmarginE, pmarginN, pdmarginN))
def test_pmargin(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()


@pytest.mark.parametrize('func', (pmarginC, pmarginE, pmarginN, pdmarginN))
def test_pmargin_raw(func: object):
    rA = th.rand(10, 32, requires_grad=True)
    rP = th.rand(10, 32, requires_grad=True)
    rN = th.rand(10, 32, requires_grad=True)
    if func._metric in ('C', 'N'):
        _N = ft.partial(F.normalize, dim=-1)
        rA, rP, rN = _N(rA), _N(rP), _N(rN)
    loss = func().raw(rA, rP, rN)
    loss.backward()
