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


def fn__pglift(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    Generalized lifted-structure loss function
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        margin = configs.glift.margin_cosine
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    elif metric in ('E',):
        margin = configs.glift.margin_euclidean
    # Sampling
    anc, pos, neg = miner(repres, labels, method='spc2-lifted', metric=metric)
    # Calculate Loss
    losses = []
    for (i, idx) in enumerate(anc):
        repA = repres[idx, :].view(-1)
        repP = repres[pos[i], :]
        repN = repres[neg[i], :]
        #
        if metric in ('E', 'N'):
            __pdist = ft.partial(th.nn.functional.pairwise_distance, p=2)
        else:
            def __pdist(p, n): return 1 - \
                th.nn.functional.cosine_similarity(p, n, dim=-1)
        pos_term = th.logsumexp(__pdist(repA, repP), dim=-1)
        neg_term = th.logsumexp(margin - __pdist(repA, repN), dim=-1)
        losses.append((pos_term + neg_term).relu())
    loss = th.mean(th.stack(losses)) + configs.glift.l2_weight * \
        th.mean(repres.norm(p=2, dim=-1))
    return loss


class pglift(th.nn.Module):
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn__pglift, metric=self._metric)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pgliftC(pglift):
    _metric = 'C'


class pgliftE(pglift):
    _metric = 'E'


class pgliftN(pglift):
    _metric = 'N'


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_fn_glift(metric):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn__pglift(output, labels, metric=metric)
    loss.backward()


@pytest.mark.parametrize('func', (pgliftC, pgliftE, pgliftN))
def test_glift(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
