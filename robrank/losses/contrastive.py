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

import torch as th
import numpy as np
from .. import configs
import functools as ft
from .miner import miner
import pytest
import itertools as it


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
    if metric == 'C':
        __cosemb = ft.partial(
            th.nn.functional.cosine_embedding_loss, margin=margin)
        targets = th.ones(repres.size(0) // 2).to(repres.device)
        lap = __cosemb(repres[ancs, :], repres[poss, :], targets)
        lan = __cosemb(repres[ancs, :], repres[negs, :], -targets)
        loss = lap + lan
    elif metric in ('E', 'N'):
        __pdist = ft.partial(th.nn.functional.pairwise_distance, p=2)
        lap = __pdist(repres[ancs, :], repres[poss, :]).mean()
        lap = th.tensor(0.).to(repres.device) if th.isnan(lap) else lap
        lan = margin - __pdist(repres[ancs, :], repres[negs, :])
        lan = th.masked_select(lan, lan > 0.).mean()
        lan = th.tensor(0.).to(repres.device) if th.isnan(lan) else lan
        loss = lap + lan
    else:
        raise ValueError('illegal metric')
    return loss


class pcontrastC(th.nn.Module):
    _metric = 'C'
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_pcontrast, metric=self._metric)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


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


def __contrastive(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    Functional version of contrastive loss function with cosine distance
    as the distance metric. Metric is either 'C' (for cosine) or 'E' for
    euclidean.
    '''
    anchor, positive, negative = tuple(zip(*miner(repres, labels)))
    if metric == 'C':
        # repres = th.nn.functional.normalize(repres, p=2, dim=-1)
        __cosine_embedding = ft.partial(th.nn.functional.cosine_embedding_loss,
                                        margin=configs.contrastive.margin_cosine)
        targets = th.ones(len(anchor)).to(repres.device)
        lap = __cosine_embedding(
            repres[anchor, :], repres[positive, :], targets)
        lan = __cosine_embedding(
            repres[anchor, :], repres[negative, :], -targets)
        loss = lap + lan
    elif metric == 'E':
        __pdist = ft.partial(th.nn.functional.pairwise_distance, p=2)
        margin = configs.contrastive.margin_euclidean
        lap = __pdist(repres[anchor, :], repres[positive, :]).mean()
        lan = margin - __pdist(repres[anchor, :], repres[negative, :])
        lan = th.masked_select(lan, lan > 0.).mean()
        loss = lap + lan
    else:
        raise ValueError('Illegal metric type!')
    return loss


contrastiveC = ft.partial(__contrastive, metric='C')
contrastiveE = ft.partial(__contrastive, metric='E')
for metric in ('C', 'E'):
    locals()[f'contrastive{metric}_determine_metric'] = lambda: metric


@pytest.mark.parametrize('metric, minermethod',
                         it.product(('C', 'E', 'N'), ('spc2-random', 'spc2-distance')))
def test_contrastive(metric, minermethod):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_pcontrast(output, labels, metric=metric, minermethod=minermethod)
    loss.backward()
