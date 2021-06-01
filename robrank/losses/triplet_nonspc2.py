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
