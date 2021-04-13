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
import pytest


def fn__pnpair(repres: th.Tensor, labels: th.Tensor, *, metric: str):
    '''
    FIXME: is my npair implementation correct? or is the equation in paper incorrect?
    '''
    # Determine the margin for the specific metric
    if metric in ('C', 'N'):
        repres = th.nn.functional.normalize(repres, p=2, dim=-1)
    # Sampling
    anc, pos, neg = miner(repres, labels, method='spc2-npair', metric=metric)
    # Calculate Loss
    losses = []
    for (i, idx) in enumerate(anc):
        #repA = repres[idx, :].view(-1)
        # XXX: this norm trick helps alot according to my own observation
        repA = th.nn.functional.normalize(repres[idx, :].view(-1), dim=-1)
        repP = repres[pos[i], :]
        repN = repres[neg[i], :]
        inner = th.mv(repN - repP, repA)
        # -- ICML 20 implementation: maybe incorrect?
        losses.append(th.log(1 + th.sum(th.exp(inner))))
        # -- Original paper (N-pair-ovo)
        # losses.append(
        #    th.logaddexp(
        #        th.tensor(0.0).to(
        #            inner.device),
        #        inner).sum())

        # [ICLM20] upstream implementation -- but it does not converge ...
        # anchor = idx
        # positive = pos[i]
        # negative_set = neg[i]
        # batch = repres
        # anchors = anc
        # a_embs, p_embs, n_embs = batch[anchor:anchor +
        #                                1], batch[positive:positive +
        #                                          1], batch[negative_set]
        # inner_sum = a_embs[:, None, :].bmm(
        #     (n_embs - p_embs[:, None, :]).permute(0, 2, 1))
        # inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
        # l = torch.mean(
        #     torch.log(
        #         torch.sum(
        #             torch.exp(inner_sum),
        #             dim=1) + 1)) / len(anchors)
        # l = l + configs.npair.l2_weight * \
        #     torch.mean(torch.norm(batch, p=2, dim=1)) / len(anchors)
        # losses.append(l)
    loss = th.mean(th.stack(losses)) + configs.npair.l2_weight * \
        th.mean(repres.norm(p=2, dim=-1))
    return loss


class pnpair(th.nn.Module):
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn__pnpair, metric=self._metric)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pnpairC(pnpair):
    _metric = 'C'


class pnpairE(pnpair):
    _metric = 'E'


class pnpairN(pnpair):
    _metric = 'N'


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_fn_pnpair(metric):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn__pnpair(output, labels, metric=metric)
    loss.backward()


@pytest.mark.parametrize('func', (pnpairC, pnpairE, pnpairN))
def test_pnpair(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
