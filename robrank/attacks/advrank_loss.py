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
import sys
import re
import functools
import torch as th
import collections
from tqdm import tqdm
import pylab as lab
import traceback
import math
import statistics
from scipy import stats
import numpy as np
import random
import torch.nn.functional as F
import pytest
import itertools as it


class AdvRankLoss(object):
    '''
    Factory of all types of loss functions used in ranking attacks
    '''

    def RankLossEmbShift(self, repv: th.Tensor, repv_orig: th.Tensor):
        '''
        Computes the embedding shift, we want to maximize it by gradient descent
        '''
        if self.metric == 'C':
            distance = 1 - F.cosine_similarity(repv, repv_orig)
            # distance = -(1 - th.mm(repv, repv_orig)).trace() # not efficient
        elif self.metric in ('E', 'N'):
            distance = F.pairwise_distance(repv, repv_orig)
        loss = -distance.sum()
        return (loss, None)

    def RankLossQueryAttack(self, qs: th.Tensor, Cs: th.Tensor, Xs: th.Tensor,
                            *, pm: str, dist: th.Tensor = None, cidx: th.Tensor = None):
        '''
        Computes the loss function for pure query attack
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])
        NIter, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        #refrank = []
        for i in range(NIter):
            # == compute the pairwise loss
            q = qs[i].view(1, D)  # [1, output_1]
            C = Cs[i, :, :].view(M, D)  # [1, output_1]
            if self.metric == 'C':
                A = (1 - th.mm(q, C.t())).view(1, M)
                B = (1 - th.mm(Xs, q.t())).view(NX, 1)
            elif self.metric in ('E', 'N'):
                A = th.cdist(q, C).view(1, M)
                B = th.cdist(Xs, q).view(NX, 1)
                # [XXX] the old method suffer from large memory footprint
                # A = (C - q).norm(2, dim=1).view(1, M)
                # B = (Xs - q).norm(2, dim=1).view(NX, 1)
            # == loss function
            if '+' == pm:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank
            if DO_RANK:
                ranks.append(th.mean(dist[i].flatten().argsort().argsort()
                                     [cidx[i, :].flatten()].float()).item())
            #refrank.append( ((A>B).float().mean()).item() )
        #print('(debug)', 'rank=', statistics.mean(refrank))
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return (loss, rank)

    def RankLossQueryAttackDistance(self, qs: th.Tensor, Cs: th.Tensor, Xs: th.Tensor, *,
                                    pm: str, dist: th.Tensor = None, cidx: th.Tensor = None):
        '''
        Actually the distance based objective is inferior
        '''
        raise NotImplementedError
        metric = self.metric
        N, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])  # D
        losses, ranks = [], []
        for i in range(N):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if (metric, pm) == ('C', '+'):
                loss = (1 - th.mm(q, C.t())).mean()
            elif (metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(q, C.t())).mean()
            elif (metric, pm) == ('E', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (metric, pm) == ('E', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            losses.append(loss)
            if metric == 'C':
                A = (1 - th.mm(q, C.t())).expand(NX, M)
                B = (1 - th.mm(Xs, q.t())).expand(NX, M)
            elif metric == 'E':
                A = (C - q).norm(2, dim=1).expand(NX, M)
                B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
            # non-normalized result
            rank = ((A > B).float().mean() * NX).item()
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossCandidateAttack(
            self, cs: th.Tensor, Qs: th.Tensor, Xs: th.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack
        '''
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            # == compute pairwise distance
            c = cs[i].view(1, D)  # [1, output_1]
            Q = Qs[i, :, :].view(W, D)  # [W, output_1]
            if self.metric == 'C':
                A = 1 - th.mm(c, Q.t()).expand(NX, W)  # [candi_0, W]
                B = 1 - th.mm(Xs, Q.t())  # [candi_0, W]
            elif self.metric in ('E', 'N'):
                A = (Q - c).norm(2, dim=1).expand(NX, W)  # [candi_0, W]
                B = th.cdist(Xs, Q, p=2.0)
                # B2 = (Xs.view(NX, 1, D).expand(NX, W, D) -
                #     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)  # [candi_0, W]
                #assert((B-B2).abs().norm() < 1e-4)
            # == loss function
            if '+' == pm:
                loss = (A - B).clamp(min=0.).mean()
            elif '-' == pm:
                loss = (-A + B).clamp(min=0.).mean()
            losses.append(loss)
            # == compute the rank. Note, the > sign is correct
            rank = ((A > B).float().mean() * NX).item()
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossCandidateAttackDistance(
            self, cs: th.Tensor, Qs: th.Tensor, Xs: th.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack
        using the inferior distance objective
        '''
        raise NotImplementedError
        metric = self.metric
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            c = cs[i].view(1, D)
            Q = Qs[i, :, :].view(W, D)
            if (metric, pm) == ('C', '+'):
                loss = (1 - th.mm(c, Q.t())).mean()
            elif (metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(c, Q.t())).mean()
            elif (metric, pm) == ('E', '+'):
                loss = (Q - c).norm(2, dim=1).mean()
            elif (metric, pm) == ('E', '-'):
                loss = -(Q - c).norm(2, dim=1).mean()
            losses.append(loss)
            if metric == 'C':
                A = (1 - th.mm(c, Q.t())).expand(NX, W)
                B = (1 - th.mm(Xs, Q.t()))
            elif metric == 'E':
                A = (Q - c).norm(2, dim=1).expand(NX, W)
                B = (Xs.view(NX, 1, D).expand(NX, W, D) -
                     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)
            rank = (
                (A > B).float().mean() *
                NX).item()  # the > sign is correct
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossFullOrderM2Attack(
            self, qs: th.Tensor, ps: th.Tensor, ns: th.Tensor):
        '''
        Computes the loss function for M=2 full-order attack
        '''
        assert(qs.shape[0] == ps.shape[0] == ns.shape[0])
        assert(qs.shape[1] == ps.shape[1] == ns.shape[1])
        if self.metric == 'C':
            dist1 = 1 - th.nn.functional.cosine_similarity(qs, ps, dim=1)
            dist2 = 1 - th.nn.functional.cosine_similarity(qs, ns, dim=1)
        elif self.metric in ('E', 'N'):
            dist1 = th.nn.functional.pairwise_distance(qs, ps, p=2)
            dist2 = th.nn.functional.pairwise_distance(qs, ns, p=2)
        else:
            raise ValueError(self.metric)
        loss = (dist1 - dist2).clamp(min=0.).mean()
        acc = (dist1 <= dist2).sum().item() / qs.shape[0]
        return (loss, acc)

    def RankLossFullOrderMXAttack(self, qs: th.Tensor, Cs: th.Tensor):
        assert(qs.shape[1] == Cs.shape[2])
        NIter, M, D = qs.shape[0], Cs.shape[1], Cs.shape[2]
        losses, taus = [], []
        for i in range(NIter):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if self.metric == 'C':
                dist = 1 - th.mm(q, C.t())
            elif self.metric in ('E', 'N'):
                dist = (C - q).norm(2, dim=1)
            tau = stats.kendalltau(
                np.arange(M), dist.cpu().detach().numpy())[0]
            taus.append(tau)
            dist = dist.expand(M, M)
            loss = (dist.t() - dist).triu(diagonal=1).clamp(min=0.).mean()
            losses.append(loss)
        loss = th.stack(losses).mean()
        tau = statistics.mean(taus)
        return (loss, tau)

    def __init__(self, request: str, metric: str):
        '''
        Initialize various loss functions
        '''
        assert(metric in ('E', 'N', 'C'))
        self.metric = metric
        self.funcmap = {
            'ES': self.RankLossEmbShift,
            'QA': self.RankLossQueryAttack,
            'QA+': functools.partial(self.RankLossQueryAttack, pm='+'),
            'QA-': functools.partial(self.RankLossQueryAttack, pm='-'),
            'QA-DIST': self.RankLossQueryAttackDistance,
            'CA': self.RankLossCandidateAttack,
            'CA+': functools.partial(self.RankLossCandidateAttack, pm='+'),
            'CA-': functools.partial(self.RankLossCandidateAttack, pm='-'),
            'CA-DIST': self.RankLossCandidateAttackDistance,
            'FOA2': self.RankLossFullOrderM2Attack,
            'FOAX': self.RankLossFullOrderMXAttack,
        }
        if request not in self.funcmap.keys():
            raise KeyError(f'Requested loss function "{request}" not found!')
        self.request = request

    def __call__(self, *args, **kwargs):
        '''
        Note, you should handle the normalization outside of this class
        '''
        return self.funcmap[self.request](*args, **kwargs)


@pytest.mark.parametrize('metric', ['C', 'E', 'N'])
def test_arl_embshift(metric: str):
    N, D = 10, 8
    reps1 = th.rand(N, D, requires_grad=True)
    reps2 = th.rand(N, D, requires_grad=True)
    loss, _ = AdvRankLoss('ES', metric)(reps1, reps2)
    loss.backward()


@pytest.mark.parametrize('pm, metric', it.product(('+', '-'), ('C', 'E', 'N')))
def test_arl_qa(pm: str, metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    Cs = th.rand(10, 20, 8)
    Xs = th.rand(30, 8)
    loss, _ = AdvRankLoss(f'QA{pm}', metric)(qs, Cs, Xs)
    loss.backward()


@pytest.mark.parametrize('pm, metric', it.product(('+', '-'), ('C', 'E', 'N')))
def test_arl_ca(pm: str, metric: str):
    cs = th.rand(10, 8, requires_grad=True)
    Qs = th.rand(10, 20, 8)
    Xs = th.rand(30, 8)
    loss, _ = AdvRankLoss(f'CA{pm}', metric)(cs, Qs, Xs)
    loss.backward()


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_arl_foa2(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    ps = th.rand(10, 8)
    ns = th.rand(10, 8)
    loss, _ = AdvRankLoss('FOA2', metric)(qs, ps, ns)
    loss.backward()


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_arl_foax(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    Cs = th.rand(10, 20, 8)
    loss, _ = AdvRankLoss('FOAX', metric)(qs, Cs)
    loss.backward()
