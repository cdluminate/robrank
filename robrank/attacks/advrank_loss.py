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

# pylint: disable=no-member
import functools
import torch as th
import statistics
from scipy import stats
import numpy as np
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

        Arguments:
            repv: size(batch, embdding_dim), requires_grad.
            repv_orig: size(batch, embedding_dim).
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

        Arguments:
            qs: size(batch, embedding_dim), query embeddings.
            Cs: size(batch, M, embedding_dim), selected candidates.
            Xs: size(testsize, embedding_dim), embedding of test set.
            pm: either '+' or '-'.
            dist: size(batch, testsize), pairwise distance matrix.
            cidx: size(batch, M), index of candidates in Xs.
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
        the distance based objective is worse.
        '''
        assert(qs.shape[1] == Cs.shape[2] == Xs.shape[1])  # D
        N, M, D, NX = qs.shape[0], Cs.shape[1], Cs.shape[2], Xs.shape[0]
        DO_RANK = (dist is not None) and (cidx is not None)
        losses, ranks = [], []
        for i in range(N):
            q = qs[i].view(1, D)
            C = Cs[i, :, :].view(M, D)
            if (self.metric, pm) == ('C', '+'):
                loss = (1 - th.mm(q, C.t())).mean()
            elif (self.metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(q, C.t())).mean()
            elif (self.metric, pm) == ('E', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('E', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '+'):
                loss = (C - q).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '-'):
                loss = -(C - q).norm(2, dim=1).mean()
            losses.append(loss)
            if DO_RANK:
                if self.metric == 'C':
                    A = (1 - th.mm(q, C.t())).expand(NX, M)
                    B = (1 - th.mm(Xs, q.t())).expand(NX, M)
                elif self.metric in ('E', 'N'):
                    A = (C - q).norm(2, dim=1).expand(NX, M)
                    B = (Xs - q).norm(2, dim=1).view(NX, 1).expand(NX, M)
                # non-normalized result
                rank = ((A > B).float().mean() * NX).item()
                ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks) if DO_RANK else None
        return (loss, rank)

    def RankLossCandidateAttack(
            self, cs: th.Tensor, Qs: th.Tensor, Xs: th.Tensor, *, pm: str):
        '''
        Computes the loss function for pure candidate attack

        Arguments:
            cs: size(batch, embedding_dim), embeddings of candidates.
            Qs: size(batch, W, embedding_dim), embedding of selected queries.
            Xs: size(testsize, embedding_dim), embedding of test set.
            pm: either '+' or '-'
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
        assert(cs.shape[1] == Qs.shape[2] == Xs.shape[1])
        NIter, W, D, NX = cs.shape[0], Qs.shape[1], Qs.shape[2], Xs.shape[0]
        losses, ranks = [], []
        for i in range(NIter):
            c = cs[i].view(1, D)
            Q = Qs[i, :, :].view(W, D)
            if (self.metric, pm) == ('C', '+'):
                loss = (1 - th.mm(c, Q.t())).mean()
            elif (self.metric, pm) == ('C', '-'):
                loss = -(1 - th.mm(c, Q.t())).mean()
            elif (self.metric, pm) == ('E', '+'):
                loss = (Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('E', '-'):
                loss = -(Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '+'):
                loss = (Q - c).norm(2, dim=1).mean()
            elif (self.metric, pm) == ('N', '-'):
                loss = -(Q - c).norm(2, dim=1).mean()
            losses.append(loss)
            if self.metric == 'C':
                A = (1 - th.mm(c, Q.t())).expand(NX, W)
                B = (1 - th.mm(Xs, Q.t()))
            elif self.metric in ('E', 'N'):
                A = (Q - c).norm(2, dim=1).expand(NX, W)
                B = (Xs.view(NX, 1, D).expand(NX, W, D) -
                     Q.view(1, W, D).expand(NX, W, D)).norm(2, dim=2)
            rank = ((A > B).float().mean() * NX).item()  # ">" sign is correct
            ranks.append(rank)
        loss = th.stack(losses).mean()
        rank = statistics.mean(ranks)
        return (loss, rank)

    def RankLossFullOrderM2Attack(
            self, qs: th.Tensor, ps: th.Tensor, ns: th.Tensor):
        '''
        Computes the loss function for M=2 full-order attack

        Arguments:
            qs: size(batch, embedding_dim), queries/anchors
            ps: size(batch, embedding_dim), positive samples
            ns: size(batch, embedding_dim), negative samples
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

    def RankLossGreedyTop1Misrank(self, qs: th.Tensor, emm: th.Tensor,
                                  emu: th.Tensor, ems: th.Tensor, Xs: th.Tensor):
        '''
        <Compound loss> Greedy Top-1 Misranking. (GTM)
        Goal: top1.class neq original class.
        Arguments:
            qs: size(batch, embedding_dim), query to be perturbed.
            emm: size(batch, 1, embdding_dim), top-1 matching candidate
            emu: size(batch, 1, embdding_dim), top-1 unmatching cancidate

        Observations[MNIST-rc2f2-ptripletN]:
            1) only loss_match: very weak compared to ES
            2) only loss_unmatch: relatively weak
            3) combined: weak
            4) loss_unmatch (dist)
            5) loss_match (dist)
            6) dist both
            7) dist match
        [*] 8) dist unmatch (still best after qc selection bugfix)
        '''
        assert(qs.shape[1] == emm.shape[2] == emu.shape[2])
        #loss_match, _ = self.funcmap['QA-'](qs, emm, Xs)
        #loss_match, _ = self.funcmap['QA-DIST'](qs, emm, Xs, pm='-')
        #loss_unmatch, _ = self.funcmap['QA+'](qs, emu, Xs)
        #loss_unmatch, _ = self.funcmap['QA-DIST'](qs, emu, Xs, pm='+')
        #loss = loss_match + loss_unmatch
        # [scratch]
        emm = emm.squeeze()
        emu = emu.squeeze()
        ems = ems.squeeze()
        if self.metric in ('C',):
            #l_m = -(1 - F.cosine_similarity(qs, emm))
            l_u = (1 - F.cosine_similarity(qs, emu))
        elif self.metric in ('E', 'N'):
            #l_m = -F.pairwise_distance(qs, emm)
            l_u = F.pairwise_distance(qs, emu)
            #l_s = -F.pairwise_distance(qs, ems)
        loss = (l_u).mean()
        return loss

    def RankLossGreedyTop1Translocation(self, qs: th.Tensor, emm: th.Tensor,
                                        emu: th.Tensor, ems: th.Tensor, Xs: th.Tensor):
        '''
        <Compound loss> Greedy Top-1 Translocation (GTT)
        Goal: top1.identity neq original identity.
        Arguments:
            see document for GTM
        observations:
            1) TODO
        '''
        assert(qs.shape[1] == emm.shape[2] == emu.shape[2])
        loss_match, _ = self.funcmap['QA-'](qs, emm, Xs)
        #loss_match, _ = self.funcmap['QA-DIST'](qs, emm, Xs, pm='-')
        #loss_unmatch, _ = self.funcmap['QA+'](qs, emu, Xs)
        #loss_unmatch, _ = self.funcmap['QA-DIST'](qs, emu, Xs, pm='+')
        loss = loss_match  # + loss_unmatch
        # [scratch]
        #emm = emm.squeeze()
        #emu = emu.squeeze()
        #ems = ems.squeeze()
        # if self.metric in ('C',):
        #    #l_m = -(1 - F.cosine_similarity(qs, emm))
        #    l_u = (1 - F.cosine_similarity(qs, emu))
        # elif self.metric in ('E', 'N'):
        #    #l_m = -F.pairwise_distance(qs, emm)
        #    #l_u = F.pairwise_distance(qs, emu)
        #    l_s = -F.pairwise_distance(qs, ems)
        #loss = (l_s).mean()
        return loss

    def RankLossTargetedMismatchAttack(
            self, qs: th.Tensor, embrand: th.Tensor):
        '''
        Targeted Mismatch Attack using Global Descriptor (ICCV'19)
        https://arxiv.org/pdf/1908.09163.pdf
        '''
        assert(qs.shape[0] == embrand.shape[0])
        assert(self.metric in ('C', 'N'))
        loss = (1 - F.cosine_similarity(qs, embrand)).mean()
        return loss

    def RankLossLearningToMisrank(self, qs: th.Tensor, embp: th.Tensor,
                                  embn: th.Tensor):
        '''
        Learning-To-Mis-Rank
        But the paper did not specify a margin.
        Following Eq.1 of https://arxiv.org/pdf/2004.04199.pdf
        '''
        assert(qs.shape == embp.shape == embn.shape)
        if self.metric == 'C':
            loss = (1 - F.cosine_similarity(qs, embn)) - \
                   (1 - F.cosine_similarity(qs, embp))
        elif self.metric in ('N', 'E'):
            loss = F.pairwise_distance(qs, embn) - \
                F.pairwise_distance(qs, embp)
        return loss.mean()

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
            'GTM': self.RankLossGreedyTop1Misrank,
            'GTT': self.RankLossGreedyTop1Translocation,
            'TMA': self.RankLossTargetedMismatchAttack,
            'LTM': self.RankLossLearningToMisrank,
        }
        if request not in self.funcmap.keys():
            raise KeyError(f'Requested loss function "{request}" not found!')
        self.request = request

    def __call__(self, *args, **kwargs):
        '''
        Note, you should handle the normalization outside of this class.
        The input and output of the function also vary based on the request.
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


@pytest.mark.parametrize('pm, metric', it.product('+-', 'NEC'))
def test_arl_qadist(pm: str, metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    Cs = th.rand(10, 20, 8)
    Xs = th.rand(30, 8)
    loss, _ = AdvRankLoss(f'QA-DIST', metric)(qs, Cs, Xs, pm=pm)


@pytest.mark.parametrize('pm, metric', it.product(('+', '-'), ('C', 'E', 'N')))
def test_arl_ca(pm: str, metric: str):
    cs = th.rand(10, 8, requires_grad=True)
    Qs = th.rand(10, 20, 8)
    Xs = th.rand(30, 8)
    loss, _ = AdvRankLoss(f'CA{pm}', metric)(cs, Qs, Xs)
    loss.backward()


@pytest.mark.parametrize('pm, metric', it.product('+-', 'NEC'))
def test_arl_cadist(pm: str, metric: str):
    cs = th.rand(10, 8, requires_grad=True)
    Qs = th.rand(10, 20, 8)
    Xs = th.rand(30, 8)
    loss, _ = AdvRankLoss('CA-DIST', metric)(cs, Qs, Xs, pm=pm)


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


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_arl_gtm(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    emm = th.rand(10, 1, 8)
    emu = th.rand(10, 1, 8)
    Xs = th.rand(30, 8)
    loss = AdvRankLoss('GTM', metric)(qs, emm, emu, emm, Xs)
    loss.backward()


@pytest.mark.parametrize('metric', 'NEC')
def test_arl_gtt(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    emm = th.rand(10, 1, 8)
    emu = th.rand(10, 1, 8)
    Xs = th.rand(30, 8)
    loss = AdvRankLoss('GTT', metric)(qs, emm, emu, emm, Xs)
    loss.backward()


@pytest.mark.parametrize('metric', 'NC')
def test_arl_tma(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    embrand = th.rand(10, 8)
    loss = AdvRankLoss('TMA', metric)(qs, embrand)
    loss.backward()


@pytest.mark.parametrize('metric', 'NEC')
def test_arl_ltm(metric: str):
    qs = th.rand(10, 8, requires_grad=True)
    emm = th.rand(10, 8)
    emu = th.rand(10, 8)
    loss = AdvRankLoss('LTM', metric)(qs, emm, emu)
    loss.backward()
