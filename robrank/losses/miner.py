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
import itertools as it
import functools as fun
import random
import torch.nn.functional as F
import pytest


def miner(repres: th.Tensor, labels: th.Tensor, *,
          method: str = 'random-triplet', metric: str = None,
          margin: float = None, p_switch: float = -1.0):
    '''
    Dispatcher for different batch data miners
    '''
    assert(len(repres.shape) == 2)
    assert(metric is not None)
    labels = labels.view(-1)
    if method == 'random-triplet':
        return __miner_random(repres, labels)
    elif method == 'spc2-random':
        anchor, positive, negative = __miner_spc2_random(repres, labels)
    elif method == 'spc2-semihard':
        anchor, positive, negative = __miner_spc2_semihard(
            repres, labels, metric, margin)
    elif method == 'spc2-hard':
        anchor, positive, negative = __miner_spc2_hard(repres, labels, metric)
    elif method == 'spc2-softhard':
        anchor, positive, negative = __miner_spc2_softhard(
            repres, labels, metric)
    elif method == 'spc2-distance':
        anchor, positive, negative = __miner_spc2_distance(
            repres, labels, metric)
    elif method == 'spc2-lifted':
        anchor, positive, negative = __miner_spc2_lifted(repres, labels)
    elif method == 'spc2-npair':
        anchor, positive, negative = __miner_spc2_npair(repres, labels)
    else:
        raise NotImplementedError
    if p_switch > 0.0 and (np.random.rand() < p_switch):
        # spectrum regularization (following ICML20 paper text description)
        # return (anchor, negative, positive)  # XXX: lead to notable performance drop
        # spectrum regulairzation (following upstream code)
        return (anchor, anchor, positive)
    return (anchor, positive, negative)


@pytest.mark.parametrize('mmethod, metric', it.product(['spc2-random',
                                                        'spc2-semihard', 'spc2-hard', 'spc2-softhard', 'spc2-distance',
                                                        'spc2-lifted', 'spc2-npair'], ['C', 'N', 'E']))
def test_miner(mmethod, metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = miner(repres, labels, metric=metric,
                          method=mmethod, margin=0.2)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(la is not None)
        assert(lp is not None)
        assert(ln is not None)


def __miner_spc2_npair(repres: th.Tensor, labels: th.Tensor) -> tuple:
    '''
    Miner for N-Pair loss
    return type is a little bit special:
        (int, int, list[int]) or alike
    '''
    negatives = []
    for i in range(repres.size(0) // 2):
        # identify negative
        mask_lneg = (labels[2 * i] != labels)
        if mask_lneg.sum() > 0:
            negatives.append(th.where(mask_lneg)[0])
        else:
            negatives.append([np.random.choice(len(labels))])
    anchors = th.arange(0, len(labels), 2)
    positives = th.arange(1, len(labels), 2)
    return (anchors, positives, negatives)


def test_miner_spc2_npair():
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_npair(repres, labels)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, sn) in zip(anc, pos, neg):
        la, lp = labels[a], labels[p]
        for n in sn:
            ln = labels[n]
            assert(a != p and a != n and p != n)
            assert(la == lp and la != ln)


def __miner_spc2_lifted(repres: th.Tensor, labels: th.Tensor) -> tuple:
    '''
    Miner for generalized lifted-structure loss function

    The return type is a little bit special:
      (int, list[int], list[int]) or alike.
    '''
    positives, negatives = [], []
    ###
    for i in range(repres.size(0) // 2):
        # identify positive
        mask_lpos = (labels[2 * i] == labels)
        positive = list(set(th.where(mask_lpos)[0].tolist()) - {2 * i, })
        positives.append(positive)
        # identify negative
        mask_lneg = (labels[2 * i] != labels)
        if mask_lneg.sum() > 0:
            negatives.append(th.where(mask_lneg)[0])
        else:
            negatives.append([np.random.choice(len(labels))])
    anchors = th.arange(0, len(labels), 2)
    return (anchors, positives, negatives)


def test_miner_spc2_lifted():
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_lifted(repres, labels)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, sp, sn) in zip(anc, pos, neg):
        la = labels[a]
        for n in sn:
            ln = labels[n]
            for p in sp:
                lp = labels[p]
                assert(a != p and a != n and p != n)
                assert(la == lp and la != ln)


def __miner_pdist(repres: th.Tensor, metric: str) -> th.Tensor:
    '''
    Helper: compute pairwise distance matrix.
    '''
    assert(len(repres.shape) == 2)
    with th.no_grad():
        if metric == 'C':
            repres = F.normalize(repres, dim=-1)
            pdist = 1.0 - th.mm(repres, repres.t())
        elif metric in ('E', 'N'):
            if metric == 'N':
                repres = F.normalize(repres, dim=-1)
            # Memory efficient pairwise euclidean distance matrix
            prod = th.mm(repres, repres.t())
            norm = prod.diag().unsqueeze(1).expand_as(prod)
            pdist = (norm + norm.t() - 2 * prod).sqrt()
        else:
            raise ValueError(f'illegal metric {metric}')
    return pdist


def __miner_inverse_sphere_distance(
        dists: th.Tensor, labels: th.Tensor, thisidx: int, dim: float) -> np.array:
    '''
    Reference ICML20 revisiting ...
    '''
    log_q_d_inv = (2.0 - dim) * th.log(dists) - ((dim - 3.0) /
                                                 2.0) * th.log(1.0 - 0.25 * dists.pow(2))
    log_q_d_inv[th.where(labels == labels[thisidx])[0]] = 0.
    q_d_inv = th.exp(log_q_d_inv - log_q_d_inv.max())
    q_d_inv[th.where(labels == labels[thisidx])[0]] = 0.
    q_d_inv = np.nan_to_num(q_d_inv.detach().cpu().numpy())
    if q_d_inv.sum() == 0.:
        q_d_inv[:] = 1e-7
    q_d_inv = q_d_inv / q_d_inv.sum()
    if np.isnan(q_d_inv).sum() > 0:
        # remove the NaN elements, or np.random.choice would complain
        nan_mask = np.isnan(q_d_inv)
        q_d_inv[nan_mask] = 0.
        residual = np.max([1.0 - q_d_inv.sum(), 0.0])
        q_d_inv[nan_mask] = residual / nan_mask.sum()
    return q_d_inv


def __miner_spc2_distance(
        repres: th.Tensor, labels: th.Tensor, metric: str) -> tuple:
    '''
    Distance-weighted tuple mining (Wu et al. 2017)
    (unit hyper-sphere)
    '''
    negs = []
    pdist = __miner_pdist(repres, metric)
    for i in range(repres.size(0) // 2):
        # inverse distribution
        inv_q_d = __miner_inverse_sphere_distance(
            pdist[2 * i, :], labels, 2 * i, repres.size(1))
        negs.append(np.random.choice(len(labels), p=inv_q_d))
    anchors = th.arange(0, len(labels), 2)
    positives = th.arange(1, len(labels), 2)
    negatives = th.tensor(negs, dtype=th.long, device=repres.device)
    return (anchors, positives, negatives)


@pytest.mark.parametrize('metric', ('C', 'N'))
def test_miner_spc2_distance(metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_distance(repres, labels, metric=metric)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(a != p and a != n and p != n)
        assert(la == lp and la != ln)


def __miner_spc2_semihard(
        repres: th.Tensor, labels: th.Tensor, metric: str, margin: float) -> tuple:
    '''
    Sampling semihard negatives from pairwise (SPC-2) data batch.
    '''
    negs = []
    pdist = __miner_pdist(repres, metric)
    for i in range(repres.size(0) // 2):
        # XXX: the > is actually a part of Softhard
        # mask_pdist = ((pdist[i, 2 * i] - pdist[i, 2 * i + 1]).pow(2) >
        #               (pdist[i, 2 * i] - pdist[i, :]).pow(2))
        mask_pdist = (pdist[2 * i, 2 * i + 1].pow(2) < pdist[2 * i, :].pow(2))
        mask_tripl = (pdist[2 * i, 2 * i + 1] - pdist[2 * i, :] + margin > 0.0)
        mask_label = (labels != labels[2 * i])
        mask = fun.reduce(th.logical_and, [mask_pdist, mask_tripl, mask_label])
        if mask.sum() > 0:
            argwhere = th.where(mask)[0]
        elif mask_label.sum() > 0:
            argwhere = th.where(mask_label)[0]
        else:
            argwhere = th.arange(len(labels))
        negs.append(random.choice(argwhere).item())
    anchors = th.arange(0, len(labels), 2)
    positives = th.arange(1, len(labels), 2)
    negatives = th.tensor(negs, dtype=th.long, device=repres.device)
    return (anchors, positives, negatives)


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_miner_spc2_semihard(metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_semihard(
        repres, labels, metric=metric, margin=0.2)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(a != p and a != n and p != n)
        assert(la == lp and la != ln)


def __miner_spc2_softhard(
        repres: th.Tensor, labels: th.Tensor, metric: str) -> tuple:
    '''
    Sampling softhard negatives from pairwise (SPC-2) data batch.
    '''
    negs, poss = [], []
    pdist = __miner_pdist(repres, metric)
    for i in range(repres.size(0) // 2):
        # mark positive and negative
        mask_lneg = (labels != labels[2 * i])
        mask_lpos = (labels == labels[2 * i])
        # sample soft negative
        if mask_lneg.sum() > 0:
            maxap2 = th.masked_select(pdist[2 * i, :], mask_lpos).max().pow(2)
            mask_sneg = th.logical_and(
                pdist[2 * i, :].pow(2) < maxap2, mask_lneg)
            if mask_sneg.sum() > 0:
                argwhere = th.where(mask_sneg)[0]
            else:
                argwhere = th.where(mask_lneg)[0]
        else:
            argwhere = th.arange(len(labels))
        negs.append(random.choice(argwhere).item())
        # sample soft positive
        if mask_lpos.sum() > 0:
            if mask_lneg.sum() > 0:
                minan2 = th.masked_select(
                    pdist[2 * i, :], mask_lneg).min().pow(2)
                mask_spos = th.logical_and(
                    pdist[2 * i, :].pow(2) > minan2, mask_lpos)
                if mask_spos.sum() > 0:
                    argwhere = th.where(mask_spos)[0]
                else:
                    argwhere = th.where(mask_lpos)[0]
                poss.append(random.choice(argwhere).item())
            else:
                poss.append(2 * i + 1)
        else:
            poss.append(2 * i + 1)
    anchors = th.arange(0, len(labels), 2)
    positives = th.tensor(poss, dtype=th.long, device=repres.device)
    negatives = th.tensor(negs, dtype=th.long, device=repres.device)
    return (anchors, positives, negatives)


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_miner_spc2_softhard(metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_softhard(repres, labels, metric=metric)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(a != p and a != n and p != n)
        assert(la == lp and la != ln)


def __miner_spc2_hard(
        repres: th.Tensor, labels: th.Tensor, metric: str) -> list:
    '''
    Sampling hard negatives from pairwise (SPC-2) data batch.
    XXX: Very unstable due to noisy hardest.
    '''
    negs = []
    pdist = __miner_pdist(repres, metric)
    for i in range(repres.size(0) // 2):
        mask_label = (labels != labels[2 * i])
        dist = mask_label * pdist[2 * i, :]
        nonzero = dist.nonzero(as_tuple=False).flatten()
        nzargmin = nonzero[dist[nonzero].argmin()].item()
        negs.append(nzargmin)
    anchors = th.arange(0, len(labels), 2)
    positives = th.arange(1, len(labels), 2)
    negatives = th.tensor(negs, dtype=th.long, device=repres.device)
    return (anchors, positives, negatives)


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_miner_spc2_hard(metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_hard(repres, labels, metric=metric)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(a != p and a != n and p != n)
        assert(la == lp and la != ln)


def __miner_spc2_random(
        repres: th.Tensor, labels: th.Tensor) -> (th.Tensor, th.Tensor, th.Tensor):
    '''
    Sampling triplets from pairwise data
    '''
    negs = []
    for i in range(labels.nelement() // 2):
        # [ method 1: 40it/s legion
        mask_neg = (labels != labels[2 * i])
        if mask_neg.sum() > 0:
            negs.append(random.choice(th.where(mask_neg)[0]).item())
        else:
            # handle rare/corner cases where the batch is bad
            negs.append(np.random.choice(len(labels)))
        # [ method 2: 35it/s legion
        # candidates = tuple(filter(lambda x: x // 2 != i,
        #                          range(labels.nelement())))
        # while True:
        #    neg = random.sample(candidates, 1)
        #    if labels[i * 2].item() != labels[neg].item():
        #        break
        # negs.append(*neg)
    anchors = th.arange(0, len(labels), 2)
    positives = th.arange(1, len(labels), 2)
    negatives = th.tensor(negs, dtype=th.long, device=repres.device)
    return (anchors, positives, negatives)


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_miner_spc2_random(metric):
    repres = F.normalize(th.rand(10, 8))
    labels = th.randint(3, (5,)).view(-1, 1).expand(5, 2).flatten()
    anc, pos, neg = __miner_spc2_random(repres, labels)
    assert(len(anc) == len(pos) and len(anc) == len(neg))
    for (a, p, n) in zip(anc, pos, neg):
        la, lp, ln = labels[a], labels[p], labels[n]
        assert(a != p and a != n and p != n)
        assert(la == lp and la != ln)


def __miner_random(repres: th.Tensor, labels: th.Tensor):
    if isinstance(labels, th.Tensor):
        labels = labels.detach().cpu().numpy()
    # {0.052} sec the commented implementation is slower
    # uniq, counts = np.unique(labels, return_counts=True)
    # if all(x < 2 for x in counts):
    #   # No Anchor-Positive Collision, We use a fallback A-A-N strategy
    #   sampled_triplets = [(x, x, random.choice(tuple(set(range(len(labels))) - {x}))) for x in range(len(labels))]
    # else:
    #   cls2idx = {i: set(np.argwhere(i == labels).ravel()) for i in uniq}
    #   apcomb = [list(it.product(cls2idx[x], cls2idx[x])) for (i,x) in enumerate(uniq) if counts[i]>1]
    #   apcomb = [list(it.filterfalse(lambda x: x[0] == x[1], x)) for x in apcomb]
    #   negs = [set(range(len(labels))) - set(cls2idx[x]) for (i,x) in enumerate(uniq) if counts[i]>1]
    #   groups = [list((*xx, yy) for (xx, yy) in it.product(x, y)) for (x,y) in zip(apcomb, negs)]
    #   sampled_triplets = list(fun.reduce(list.__add__, groups))

    # {0.051} sec slow
    # uniq, counts = np.unique(labels, return_counts=True)
    # if all(x < 2 for x in counts):
    #    # No Anchor-Positive Collision, We use a fallback A-A-N strategy
    #    sampled_triplets = [(x, x, random.choice(tuple(set(range(len(labels))) - {x}))) for x in range(len(labels))]
    # else:
    #    cls2idx = {i: set(np.argwhere(i == labels).ravel()) for i in uniq}
    #    negs = {x: set(range(len(labels))) - set(cls2idx[x]) for (i,x) in enumerate(uniq) if counts[i]>1}
    #    apncomb = [list(it.product(it.filterfalse(lambda y: y[0] == y[1],
    #       it.product(cls2idx[x], cls2idx[x])), negs[x])) for (i,x) in enumerate(uniq) if counts[i]>1]
    #    sampled_triplets = [(*x, y) for z in apncomb for (x, y) in z]

    # {0.046} sec fastest
    unique_classes, counts = np.unique(labels, return_counts=True)
    if all(x < 2 for x in counts):
        # No Anchor-Positive Collision, We use a fallback A-A-N strategy
        sampled_triplets = [(x, x, random.choice(
            tuple(set(range(len(labels))) - {x}))) for x in range(len(labels))]
    else:
        class_dict = {i: np.argwhere(labels == i).ravel()
                      for i in unique_classes}
        sampled_triplets = [list(it.product(
            [x], [x], [y for y in unique_classes if x != y])) for x in unique_classes]
        sampled_triplets = [x for y in sampled_triplets for x in y]
        sampled_triplets = [[x for x in list(it.product(
            *[class_dict[j] for j in i])) if x[0] != x[1]] for i in sampled_triplets]
        sampled_triplets = [x for y in sampled_triplets for x in y]
    try:
        sampled_triplets = random.sample(sampled_triplets, repres.shape[0])
    except ValueError:
        sampled_triplets = random.choices(sampled_triplets, k=repres.shape[0])
    return sampled_triplets
