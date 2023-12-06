import torch as th
import torch.nn.functional as F
import pytest
import numpy as np
from .miner import miner
import itertools as it
import functools as ft

__ALL__ = ['piceC']

'''
fashion:rc2f2:piceC
t=0    R@1 = 88.8
t=0.07 R@1 = 87.9
t=0.2  R@1 = 88.4
'''

def infonce(repA: th.Tensor, repB: th.Tensor, *, metric:str='C', t:float = 0.2) -> th.Tensor:
    # make sure shape is correct
    repA, repB = th.flatten(repA, 1), th.flatten(repB, 1)
    assert metric == 'C'
    # 
    repA = F.normalize(repA)
    repB = F.normalize(repB)
    cos = th.mm(repA, repB.T)
    logits = cos * th.exp(th.tensor(t, device=repA.device))
    pseudo_labels = th.arange(repA.size(0), device=repA.device)
    loss = F.cross_entropy(logits, pseudo_labels) * 0.5 \
            + F.cross_entropy(logits.t(), pseudo_labels) * 0.5
    return loss


def fn_infonce(repres: th.Tensor, labels: th.Tensor, *,
               metric: str = 'C', minermethod: str = 'spc2-random'):
    ancs, poss, negs = miner(repres, labels, method=minermethod, metric=metric)
    loss = infonce(repres[ancs, :], repres[poss, :], metric=metric)
    return loss
    

class piceC(th.nn.Module):
    _metric = 'C'
    _datasetspec = 'SPC-2'
    _minermethod = 'spc2-random'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_infonce, metric=self._metric,
                          minermethod=self._minermethod)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec
