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
import torch.nn.functional as F
import numpy as np
from .. import configs
import functools as ft
import itertools as it
import pytest


def fn_multisim(repres: th.Tensor, labels: th.Tensor, *, metric: str,
                pos_weight: float = configs.multisim.pos_weight,
                neg_weight: float = configs.multisim.neg_weight,
                margin: float = configs.multisim.margin,
                threshold: float = configs.multisim.threshold,
                ):
    '''
    MultiSimilarity Loss (Wang et al., 2019)
    '''
    assert(len(repres.shape) == 2)
    # calculate similarity matrix
    repres = F.normalize(repres)
    sim = repres.mm(repres.T)
    labels = labels.view(-1)
    # Loss
    loss = []
    for i in np.arange(0, repres.size(0), 2):
        #
        maskpos = (labels == labels[i])
        maskneg = (labels != labels[i])
        if maskneg.sum() == 0:
            maskneg = np.random.choice(repres.size(0))
        sAP = sim[i][maskpos]
        sAN = sim[i][maskneg]
        # we need many positives in the batch
        maskpos = (sAP + margin) > th.min(sAP)
        maskneg = (sAN - margin) < th.max(sAN)
        if maskpos.sum() == 0 or maskneg.sum() == 0:
            continue
        sAP = sAP[maskpos]
        sAN = sAN[maskneg]
        #
        ipos = th.log(
            1 + th.sum(th.exp(-pos_weight * (sAP - threshold)))) / pos_weight
        ineg = th.log(
            1 + th.sum(th.exp(+neg_weight * (sAN - threshold)))) / neg_weight
        loss.append(ipos + ineg)
    loss = th.mean(th.stack(loss))
    return loss


class pmsC(th.nn.Module):
    _metric = 'C'
    _datasetspec = 'SPC-2'

    def __call__(self, *args, **kwargs):
        return ft.partial(fn_multisim, metric=self._metric)(*args, **kwargs)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pmsN(pmsC):
    _metric = 'N'


@pytest.mark.parametrize('metric', ('C', 'E', 'N'))
def test_fn_multisim(metric):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = fn_multisim(output, labels, metric=metric)
    loss.backward()


@pytest.mark.parametrize('func', (pmsC, pmsN))
def test_multisim(func):
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = func()(output, labels)
    loss.backward()
