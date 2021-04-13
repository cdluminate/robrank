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

Loss functions borrowed from Pytorch-Metric-Learning
  https://kevinmusgrave.github.io/pytorch-metric-learning
'''
import numpy as np
import torch as th
from .. import configs
import pytorch_metric_learning as dml
import pytorch_metric_learning.losses
import pytorch_metric_learning.miners
import pytorch_metric_learning.reducers
import pytorch_metric_learning.distances


def _index_filter(indeces: tuple, most: int):
    '''
    Pytorch-metric-learning's miners outputs too many usable tuples
    so that OOM is very easy to trigger.
    '''
    sel = th.randint(len(indeces[0]), (most,)).to(indeces[0].device)
    return tuple(indeces[i][sel] for i in range(len(indeces)))


class ExtraLossN(th.nn.Module):

    def __call__(self, *args, **kwargs):
        repres, labels = args[0], args[1].view(-1)
        repres = th.nn.functional.normalize(repres, p=2)
        indeces = self._miner(repres, labels)
        indeces = _index_filter(indeces, repres.size(0))
        return self._lossfunc(repres, labels, indeces)

    def determine_metric(self):
        return self._metric

    def datasetspec(self):
        return self._datasetspec


class pstripN(ExtraLossN):
    _datasetspec = 'SPC-2'
    _lossfunc = dml.losses.TripletMarginLoss(
        margin=configs.triplet.margin_euclidean,
        reducer=dml.reducers.ThresholdReducer(low=0.),
        distance=dml.distances.LpDistance(
            p=2, power=1, normalize_embeddings=True)
    )
    _miner = dml.miners.TripletMarginMiner(
        margin=configs.triplet.margin_euclidean,
        type_of_triplets='semihard')
    _metric = 'N'


class pangularN(ExtraLossN):
    _datasetspec = 'SPC-2'
    _lossfunc = dml.losses.AngularLoss()
    _miner = dml.miners.AngularMiner()
    _metric = 'N'


class pcontN(ExtraLossN):
    _datasetspec = 'SPC-2'
    _lossfunc = dml.losses.ContrastiveLoss(
        pos_margin=configs.contrastive.margin_euclidean,
        neg_margin=configs.contrastive.margin_euclidean
    )
    _miner = dml.miners.PairMarginMiner(
        pos_margin=configs.contrastive.margin_euclidean,
        neg_margin=configs.contrastive.margin_euclidean)
    _metric = 'N'


class pncaN(ExtraLossN):
    _datasetspec = 'SPC-2'
    _lossfunc = dml.losses.NCALoss()
    _miner = dml.miners.TripletMarginMiner(
        margin=configs.triplet.margin_euclidean,
        type_of_triplets='semihard')
    _metric = 'N'


def test_pstripN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pstripN()(output, labels)
    loss.backward()


def test_pangularN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pangularN()(output, labels)
    loss.backward()


def test_pcontN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pcontN()(output, labels)
    loss.backward()

def test_pncaN():
    output, labels = th.rand(10, 32, requires_grad=True), th.randint(3, (10,))
    loss = pncaN()(output, labels)
    loss.backward()
