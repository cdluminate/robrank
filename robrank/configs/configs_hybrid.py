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

from dataclasses import dataclass
import dataclasses
import os
import multiprocessing as mp
import torch as th
import re

#################
# Model Configs #
#################


@dataclass
class __classify:
    allowed_losses: tuple = ('ce',)


@dataclass
class __classify_model_28x28(__classify):
    allowed_datasets: tuple = ('mnist', 'fashion')


@dataclass
class __classify_model_32x32(__classify):
    allowed_datasets: tuple = ('cifar10',)


@dataclass
class __classify_model_224x224(__classify):
    allowed_datasets: tuple = ('ilsvrc',)


@dataclass
class __ranking:
    allowed_losses: tuple = (
        'contrast',  # default setting of contrastive loss
        'contrastiveC', 'contrastiveE',
        'pcontrastC', 'pcontrastE', 'pcontrastN',
        'pdcontrastN', 'pDcontrastN',
        'ctripletC', 'ctripletE', 'ctripletN',
        'ptripletC', 'ptripletE', 'ptripletN',
        'pmtripletC', 'pmtripletE', 'pmtripletN',
        'phtripletC', 'phtripletE', 'phtripletN',
        'pstripletC', 'pstripletE', 'pstripletN',
        'pdtripletN', 'pDtripletN',
        'ttripletC', 'ttripletE', 'ttripletN',
        'pgliftE',  # only E makes sense for pglift
        'pmarginC', 'pmarginE', 'pmarginN',
        'pdmarginN', 'pDmarginN',
        'pnpairE',  # only E makes sense for pnpair, presumably
        'pquadC', 'pquadE', 'pquadN', 'pdquadN',
        'prhomC', 'prhomE', 'prhomN', 'pdrhomN',
        'pmsC', 'pmsN',
        # extra.py: borrowed functions
        'pstripN',
        'pangularN',
        'pcontN',
        'pncaN',
    )


@dataclass
class __ranking_model_28x28(__ranking):
    allowed_datasets: tuple = ('mnist', 'fashion')
    advtrain_eps: float = 0.3


@dataclass
class __ranking_model_224x224(__ranking):
    allowed_datasets: tuple = ('sop', 'cub', 'cars')
    advtrain_eps: float = 16. / 255.

    def __init__(self, dataset, loss):
        if dataset == 'cub':
            self.num_class = 100
        elif dataset == 'cars':
            self.num_class = 98
        elif dataset == 'sop':
            self.num_class = 11316


@dataclass
class __ranking_model_224x224_icml(__ranking_model_224x224):
    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
        if dataset == 'sop':
            self.maxepoch = 100
        self.validate_every = {'sop': 5, 'cub': 5, 'cars': 5}[dataset]
        if re.match(r'c.+', loss):
            self.batchsize = 112  # [lock]
        elif re.match(r'p.+', loss):
            self.batchsize = 56  # [lock]
            self.maxepoch //= 2
        elif re.match(r't.+', loss):
            self.batchsize = 37  # [lock]
            self.maxepoch //= 3


@dataclass
class lenet(__classify_model_28x28):
    maxepoch: int = 10
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1


@dataclass
class c2f2(__classify_model_28x28):
    maxepoch: int = 16
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1


@dataclass
class c2f1(__ranking_model_28x28):
    maxepoch: int = 16  # eph-with-cls-batch, equals 2 * eph-with-spc2-batch
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1
    valbatchsize: int = 128
    num_class: int = 10

    def __init__(self, dataset, loss):
        if re.match(r'p.+', loss):
            self.maxepoch //= 2
        if re.match(r't.+', loss):
            self.maxepoch //= 3


@dataclass
class csres18(__classify_model_32x32):
    maxepoch: int = 200  # [lock] resnet
    validate_every: int = 1
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 256  # [lock] resnet
    lr: float = 1e-1  # [lock] resnet
    momentum: float = 9e-1  # [lock] resnet
    milestones: tuple = (100, 150)  # [lock] resnet
    weight_decay: float = 2e-4  # [lock] resnet
    validate_every: int = 1


@dataclass
class res18(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)


@dataclass
class res50(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)


@dataclass
class mnas(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)


@dataclass
class enb0(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)


@dataclass
class enb4(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = -1
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
