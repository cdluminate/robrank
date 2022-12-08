'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

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

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring
# pylint: disable=useless-super-delegation
# pylint: disable=unused-argument
import re
from dataclasses import dataclass
import multiprocessing as mp
import os
import rich
c = rich.get_console()

#################
# Model Configs #
#################


@dataclass
class __ranking:
    '''the ranking task'''
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
        'pgilC', 'pgilE', 'pgilN', 'ptripxaN',
        # extra.py: borrowed functions
        'pstripN',
        'pangularN',
        'pcontN',
        'pncaN',
        'psnrN', 'psnrE', 'psnrC',
        # barlow twins extension
        'ptripletBTN', 'pmtripletBTN', 'pstripletBTN',
        'pdtripletBTN', 'phtripletBTN',
    )


@dataclass
class __ranking_model_28x28(__ranking):
    '''
    Set pgditer = 1 to enable FGSM adversarial training.
    '''
    allowed_datasets: tuple = ('mnist', 'fashion')
    embedding_dim: int = 512
    advtrain_eps: float = 77. / 255.
    advtrain_alpha: float = 3. / 255.
    advtrain_pgditer: int = 32

    def __init__(self, dataset, loss):
        # used for overriding this configuration.
        if os.path.exists('override_pgditer_16'):
            self.advtrain_pgditer = 16
            c.print('[bold yellow]! Overriding advtrain_pgditer to 16 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_8'):
            self.advtrain_pgditer = 8
            c.print('[bold yellow]! Overriding advtrain_pgditer to 8 as' +
                    ' indicated by override[/bold yellow]')

@dataclass
class __ranking_model_224x224(__ranking):
    '''
    Set pgditer = 1 to enable FGSM adversarial training.

    [ Note ]
    If you find the adversarial training with pgditer=32
    (32 steps of PGD update) extremely slow, you may
    decrease that value to, e.g., 8. We will use 32 by
    default to retain consistency and the best robustness.
    '''
    allowed_datasets: tuple = ('sop', 'cub', 'cars')
    embedding_dim: int = 512
    advtrain_eps: float = 8. / 255.
    advtrain_alpha: float = 1. / 255.
    advtrain_pgditer: int = 32

    def __init__(self, dataset, loss):
        if dataset == 'cub':
            self.num_class = 100
        elif dataset == 'cars':
            self.num_class = 98
        elif dataset == 'sop':
            self.num_class = 11316
        # used for overriding this configuration.
        if os.path.exists('override_pgditer_16'):
            self.advtrain_pgditer = 16
            c.print('[bold yellow]! Overriding advtrain_pgditer to 16 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_15'):
            self.advtrain_pgditer = 15
            c.print('[bold yellow]! Overriding advtrain_pgditer to 15 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_8'):
            self.advtrain_pgditer = 8
            c.print('[bold yellow]! Overriding advtrain_pgditer to 8 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_7'):
            self.advtrain_pgditer = 7
            c.print('[bold yellow]! Overriding advtrain_pgditer to 7 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_4'):
            self.advtrain_pgditer = 4
            c.print('[bold yellow]! Overriding advtrain_pgditer to 4 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_3'):
            self.advtrain_pgditer = 3
            c.print('[bold yellow]! Overriding advtrain_pgditer to 3 as' +
                    ' indicated by override[/bold yellow]')
        if os.path.exists('override_pgditer_2'):
            self.advtrain_pgditer = 2
            c.print('[bold yellow]! Overriding advtrain_pgditer to 2 as' +
                    ' indicated by override[/bold yellow]')


@dataclass
class __ranking_model_224x224_icml(__ranking_model_224x224):
    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
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
class rc2f2(__ranking_model_28x28):
    maxepoch: int = 16  # eph-with-cls-batch, equals 2 * eph-with-spc2-batch
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1
    valbatchsize: int = 128
    num_class: int = 10

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
        if re.match(r'p.+', loss):
            self.maxepoch //= 2
        if re.match(r't.+', loss):
            self.maxepoch //= 3


@dataclass
class rlenet(rc2f2):
    embedding_dim: int = 128


@dataclass
class rres18(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112
    optimizer: str = 'Adam'

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)


@dataclass
class rres50(__ranking_model_224x224_icml):
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
class rmnas(__ranking_model_224x224_icml):
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
class rswint(__ranking_model_224x224_icml):
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
class ribn(__ranking_model_224x224_icml):
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
class reffb0(__ranking_model_224x224_icml):
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
class reffb4(__ranking_model_224x224_icml):
    maxepoch: int = 150
    loader_num_workers: int = min(8, mp.cpu_count())
    lr: float = 1e-5  # [lock]
    weight_decay: float = 4e-4  # [lock] 2002.08473
    embedding_dim: int = 512
    freeze_bn: bool = True
    valbatchsize: int = 112

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
