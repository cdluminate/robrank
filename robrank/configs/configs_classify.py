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

from dataclasses import dataclass
import multiprocessing as mp

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
    allowed_datasets: tuple = ('cifar10', 'cifar100')


@dataclass
class __classify_model_224x224(__classify):
    allowed_datasets: tuple = ('ilsvrc',)


@dataclass
class cslp(__classify_model_28x28):
    maxepoch: int = 10
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1


@dataclass
class clenet(__classify_model_28x28):
    maxepoch: int = 10
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1


@dataclass
class cc2f2(__classify_model_28x28):
    maxepoch: int = 16
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-7
    validate_every: int = 1


@dataclass
class cres18(__classify_model_32x32):
    '''
    The default configuration is used for CIFAR10.
    '''
    maxepoch: int = 200  # [lock] resnet
    validate_every: int = 1
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 256  # [lock] resnet
    lr: float = 1e-1  # [lock] resnet
    momentum: float = 9e-1  # [lock] resnet
    milestones: tuple = (100, 150)  # [lock] resnet
    weight_decay: float = 2e-4  # [lock] resnet
    validate_every: int = 1

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
        if dataset == 'cifar100':
            self.maxepoch = 300
            self.milestones = (100, 200)


@dataclass
class cres50(__classify_model_32x32):
    '''
    The default configuration is used for CIFAR10.
    '''
    maxepoch: int = 200  # [lock] resnet
    validate_every: int = 1
    loader_num_workers: int = min(8, mp.cpu_count())
    batchsize: int = 256  # [lock] resnet
    lr: float = 1e-1  # [lock] resnet
    momentum: float = 9e-1  # [lock] resnet
    milestones: tuple = (100, 150)  # [lock] resnet
    weight_decay: float = 2e-4  # [lock] resnet
    validate_every: int = 1

    def __init__(self, dataset, loss):
        super().__init__(dataset, loss)
        if dataset == 'cifar100':
            self.maxepoch = 300
            self.milestones = (100, 200)
