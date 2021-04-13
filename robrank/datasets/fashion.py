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
import torch.utils.data
import torchvision as vision
from torchvision import transforms
import os
import multiprocessing as mp
from .. import configs
from collections import defaultdict
import random
from .mnist import _MNIST_TRIPLET, MNISTPairDataset


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return _get_classification_dataset()
    elif kind == 'SPC-2':
        return __get_spc2_dataset()
    elif kind == 'triplet':
        return _get_triplet_dataset()
    else:
        raise NotImplementedError


def _get_classification_dataset():
    dataset = vision.datasets.FashionMNIST(configs.fashion.path,
                                           train=True, download=True, transform=transforms.ToTensor())
    train, val = th.utils.data.random_split(dataset, [55000, 5000])
    test = vision.datasets.FashionMNIST(configs.fashion.path,
                                        train=False, download=True, transform=transforms.ToTensor())
    return (train, val, test)


def _get_triplet_dataset():
    train = _MNIST_TRIPLET(
        configs.fashion.path,
        train=True,
        name='FashionMNIST')
    test = _MNIST_TRIPLET(
        configs.fashion.path,
        train=False,
        name='FashionMNIST')
    return (train, test)


def __get_spc2_dataset():
    train = MNISTPairDataset(configs.fashion.path,
                             train=True, name='FashionMNIST')
    test = MNISTPairDataset(configs.fashion.path,
                            train=False, name='FashionMNIST')
    return (train, test)
