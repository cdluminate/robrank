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

###################
# Dataset Configs #
###################


@dataclass
class mnist:
    path: str = os.path.expanduser('~/.torch')


@dataclass
class fashion:
    path: str = os.path.expanduser('~/.torch')


@dataclass
class cifar10:
    path: str = os.path.expanduser('~/.torch/cifar-10-batches-py')


@dataclass
class sop:
    path: str = os.path.expanduser('/dev/shm/Stanford_Online_Products/') if \
        os.path.exists(os.path.expanduser('/dev/shm/Stanford_Online_Products/')) \
        else os.path.expanduser('~/.torch/Stanford_Online_Products/')
    list_train: str = 'Ebay_train.txt'
    list_test: str = 'Ebay_test.txt'


@dataclass
class cub:
    path: str = os.path.expanduser('/dev/shm/CUB_200_2011/') if \
        os.path.exists(os.path.expanduser('/dev/shm/CUB_200_2011/')) else \
        os.path.expanduser('~/.torch/CUB_200_2011/')
    list_images: str = 'images.txt'
    list_split: str = 'train_test_split.txt'


@dataclass
class cars:
    path: str = os.path.expanduser('/dev/shm/cars/') if \
            os.path.exists('/dev/shm/cars') else \
            os.path.expanduser('~/.torch/cars/')
