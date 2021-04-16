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
import torch as th
import torchvision as vision
from torchvision import transforms
import os
from .. import configs
from collections import defaultdict
import random


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return _get_classification_dataset()
    elif kind == 'SPC-2':
        return __get_spc2_dataset()
    elif kind == 'triplet':
        return _get_triplet_dataset()
    else:
        raise NotImplementedError


def __get_spc2_dataset():
    train = MNISTPairDataset(configs.mnist.path, train=True)
    test = MNISTPairDataset(configs.mnist.path, train=False)
    return (train, test)


def _get_classification_dataset():
    dataset = vision.datasets.MNIST(configs.mnist.path,
                                    train=True, download=True, transform=transforms.ToTensor())
    train, val = th.utils.data.random_split(dataset, [55000, 5000])
    test = vision.datasets.MNIST(configs.mnist.path,
                                 train=False, download=True, transform=transforms.ToTensor())
    return (train, val, test)


class _MNIST_TRIPLET(th.utils.data.Dataset):
    def __init__(self, path, train: bool = True, name: str = 'MNIST'):
        import gzip
        import numpy as np
        self.kind = 'train' if train else 't10k'
        labels_path = os.path.join(
            path, f'{name}/raw/%s-labels-idx1-ubyte.gz' % self.kind)
        images_path = os.path.join(
            path, f'{name}/raw/%s-images-idx3-ubyte.gz' % self.kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(
                lbpath.read(), dtype=np.uint8, offset=8).reshape(-1)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(
                imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        self.labels = th.from_numpy(labels)
        self.labelset = set(list(labels))
        self.images = th.from_numpy(images).view(-1, 1, 28, 28) / 255.0
        self.cls2idx = defaultdict(list)
        for (i, lb) in enumerate(self.labels):
            self.cls2idx[lb.item()].append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        if index >= len(self.labels):
            raise IndexError
        anchor = self.images[index]
        label = self.labels[index]
        if self.kind == 'train':
            posidx = random.choice(list(self.cls2idx[label.item()]))
            posimg = self.images[posidx]
            negidx = random.choice(self.cls2idx[
                random.choice(list(self.labelset - {label.item()}))])
            negimg = self.images[negidx]
            assert(label.item() == self.labels[posidx].item())
            assert(label.item() != self.labels[negidx].item())
            return th.stack([anchor, posimg, negimg]), label
        else:
            return anchor, label


class MNISTPairDataset(th.utils.data.Dataset):
    def __init__(self, path: str, train: bool = True, *, name: str = 'MNIST'):
        import gzip
        import numpy as np
        self.kind = 'train' if train else 't10k'
        labels_path = os.path.join(
            path, f'{name}/raw/%s-labels-idx1-ubyte.gz' % self.kind)
        images_path = os.path.join(
            path, f'{name}/raw/%s-images-idx3-ubyte.gz' % self.kind)
        with gzip.open(labels_path, 'rb') as lbpath:
            labels = np.frombuffer(
                lbpath.read(), dtype=np.uint8, offset=8).reshape(-1)
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(
                imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
        self.labels = th.from_numpy(np.copy(labels))
        self.labelset = set(list(labels))
        self.images = th.from_numpy(
            np.copy(images)).view(-1, 1, 28, 28) / 255.0
        self.cls2idx = defaultdict(list)
        for (i, lb) in enumerate(self.labels):
            self.cls2idx[lb.item()].append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int):
        if index >= len(self.labels):
            raise IndexError
        anchor = self.images[index]
        label = self.labels[index]
        if self.kind == 'train':
            posidx = random.choice(list(self.cls2idx[label.item()]))
            posimg = self.images[posidx]
            assert(label.item() == self.labels[posidx].item())
            return th.stack([anchor, posimg]), th.stack([label, label])
        else:
            return anchor, label


def _get_triplet_dataset():
    train = _MNIST_TRIPLET(configs.mnist.path, train=True)
    test = _MNIST_TRIPLET(configs.mnist.path, train=False)
    return (train, test)
