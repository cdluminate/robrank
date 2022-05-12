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

# pylint: disable=too-many-function-args
import os
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision as V
from .. import configs
from .cifar10 import unpickle
import pytest


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return __get_classification_dataset()
    else:
        raise NotImplementedError


def __get_classification_dataset():
    train = Cifar100Dataset(configs.cifar100.path, 'train')
    test = Cifar100Dataset(configs.cifar100.path, 'test')
    return (train, None, test)


class Cifar100Dataset(Dataset):
    '''
    the cifar 100 dataset
    '''

    def __init__(self, path: str, kind='train'):
        self.path = path
        self.transform = get_transform(kind)
        #
        file_train = os.path.join(path, 'train')
        file_test = os.path.join(path, 'test')
        file_meta = os.path.join(path, 'meta')
        #
        self.meta = unpickle(file_meta)
        if kind == 'train':
            data = unpickle(file_train)
            images = np.array(data['data']).reshape(-1, 3, 32, 32)
            labels = np.array(data['fine_labels'])
        elif kind == 'test':
            data = unpickle(file_test)
            images = np.array(data['data']).reshape(-1, 3, 32, 32)
            labels = np.array(data['fine_labels'])
        else:
            raise ValueError('unknown kind')
        self.images = images.transpose((0, 2, 3, 1))
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transform(kind: str = 'train'):
    """
    Load cifar10 data and turn them into dataloaders
    """
    if kind == 'train':
        transform = V.transforms.Compose([
            V.transforms.RandomCrop(32, padding=4),
            V.transforms.RandomHorizontalFlip(),
            V.transforms.ToTensor(),
        ])
    else:
        transform = V.transforms.Compose([
            V.transforms.ToTensor(),
        ])
    return transform


@pytest.mark.skipif(not os.path.exists(configs.cifar100.path),
                    reason='test data is not available')
@pytest.mark.parametrize('kind', ('classification',))
def test_cifar100_getdataset(kind: str):
    x = getDataset(kind=kind)
    if kind == 'classification':
        assert(all([len(x[0]) == 50000, len(x[2]) == 10000]))
