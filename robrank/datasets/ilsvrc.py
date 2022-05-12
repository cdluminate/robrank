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
import torchvision as V
from .. import configs


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return __get_classification_dataset()
    else:
        raise NotImplementedError


def __get_classification_dataset():
    '''
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    '''
    normalize = V.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train = V.datasets.ImageFolder(
        os.path.join(configs.ilsvrc.path, 'train'),
        V.transforms.Compose([
            V.transforms.RandomResizedCrop(224),
            V.transforms.RandomHorizontalFlip(),
            V.transforms.ToTensor(),
            normalize,
        ]))
    test = V.datasets.ImageFolder(
        os.path.join(configs.ilsvrc.path, 'val-symlink'),
        V.transforms.Compose([
            V.transforms.Resize(256),
            V.transforms.CenterCrop(224),
            V.transforms.ToTensor(),
            normalize,
        ]))
    return (train, None, test)
