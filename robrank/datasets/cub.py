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

# pylint: disable=no-member,not-callable
import os
import torch as th
import torchvision as vision
from PIL import Image
import random
from collections import defaultdict
from .. import configs
import csv
import re


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return __get_classification_dataset()
    elif kind == 'SPC-2':
        return __get_pair_dataset()
    else:
        raise NotImplementedError


def __get_classification_dataset():
    '''
    Load Stanford Online Products Dataset. Classification version.
    '''
    train = CUBDataset('train')
    test = CUBDataset('test')
    return (train, test)


def __get_pair_dataset():
    train = CUBPairDataset('train')
    test = CUBDataset('test')
    return (train, test)


def getTransform(kind: str):
    '''
    training: (orig) -> resize (256,256) -> randcrop (224,224)
    testing: (orig) -> resize (256,256) -> centercrop (224,224)
    '''
    transforms = []
    if kind == 'train':
        transforms.append(vision.transforms.Resize((256, 256)))
        transforms.append(vision.transforms.RandomCrop((224, 224)))
        transforms.append(vision.transforms.RandomHorizontalFlip(p=0.5))
        transforms.append(vision.transforms.ToTensor())
    elif kind == 'test':
        # This is correct.
        transforms.append(vision.transforms.Resize((256, 256)))
        transforms.append(vision.transforms.CenterCrop((224, 224)))
        transforms.append(vision.transforms.ToTensor())
    else:
        raise ValueError(f'what is {kind} dataset?')
    return vision.transforms.Compose(transforms)


class CUBDataset(th.utils.data.Dataset):
    '''
    The Caltech-UCSD Birds-200-2011 Dataset
    '''

    def __init__(self, kind: str, *, zeroshot=True):
        self.kind = kind
        self.basepath = configs.cub.path
        self.transform = getTransform(self.kind)
        with open(os.path.join(self.basepath, 'images.txt'), 'rt') as f_images:
            idx2path = {int(idx): path for (idx, path)
                        in csv.reader(f_images, delimiter=' ')}
        with open(os.path.join(self.basepath, 'train_test_split.txt'), 'rt') as f_split:
            idx2split = {int(idx): 'train' if int(istrain) == 1 else 'test' for (
                idx, istrain) in csv.reader(f_split, delimiter=' ')}
        if zeroshot:
            # override idx2split and re-split the dataset following ICML20
            with open(os.path.join(self.basepath, 'image_class_labels.txt'),
                      'rt') as f_label:
                idx2split = {int(idx): 'train' if int(label) <= 100 else 'test'
                             for (idx, label) in csv.reader(f_label, delimiter=' ')}
        self.idx2path = {idx: os.path.join(self.basepath, 'images', path) for (
            idx, path) in idx2path.items() if idx2split[idx] == self.kind}
        self.idx2label = {idx: int(re.match(
            r'^(\d+)\..+', path).groups()[0]) - 1 for (idx, path) in idx2path.items()}
        self.indexes = tuple(sorted(self.idx2path.keys()))
        print(f'CUB-200-2011[{self.kind}]: Got {len(self.indexes)} Images.')

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        index = index.item() if isinstance(index, th.Tensor) else index
        idx = self.indexes[index]
        image_path, label = self.idx2path[idx], self.idx2label[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label


class CUBPairDataset(th.utils.data.Dataset):
    '''
    Produce Data pairs [ cls1, cls1, cls2, cls2, cls3, cls3, etc... ]
    '''

    def __init__(self, kind: str):
        self.data = CUBDataset(kind)
        self.label2idxs = defaultdict(list)
        for (seqidx, idx) in enumerate(self.data.indexes):
            label = self.data.idx2label[idx]
            self.label2idxs[label].append(seqidx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        anchor_im, anchor_label = self.data[index]
        another_seqidx = random.choice(self.label2idxs[anchor_label])
        another_im, another_label = self.data[another_seqidx]
        assert(anchor_label == another_label)
        images = th.stack([anchor_im, another_im])
        labels = th.tensor([anchor_label, another_label], dtype=th.long)
        return (images, labels)
