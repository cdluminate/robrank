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
import os
import torch as th
import torchvision as vision
from PIL import Image
import random
from collections import defaultdict
from .. import configs
from scipy.io import loadmat


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return __get_classification_dataset()
    elif kind == 'SPC-2':
        return __get_pair_dataset()
    else:
        raise NotImplementedError


def __get_classification_dataset():
    '''
    Load Cars196 Dataset. Classification version.
    '''
    train = CarsDataset('train')
    test = CarsDataset('test')
    return (train, test)


def __get_pair_dataset():
    train = CarsPairDataset('train')
    test = CarsDataset('test')
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


class CarsDataset(th.utils.data.Dataset):
    '''
    The Cars196 Dataset
    https://ai.stanford.edu/~jkrause/cars/car_dataset.html
    # The following two files are used
    http://imagenet.stanford.edu/internal/car196/car_ims.tgz
    http://imagenet.stanford.edu/internal/car196/cars_annos.mat
    '''

    def __init__(self, kind: str, *, zeroshot=True):
        self.kind = kind
        self.basepath = configs.cars.path
        self.transform = getTransform(self.kind)
        # get image list
        annos = loadmat(os.path.join(self.basepath, 'cars_annos.mat'),
                        squeeze_me=True)['annotations']
        imlbs = []
        for entry in annos:
            label = int(entry[5])
            istrain = bool(entry[6])
            if zeroshot:
                istrain = True if (label <= 98) else False
            if (istrain and kind == 'test') or (
                    not istrain and kind == 'train'):
                continue
            imlbs.append((
                os.path.join(self.basepath, entry[0]), label))
        self.imlbs = sorted(imlbs, key=lambda x: x[0])
        print(f'Cars-196[{self.kind}]: Got {len(self.imlbs)} Images.')

    def __len__(self):
        return len(self.imlbs)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        index = index.item() if isinstance(index, th.Tensor) else index
        image_path, label = self.imlbs[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label


class CarsPairDataset(th.utils.data.Dataset):
    '''
    Produce Data pairs [ cls1, cls1, cls2, cls2, cls3, cls3, etc... ]
    '''

    def __init__(self, kind: str):
        self.data = CarsDataset(kind)
        self.label2idxs = defaultdict(list)
        for (seqidx, (impath, lb)) in enumerate(self.data.imlbs):
            self.label2idxs[lb].append(seqidx)

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
