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

# pylint: disable=not-callable,no-member
import os
import torch as th
import torchvision as vision
from PIL import Image
import random
from collections import defaultdict
from .. import configs


def getDataset(kind: str = 'classification'):
    if kind == 'classification':
        return __get_classification_dataset()
    elif kind == 'triplet':
        return __get_triplet_dataset()
    elif kind == 'SPC-2':
        return __get_pair_dataset()
    else:
        raise NotImplementedError


def __get_classification_dataset():
    '''
    Load Stanford Online Products Dataset. Classification version.
    '''
    train = SOPDataset(configs.sop.path, configs.sop.list_train)
    test = SOPDataset(configs.sop.path, configs.sop.list_test)
    return (train, test)


def __get_triplet_dataset():
    '''
    Load Stanford Online Products Dataset. Triplet version.
    '''
    train = SOPTripletDataset(configs.sop.path, configs.sop.list_train)
    test = SOPDataset(configs.sop.path, configs.sop.list_test)
    return (train, test)


def __get_pair_dataset():
    train = SOPPairDataset(configs.sop.path, configs.sop.list_train)
    test = SOPDataset(configs.sop.path, configs.sop.list_test)
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
        transforms.append(vision.transforms.Resize((256, 256)))
        transforms.append(vision.transforms.CenterCrop((224, 224)))
        transforms.append(vision.transforms.ToTensor())
    else:
        raise ValueError(f'what is {kind} dataset?')
    return vision.transforms.Compose(transforms)


class SOPDataset(th.utils.data.Dataset):
    '''
    For loading a single image from the dataset.
    '''

    def __init__(self, basepath, listfile):
        self.basepath = basepath
        self.listfile = listfile
        self.kind = 'train' if 'train' in listfile else 'test'
        self.transform = getTransform(self.kind)
        with open(os.path.join(basepath, listfile), 'rt') as fp:
            lines = [x.split() for x in fp.readlines()[1:]]
        print(f'SOPDataset[{self.kind}]: Got {len(lines)} Images.')

        # build helper mappings
        self.ImageId2Path = {int(x[0]): str(x[3]) for x in lines}
        self.ImageId2FineClass = {int(x[0]): int(x[1]) for x in lines}
        self.ImageId2CoarseClass = {int(x[0]): int(x[2]) for x in lines}
        self.Class2ImageIds = defaultdict(list)
        for x in lines:
            self.Class2ImageIds[int(x[1])].append(int(x[0]))

    def __len__(self):
        return len(self.ImageId2Path)

    def __getitem__(self, index: int, *, byiid=False):
        '''
        Get image by ImageID
        '''
        if byiid is False and index >= len(self):
            raise IndexError
        if byiid and index not in self.ImageId2Path:
            raise IndexError
        idx_offset = 1 if self.kind == 'train' else 59552
        if byiid:
            idx_offset = 0
        if isinstance(index, th.Tensor):
            index = index.item()
        path = str(self.ImageId2Path[index + idx_offset])
        label = th.tensor(int(self.ImageId2FineClass[index + idx_offset]))
        impath = os.path.join(self.basepath, path)
        image = Image.open(impath).convert('RGB')
        image = self.transform(image)
        return image, label


class SOPTripletDataset(th.utils.data.Dataset):
    '''
    Produce Datapairs for Triplet
    '''

    def __init__(self, basepath, listfile):
        self.data = SOPDataset(basepath, listfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        anchor_im, anchor_label = self.data[index]
        # positive
        pid = random.choice(self.data.Class2ImageIds[anchor_label.item()])
        positive_im, positive_label = self.data.__getitem__(pid, byiid=True)
        # netgative
        nid = random.randint(0, len(self) - 1)
        negative_im, negative_label = self.data[nid]
        iids = th.LongTensor([index, pid, nid])
        images = th.stack([anchor_im, positive_im, negative_im])
        return images, iids


class SOPPairDataset(th.utils.data.Dataset):
    '''
    Produce Data pairs [ cls1, cls1, cls2, cls2, cls3, cls3, etc... ]
    '''

    def __init__(self, basepath, listfile):
        self.data = SOPDataset(basepath, listfile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        anchor_im, anchor_label = self.data[index]
        another_id = random.choice(
            self.data.Class2ImageIds[anchor_label.item()])
        another_im, another_label = self.data.__getitem__(
            another_id, byiid=True)
        images = th.stack([anchor_im, another_im])
        labels = th.stack([anchor_label, another_label])
        return (images, labels)
