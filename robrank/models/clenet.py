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
import pytorch_lightning as thl
import os
from .. import datasets
from .. import configs
import multiprocessing as mp
import statistics
from .template_classify import ClassifierTemplate


class LeNet(th.nn.Module):
    """
    LeNet convolutional neural network for classification
    """

    def __init__(self):
        '''
        reference: Caffe-LeNet
        '''
        super(LeNet, self).__init__()
        self.conv1 = th.nn.Conv2d(1, 20, 5, stride=1)
        self.conv2 = th.nn.Conv2d(20, 50, 5, stride=1)
        self.fc1 = th.nn.Linear(800, 500)
        self.fc2 = th.nn.Linear(500, 10)

    def forward(self, x, *, l2norm=False):
        # -1, 1, 28, 28
        x = self.conv1(x)
        # -1, 20, 24, 24
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 20, 12, 12
        x = self.conv2(x)
        # -1, 50, 8, 8
        x = th.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        # -1, 50, 4, 4
        x = x.view(-1, 4 * 4 * 50)
        # -1, 800
        x = th.nn.functional.relu(self.fc1(x))
        # -1, 500
        x = self.fc2(x)
        return x


class Model(ClassifierTemplate, thl.LightningModule):
    '''
    A 2-Conv Layer 2-FC Layer Network for Classification
    '''
    BACKBONE = 'clenet'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.clenet.allowed_datasets)
        assert(loss in configs.clenet.allowed_losses)
        self.dataset = dataset
        self.loss = loss
        self.config = configs.clenet(dataset, loss)
        self.backbone = LeNet()
