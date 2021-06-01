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
import pytorch_lightning.metrics.functional
from .. import datasets
from .. import configs
import multiprocessing as mp
import statistics
from .template_classify import ClassifierTemplate


class C2F2(th.nn.Module):
    '''
    A 2-Conv Layer 2-FC Layer Network for Classification
    '''

    def __init__(self):
        super(C2F2, self).__init__()
        self.conv1 = th.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = th.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = th.nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = th.nn.Linear(1024, 10)

    def forward(self, x):
        __relu = th.nn.functional.relu
        __pool = th.nn.functional.max_pool2d
        x = __relu(self.conv1(x))  # -1, 32, 28, 28
        x = __pool(x, kernel_size=2, stride=2)  # -1, 32, 14, 14
        x = __relu(self.conv2(x))  # -1, 64, 14, 14
        x = __pool(x, kernel_size=2, stride=2)  # -1, 64, 7, 7
        x = x.view(-1, 64 * 7 * 7)
        x = __relu(self.fc1(x))  # -1, 1024
        x = th.nn.functional.dropout(x, p=0.4, training=self.training)
        x = self.fc2(x)
        return x


class Model(ClassifierTemplate, thl.LightningModule):
    BACKBONE = 'cc2f2'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.cc2f2.allowed_datasets)
        assert(loss in configs.cc2f2.allowed_losses)
        # config
        self.dataset = dataset
        self.loss = loss
        self.config = getattr(configs, self.BACKBONE)(dataset, loss)
        # backbone
        self.backbone = C2F2()
