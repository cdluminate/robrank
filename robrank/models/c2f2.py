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
from termcolor import cprint
from .template import ClassifierTemplate


class Model(ClassifierTemplate, thl.LightningModule):
    '''
    A 2-Conv Layer 2-FC Layer Network for Classification
    '''

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.c2f2.allowed_datasets)
        assert(loss in configs.c2f2.allowed_losses)
        self.dataset = dataset
        # modules
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

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self.parameters(), lr=configs.c2f2.lr, weight_decay=configs.c2f2.weight_decay)
        return optim

    def setup(self, stage=None):
        train, val, test = getattr(datasets, self.dataset).getDataset()
        self.data_train = train
        self.data_val = val
        self.data_test = test

    def train_dataloader(self):
        train_loader = th.utils.data.DataLoader(self.data_train,
                                                batch_size=configs.c2f2.batchsize,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=configs.c2f2.loader_num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = th.utils.data.DataLoader(self.data_val,
                                              batch_size=configs.c2f2.batchsize,
                                              pin_memory=True,
                                              num_workers=configs.c2f2.loader_num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = th.utils.data.DataLoader(self.data_test,
                                               batch_size=configs.c2f2.batchsize,
                                               pin_memory=True,
                                               num_workers=configs.c2f2.loader_num_workers)
        return test_loader
