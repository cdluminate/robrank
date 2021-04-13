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
import functools
from termcolor import cprint, colored
from tqdm import tqdm
import statistics
from .. import losses
from .template import MetricTemplate28
import json
import rich
c = rich.get_console()


class Model(MetricTemplate28, thl.LightningModule):
    '''
    A 2-Conv Layer 1-FC Layer Network For Ranking
    '''
    is_advtrain = False
    do_svd = False
    NET = 'c2f1'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.c2f1.allowed_datasets)
        assert(loss in configs.c2f1.allowed_losses)
        self.dataset = dataset
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        # configuration
        self.config = configs.c2f1(dataset, loss)
        # modules
        if self.NET == 'c2f1':
            self._model = th.nn.Sequential(
                th.nn.Conv2d(1, 32, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(64 * 7 * 7, 1024),
            )
        elif self.NET == 'rlenet':
            self._model = th.nn.Sequential(
                th.nn.Conv2d(1, 20, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(20, 50, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(800, 500),
                th.nn.ReLU(),
                th.nn.Linear(500, 128),
            )
        else:
            raise ValueError('unknown NET')
        # validation
        self._valvecs = None
        self._vallabs = None
        # summary
        c.print('[green]Model Meta Information[/green]', {
            'dataset': self.dataset,
            'datasestspec': self.datasetspec,
            'lossfunc': self.lossfunc,
            'metric': self.metric,
            'config': {k: v for (k, v) in self.config.__dict__.items()
                       if k not in ('allowed_losses', 'allowed_datasets')},
        })

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self._model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.lossfunc, 'getOptim'):
            optim2 = self.lossfunc.getOptim()
            return optim, optim2
        return optim

    def setup(self, stage=None):
        train, test = getattr(
            datasets, self.dataset).getDataset(self.datasetspec)
        self.data_train = train
        self.data_val = test

    def train_dataloader(self):
        train_loader = th.utils.data.DataLoader(self.data_train,
                                                batch_size=self.config.batchsize,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=self.config.loader_num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = th.utils.data.DataLoader(self.data_val,
                                              batch_size=self.config.valbatchsize,
                                              pin_memory=True,
                                              num_workers=self.config.loader_num_workers)
        return val_loader
