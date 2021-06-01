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
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import pytorch_lightning as thl
from pytorch_lightning.utilities.enums import DistributedType
import os
import re
import pytorch_lightning.metrics.functional
import torch.nn.functional as F
from .. import datasets
from .. import configs
from .. import utils
import multiprocessing as mp
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as __nmi
from ..attacks import AdvRank
from .svdreg import svdreg
from tqdm import tqdm
import functools
from .. import losses
#
try:
    import faiss
    faiss.omp_set_num_threads(4)
except ImportError:
    from sklearn.cluster import KMeans
#
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    pass
#
import rich
c = rich.get_console()


class ClassifierTemplate(object):

    def setup(self, stage=None):
        train, val, test = getattr(datasets, self.dataset).getDataset()
        if 'cifar10' == self.dataset:
            self.data_train = train
            self.data_val = test
        elif any(x == self.dataset for x in ('mnist', 'fashion')):
            self.data_train = train
            self.data_val = val
            self.data_test = test
        else:
            raise NotImplementedError

    def train_dataloader(self):
        train_loader = DataLoader(self.data_train,
                                  batch_size=getattr(
                                      configs, self.BACKBONE).batchsize,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=getattr(configs, self.BACKBONE).loader_num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.data_val,
                                batch_size=getattr(
                                    configs, self.BACKBONE).batchsize,
                                pin_memory=True,
                                num_workers=getattr(configs, self.BACKBONE).loader_num_workers)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.data_test,
                                 batch_size=getattr(
                                     configs, self.BACKBONE).batchsize,
                                 pin_memory=True,
                                 num_workers=getattr(configs, self.BACKBONE).loader_num_workers)
        return test_loader

    def configure_optimizers(self):
        if self.BACKBONE == 'csres18' and self.dataset == 'cifar10':
            optim = SGD(self.parameters(),
                        lr=self.config.lr, momentum=self.config.momentum,
                        weight_decay=configs.csres18.weight_decay)
        else:
            optim = Adam(self.parameters(),
                         lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.config, 'milestones'):
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim,
                                                          milestones=configs.csres18.milestones, gamma=0.1)
            return [optim], [scheduler]
        return optim

    def forward(self, x):
        if 'cifar10' == self.dataset:
            with th.no_grad():
                x = utils.renorm(x)
            x = self.backbone(x)
        else:
            x = self.backbone(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Train/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Train/accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Validation/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Validation/accuracy', accuracy)
        return {'loss': loss.item(), 'accuracy': accuracy}

    def validation_epoch_end(self, outputs: list):
        if str(self._distrib_type) in (
                'DistributedType.DDP', 'DistributedType.DDP2'):
            if th.distributed.get_rank() != 0:
                return
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
        c.print(f'[yellow]\nValidation |  loss= {summary["loss"]:.5f} '
                + f'accuracy= {summary["accuracy"]:.5f}')

    def test_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Test/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Test/accuracy', accuracy)
