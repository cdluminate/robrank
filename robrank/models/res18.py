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
from .. import configs
from .. import datasets
from .. import losses
from .. import utils
from .template import MetricTemplate224
from termcolor import cprint, colored
from tqdm import tqdm
import functools
import json
import multiprocessing as mp
import numpy as np
import os
import re
import rich
import pytorch_lightning as thl
import pytorch_lightning.metrics.functional
import statistics
import torch as th
import torchvision as vision
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    pass
c = rich.get_console()


class Model(MetricTemplate224, thl.LightningModule):
    '''
    ResNet-18 / ResNet-50 For Ranking (2002.08473)

    Overload the RESNET vairable to switch to resnet50, or even
    the efficientnet.
    '''
    RESNET = 'resnet18'
    is_advtrain = False
    do_svd = False

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # configuration
        if self.RESNET == 'resnet18':
            self.config = configs.res18(dataset, loss)
            self.resnet = vision.models.resnet18(pretrained=True)
        elif self.RESNET == 'resnet50':
            self.config = configs.res50(dataset, loss)
            self.resnet = vision.models.resnet50(pretrained=True)
        elif self.RESNET == 'resnet101':
            self.config = configs.res50(dataset, loss)
            self.resnet = vision.models.resnet101(pretrained=True)
        elif self.RESNET == 'resnet152':
            self.config = configs.res50(dataset, loss)
            self.resnet = vision.models.resnet152(pretrained=True)
        elif self.RESNET == 'mnas05':
            self.config = configs.mnas(dataset, loss)
            self.resnet = vision.models.mnasnet0_5(pretrained=True)
        elif self.RESNET == 'mnas10':
            self.config = configs.mnas(dataset, loss)
            self.resnet = vision.models.mnasnet1_0(pretrained=True)
        elif self.RESNET == 'mnas13':
            self.config = configs.mnas(dataset, loss)
            self.resnet = vision.models.mnasnet1_3(pretrained=True)
        elif self.RESNET == 'efficientnet-b0':
            self.config = configs.enb0(dataset, loss)
            self.resnet = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.RESNET == 'efficientnet-b4':
            self.config = configs.enb4(dataset, loss)
            self.resnet = EfficientNet.from_pretrained('efficientnet-b4')
        elif self.RESNET == 'efficientnet-b7':
            self.config = configs.enb7(dataset, loss)
            self.resnet = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            raise ValueError()
        assert(dataset in self.config.allowed_datasets)
        self.dataset = dataset
        assert(loss in self.config.allowed_losses)
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        # modules
        if re.match(r'resnet', self.RESNET) and self.config.embedding_dim > 0:
            if '18' in self.RESNET:
                emb_dim = 512
            elif '50' in self.RESNET:
                emb_dim = 2048
            else:
                emb_dim = 2048
            self.resnet.fc = th.nn.Linear(emb_dim, self.config.embedding_dim)
        elif re.match(r'resnet', self.RESNET):
            self.resnet.fc = th.nn.Identity()
        elif re.match(r'mnas', self.RESNET) and self.config.embedding_dim > 0:
            self.resnet.classifier = th.nn.Linear(
                1280, self.config.embedding_dim)
        elif re.match(r'mnas', self.RESNET):
            self.resnet.classifier = th.nn.Identity()
        elif re.match(r'efficientnet', self.RESNET) and self.config.embedding_dim > 0:
            if 'b0' in self.RESNET:
                emb_dim = 1280
            elif 'b7' in self.RESNET:
                emb_dim = 2560
            self.resnet._modules['_dropout'] = th.nn.Identity()
            self.resnet._modules['_fc'] = th.nn.Linear(
                emb_dim, self.config.embedding_dim)
            # self.resnet._modules['_swish'] = th.nn.Identity() # XXX: don't
            # override swish.
        elif re.match(r'efficientnet', self.RESNET):
            self.resnet._modules['_dropout'] = th.nn.Identity()
            self.resnet._modules['_fc'] = th.nn.Identity()
            # self.resnet._modules['_swish'] = th.nn.Identity() # XXX: don't
            # override swish.
        # Freeze BatchNorm2d
        if self.config.freeze_bn and not self.is_advtrain:
            def __freeze(mod):
                if isinstance(mod, th.nn.BatchNorm2d):
                    mod.eval()
                    mod.train = lambda _: None
            # self.resnet.apply(__freeze)
            for mod in self.resnet.modules():
                __freeze(mod)
        # validation
        self._valvecs = None
        self._vallabs = None
        # adversarial attack
        self.wantsgrad = False
        # Dump configurations
        c.print('[green]Model Meta Information[/green]', {
            'dataset': self.dataset,
            'datasestspec': self.datasetspec,
            'lossfunc': self.lossfunc,
            'metric': self.metric,
            'config': {k: v for (k, v) in self.config.__dict__.items()
                       if k not in ('allowed_losses', 'allowed_datasets')},
        })

    def forward(self, x):
        if self.wantsgrad:
            return self.forward_wantsgrad(x)
        x = x.view(-1, 3, 224, 224)  # incase of datasetspec in ('p', 't')
        with th.no_grad():
            x = utils.renorm(x)
        x = self.resnet(x)
        return x

    def forward_wantsgrad(self, x):
        x = utils.renorm(x.view(-1, 3, 224, 224))
        x = self.resnet(x)
        return x

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self.resnet.parameters(),
            lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.config, 'milestones'):
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim,
                                                          milestones=self.config.milestones, gamma=0.1)
            return [optim], [scheduler]
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
