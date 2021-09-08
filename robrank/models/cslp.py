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
from .. import datasets
from .. import configs
from .template_classify import ClassifierTemplate


class SLP(th.nn.Module):
    '''
    Single-Layer Perceptron (SLP)
    '''

    def __init__(self, output_size: int = 10):
        super(SLP, self).__init__()
        self.fc1 = th.nn.Linear(28 * 28, output_size)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        return x


class Model(ClassifierTemplate, thl.LightningModule):
    BACKBONE = 'cslp'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.cslp.allowed_datasets)
        assert(loss in configs.cslp.allowed_losses)
        # config
        self.dataset = dataset
        self.loss = loss
        self.config = getattr(configs, self.BACKBONE)(dataset, loss)
        # backbone
        self.backbone = SLP(output_size=getattr(configs, dataset).num_class)
