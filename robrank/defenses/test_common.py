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
from ..losses import ptripletN
from ..configs import rc2f2
import pytorch_lightning as thl
import pytest

class TestNet(thl.LightningModule):
    __test__ = False  # surpress pytest warning
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc = th.nn.Linear(784, 5)
        self.dataset = 'mnist'
        self.loss = 'ptripletN'
        self.lossfunc = ptripletN()
        self.is_advtrain_pnp = True
        self.config = rc2f2(self.dataset, self.loss)
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
    def forward(self, x):
        x = th.nn.Flatten()(x)
        return self.fc(x)


def test_testnet():
    model = TestNet()
    images = th.rand(10, 1, 28, 28)
    output = model.forward(images)
    loss = output.mean()
    loss.backward()


@pytest.mark.skip(reason='this is test helper')
def test_xxx_training_step(training_step: callable):
    model = TestNet()
    images = th.rand(10, 1, 28, 28)
    labels = th.stack([th.arange(5), th.arange(5)]).T.flatten()
    loss = training_step(model, (images, labels), 0)
    loss.backward()
