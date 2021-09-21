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

from typing import Tuple
import re
import torch as th
import numpy as np
import torch.nn.functional as F
import rich
from ..losses.miner import miner
from ..attacks import AdvRank
from .. import configs
c = rich.get_console()


def none_freeat_step(model, batch, batch_idx):
    '''
    "Adversarial Training for Free!"
    An isolated training_step(...) method for pytorch lightning module.

    This function has some additional requirements on the pytorch lightning
    model. See the "sanity check" part below for detail.
    '''
    # sanity check
    assert(model.automatic_optimization == False)
    assert(hasattr(model, 'num_repeats'))
    assert(hasattr(model, 'maxepoch'))
    # preparation
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
    elif model.dataset in ('fashion', 'mnist'):
        images = images.view(-1, 1, 28, 28)
    else:
        raise ValueError('possibly illegal model.dataset?')
    # triplet sampling
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images)
        #loss_orig = model.lossfunc(output_orig, labels)
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
            metric=model.lossfunc._metric,
            margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E',)
                else configs.triplet.margin_cosine)
    raise NotImplementedError

    # prepare the longlasting perturbation (sigma)
    if not getattr(model, 'sigma', False):
        model.sigma = th.zeros_like(batch[0]).cuda()
    sigma = model.sigma

    # training loop
    model.train()
    sigma.requires_grad = True
    optm = th.optim.SGD(model.parameters(), lr=0.)
    optx = th.optim.SGD([sigma], lr=1.)
    for i in range(model.num_repeats):
        # create adversarial example
        images_ptb = (images + sigma).clamp(0., 1.)
        # forward adversarial example
        emb = model.forward(images)
        if model.lossfunc._metric in ('C', 'N'):
            emb = F.normalize(emb)

    # optimization template from pytorch lightning
    # opt = model.optimizers()
    # opt.zero_grad()
    # loss = self.compute_loss(batch)
    # self.manual_backward(loss)
    # opt.step()
