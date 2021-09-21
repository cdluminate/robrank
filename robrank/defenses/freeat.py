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


def freeat_common_post_init_hook(model):
    '''
    Every model that uses FAT should call this in its post_init_hook() method.
    '''
    model.automatic_optimization = False
    model.num_repeats = 4
    model.config.maxepoch_orig = model.config.maxepoch
    model.config.maxepoch = model.config.maxepoch // model.num_repeats
    c.print(f'[bold cyan]I: lowering number of training epoch from\
            {model.config.maxepoch_orig} to {model.config.maxepoch}\
            due to FAT num_repeats={model.num_repeats}[/bold cyan]')


def none_freeat_step(model, batch, batch_idx, *, dryrun: bool = True):
    '''
    "Adversarial Training for Free!"
    An isolated training_step(...) method for pytorch lightning module.
    This function is named "none" because it's the dryrun version
    for debugging purpose. It executes the algorithm of FAT, but will
    reset the perturbation sigma to zero with dryrun toggled.

    This function has some additional requirements on the pytorch lightning
    model. See the "sanity check" part below for detail.

    # optimization template from pytorch lightning
    >>> opt = model.optimizers()
    >>> opt.zero_grad()
    >>> loss = self.compute_loss(batch)  # pseudo compute_loss
    >>> self.manual_backward(loss)  # instead of loss.backward()
    >>> opt.step()
    '''
    # sanity check
    assert(model.automatic_optimization == False)
    assert(hasattr(model, 'num_repeats'))
    assert(hasattr(model.config, 'maxepoch_orig'))
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
        loss_orig = model.lossfunc(output_orig, labels)
        # logging
        model.log('Train/loss_orig', loss_orig.item())
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
            metric=model.lossfunc._metric,
            margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E',)
                else configs.triplet.margin_cosine)
    anc, pos, neg = triplets

    # prepare the longlasting perturbation (sigma)
    if not getattr(model, 'sigma', False):
        model.sigma = th.zeros_like(batch[0]).cuda()
    if len(model.sigma) > len(images):
        sigma = model.sigma[:len(images), :, :, :]
    elif len(model.sigma) == len(images):
        sigma = model.sigma
    else: # len(sigma) < len(images)
        N, C, H, W = images.shape
        model.sigma = th.stack([model.sigma, th.zeros(N-len(model.sigma), C, H, W).cuda()])
        sigma = model.sigma
    #c.print(sigma.shape)

    # training loop
    model.train()
    sigma.requires_grad = True
    optx = th.optim.SGD([sigma], lr=1.)
    opt = model.optimizers()
    for i in range(model.num_repeats):
        # create adversarial example
        images_ptb = (images + sigma).clamp(0., 1.)
        # forward adversarial example
        emb = model.forward(images)
        if model.lossfunc._metric in ('C', 'N'):
            emb = F.normalize(emb)

        # [ Update Model Parameter ]
        # zero grad and get loss
        opt.zero_grad()
        loss = model.lossfunc.raw(emb[anc, :, :, :],
                emb[pos, :, :, :], emb[neg, :, :, :]).mean()
        # manually backward and update
        # then we will have grad of Loss wrt the model parameters and sigma
        model.manual_backward(loss)
        opt.step()

        # [ Update Perturbation sigma ]
        if model.configs.advtrain_pgditer > 1:
            sigma.grad.data.copy_(-model.configs.advtrain_alpha *
                    th.sign(sigma.grad))
        elif model.configs.advtrain_pgditer == 1:
            sigma.grad.data.copy_(-model.configs.advtrain_eps *
                    th.sign(sigma.grad))
        else:
            raise ValueError('illegal value for advtrain_pgditer')
        optx.step()
        sigma.clamp_(-model.configs.advtrain_eps,
                model.configs.advtrain_eps)

        # [NOOP] the perturbation and let it be a dryrun
        sigma.zero_()

    # we don't return anything in manual optimization mode
    return None


def est_freeat_step(model, batch, batch_idx):
    raise NotImplementedError


def act_freeat_step(model, batch, batch_idx):
    raise NotImplementedError


def amd_freeat_step(model, batch, batch_idx):
    raise NotImplementedError


def amdsemi_freeat_step(model, batch, batch_idx):
    raise NotImplementedError
