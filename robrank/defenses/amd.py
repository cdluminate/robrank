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
###############################################################################
# defenses/amd.py
# A. Madry Defense (AMD)
# It is originally designed for classification problems (cross-entropy loss).
# Here we adopt it for deep metric learning.
###############################################################################


from typing import Tuple
import re
import torch as th
import numpy as np
import torch.nn.functional as F
import rich
from ..losses.miner import miner
from ..attacks import AdvRank
from .. import configs
from .pnp import PositiveNegativePerplexing
c = rich.get_console()


class MadryInnerMax(object):
    '''
    Madry defense adopted for deep metric learning.
    Here we are in charge of the inner maximization problem.
    '''

    def __init__(self,
                 model: th.nn.Module, eps: float, alpha: float, pgditer: int,
                 device: str, metric: str, verbose: bool = False):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.pgditer = pgditer
        self.device = device
        self.metric = metric
        self.verbose = verbose

    def ptbapn(self, images: th.Tensor, triplets: tuple):
        '''
        perturb (Anchor, Positive, Negative) for achieving the inner max.
        '''
        pnp = PositiveNegativePerplexing(self.model, self.eps, self.alpha,
                                         self.pgditer, self.device, self.metric, self.verbose)
        return pnp.minmaxtriplet(images, triplets)

    def ptbpn(self, images: th.Tensor, triplets: tuple):
        '''
        perturb (Positive, Negative) for achieving the inner max.
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([impos, imneg])
        images = th.cat([impos, imneg])
        images.requires_grad = True
        # Start PGD
        self.model.eval()
        ea = self.model.forward(imanc).clone().detach()
        for iteration in range(self.pgditer):
            # optimizer
            optm = th.optim.SGD(self.model.parameters(), lr=0.)
            optx = th.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ep = emb[:len(emb) // 2]
            en = emb[len(emb) // 2:]
            # maxinize the triplet
            if self.metric in ('E', 'N'):
                loss = (F.pairwise_distance(ea, en) -
                        F.pairwise_distance(ea, ep)).mean()
            elif self.metric in ('C',):
                loss = ((1 - F.cosine_similarity(ea, en)) -
                        (1 - F.cosine_similarity(ea, ep))).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * th.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad))
            optx.step()
            images = th.min(images, images_orig + self.eps)
            images = th.max(images, images_orig - self.eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(f'(PGD)>', itermsg)
        # note: it is very important to clear the junk gradients
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return th.cat([imanc, images])

    def advtstop(self, images: th.Tensor, triplets: tuple, *,
            stopat: float = 0.2):
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([imanc, impos, imneg]).clone().detach()
        images = th.cat([imanc, impos, imneg])
        images.requires_grad = True
        # start PGD
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = th.optim.SGD(self.model.parameters(), lr=0.)
            optx = th.optim.SGD([images], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # forward data to be perturbed
            emb = self.model.forward(images)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # compute the loss function
            if self.metric in ('E', 'N'):
                # [ non-stopping version ]
                #loss = (F.pairwise_distance(ea, en) -
                #        F.pairwise_distance(ea, ep)).mean()
                # [ stopping version ]
                loss = (F.pairwise_distance(ea, en) -
                        F.pairwise_distance(ea, ep)).clamp(
                                min=stopat).mean()
            elif self.metric in ('C',):
                # [ non-stopping version ]
                #loss = ((1 - F.cosine_similarity(ea, en)) -
                #        (1 - F.cosine_similarity(ea, ep))).mean()
                # [ stopping version ]
                loss = ((1 - F.cosine_similarity(ea, en)) -
                        (1 - F.cosine_similarity(ea, ep))).clamp(
                                min=stopat).mean()
            itermsg = {'loss': loss.item()}
            loss.backward()
            # projected gradient descent
            if self.pgditer > 1:
                images.grad.data.copy_(self.alpha * th.sign(images.grad))
            elif self.pgditer == 1:
                images.grad.data.copy_(self.eps * th.sign(images.grad))
            optx.step()
            images = th.min(images, images_orig + self.eps)
            images = th.max(images, images_orig - self.eps)
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
            # report for the current iteration
            if self.verbose:
                print(images.shape)
        return images


def amd_training_step(model: th.nn.Module, batch, batch_idx):
    '''
    adaptation of madry defense to triplet loss.
    we purturb (a, p, n).
    '''
    # prepare data batch in a proper shape
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
    elif model.dataset in ('mnist', 'fashion'):
        images = images.view(-1, 1, 28, 28)
    else:
        raise ValueError(f'possibly illegal dataset {model.dataset}?')
    # evaluate original benign sample
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean
                     if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    amd = MadryInnerMax(model, eps=model.config.advtrain_eps,
                        alpha=model.config.advtrain_alpha,
                        pgditer=model.config.advtrain_pgditer,
                        device=model.device, metric=model.metric,
                        verbose=False)
    model.eval()
    model.wantsgrad = True
    images_amd = amd.ptbapn(images, triplets)
    model.train()
    pnemb = model.forward(images_amd)
    if model.lossfunc._metric in ('C', 'N'):
        pnemb = F.normalize(pnemb)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(
        pnemb[:len(pnemb) // 3],
        pnemb[len(pnemb) // 3:2 * len(pnemb) // 3],
        pnemb[2 * len(pnemb) // 3:]).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss


def ramd_training_step(model: th.nn.Module, batch, batch_idx):
    '''
    revised AMD defense for DML
    we perturb (p, n)
    '''
    # prepare data batch
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
    elif model.dataset in ('mnist', 'fashion'):
        images = images.view(-1, 1, 28, 28)
    else:
        raise ValueError(f'possibly illegal dataset {model.dataset}')
    # evaluate original benign sample
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean
                     if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    amd = MadryInnerMax(model, eps=model.config.advtrain_eps,
                        alpha=model.config.advtrain_alpha,
                        pgditer=model.config.advtrain_pgditer,
                        device=model.device, metric=model.metric,
                        verbose=False)
    model.eval()
    model.wantsgrad = True
    images_amd = amd.ptbpn(images, triplets)
    model.train()
    apnemb = model.forward(images_amd)
    if model.lossfunc._metric in ('C', 'N'):
        apnemb = F.normalize(apnemb)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(
        apnemb[:len(apnemb) // 3],
        apnemb[len(apnemb) // 3:2 * len(apnemb) // 3],
        apnemb[2 * len(apnemb) // 3:]).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss


def amdsemi_training_step(model: th.nn.Module, batch, batch_idx):
    '''
    adaptation of madry defense to triplet loss.
    we purturb (a, p, n).
    '''
    # prepare data batch in a proper shape
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
    elif model.dataset in ('mnist', 'fashion'):
        images = images.view(-1, 1, 28, 28)
    else:
        raise ValueError(f'possibly illegal dataset {model.dataset}?')
    # evaluate original benign sample
    model.eval()
    with th.no_grad():
        output_orig = model.forward(images)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean
                     if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    amd = MadryInnerMax(model, eps=model.config.advtrain_eps,
                        alpha=model.config.advtrain_alpha,
                        pgditer=model.config.advtrain_pgditer,
                        device=model.device, metric=model.metric,
                        verbose=False)
    model.eval()
    model.wantsgrad = True
    images_amd = amd.advtstop(images, triplets)
    model.train()
    pnemb = model.forward(images_amd)
    if model.lossfunc._metric in ('C', 'N'):
        pnemb = F.normalize(pnemb)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(
        pnemb[:len(pnemb) // 3],
        pnemb[len(pnemb) // 3:2 * len(pnemb) // 3],
        pnemb[2 * len(pnemb) // 3:]).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss

