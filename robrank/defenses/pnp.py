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


class PositiveNegativePerplexing(object):
    '''
    Attack designed for adversarial training
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

    def pncollapse(self, images: th.Tensor, triplets: tuple):
        '''
        collapse the positive and negative sample in the embedding space.
        (p->, <-n)
        '''
        # prepare
        anc, pos, neg = triplets
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([impos, imneg]).clone().detach()
        images = th.cat([impos, imneg])
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
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                loss = F.pairwise_distance(emb[:len(emb) // 2],
                                           emb[len(emb) // 2:]).mean()
            elif self.metric in ('C',):
                loss = 1 - F.cosine_similarity(emb[:len(emb) // 2],
                                               emb[len(emb) // 2:]).mean()
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def pnanchor(self, images: th.Tensor, triplets: tuple,
                 emb_anchor_detached: th.Tensor):
        '''
        (a, p->), (a, <-n), adversary to contrastive
        '''
        # prepare
        anc, pos, neg = triplets
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([impos, imneg]).clone().detach()
        images = th.cat([impos, imneg])
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
            # (a, p->), (a, <-n)
            if self.metric in ('E', 'N'):
                loss = F.pairwise_distance(emb_anchor_detached,
                                           emb[len(emb) // 2:])
                loss -= F.pairwise_distance(emb_anchor_detached,
                                            emb[:len(emb) // 2])
                loss = loss.mean()
            elif self.metric in ('C',):
                loss = 1 - F.cosine_similarity(emb_anchor_detached,
                                               emb[len(emb) // 2:])
                loss -= 1 - F.cosine_similarity(emb_anchor_detached,
                                                emb[:len(emb) // 2])
                loss = loss.mean()
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def apsplit(self, images: th.Tensor, triplets: tuple):
        '''
        maximize d(a, p)
        '''
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([imanc, impos]).clone().detach()
        images = th.cat([imanc, impos])
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
            # <-a, p->
            if self.metric in ('E', 'N'):
                loss = -F.pairwise_distance(emb[len(emb) // 2:],
                                            emb[:len(emb) // 2]).mean()
            elif self.metric in ('C',):
                loss = -1 + F.cosine_similarity(emb[len(emb) // 2:],
                                               emb[:len(emb) // 2]).mean()
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def tribodycollapse(self, images: th.Tensor, triplets: tuple):
        '''
        collapse (a, p, n) in the embedding space.
        '''
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
            # draw two samples close to each other
            if self.metric in ('E', 'N'):
                loss = (F.pairwise_distance(ep, en) +
                        F.pairwise_distance(ea, ep) +
                        F.pairwise_distance(ea, en)).mean()
            elif self.metric in ('C',):
                loss = (3 - F.cosine_similarity(ep, en)
                          - F.cosine_similarity(ea, ep)
                          - F.cosine_similarity(ea, en)).mean()
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def minmaxtriplet(self, images: th.Tensor, triplets: tuple):
        '''
        Direct adaptation of Madry defense for triplet loss.
        Maximize triplet -> max dap, min dan. Modify all.
        '''
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
            # draw two samples close to each other
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images

    def anticolanchperplex(self, images: th.Tensor, triplets: tuple):
        '''
        collapse (p, n) and perplex (a).
        '''
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
            # draw two samples close to each other and perplex the anchor
            if np.random.random() > 0.5:
                if self.metric in ('E', 'N'):
                    loss = F.pairwise_distance(ep, en).mean()
                elif self.metric in ('C',):
                    loss = (1 - F.cosine_similarity(ep, en)).mean()
            else:
                if self.metric in ('E', 'N'):
                    loss = (F.pairwise_distance(ea, en)
                            - F.pairwise_distance(ea, ep)).mean()
                elif self.metric in ('C',):
                    loss = ((1 - F.cosine_similarity(ea, en))
                            - (1 - F.cosine_similarity(ea, ep))).mean()
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
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        # return the adversarial example
        if self.verbose:
            print(images.shape)
        return images


def pnp_training_step(model: th.nn.Module, batch, batch_idx, *,
        pgditer: int = None):
    '''
    Adversarial training with Positive/Negative Perplexing (PNP) Attack.
    Function signature follows pytorch_lightning.LightningModule.training_step,
    where model is a lightning model, batch is a tuple consisting images
    (th.Tensor) and labels (th.Tensor), and batch_idx is just an integer.

    Collapsing positive and negative -- Anti-Collapse (ACO) defense.
    force the model to learning robust feature and prevent the
    adversary from exploiting the non-robust feature and collapsing
    the positive/negative samples again.

    This defense is not agnostic to backbone architecure and metric learning
    loss. But it is recommended to use it in conjunction with triplet loss.
    '''
    # check loss function
    if not re.match(r'p.?triplet.*', model.loss) and \
            not re.match(r'psnr.*', model.loss) and \
            not re.match(r'pmargin.*', model.loss) and \
            not re.match(r'pcontrast.*', model.loss):
        raise ValueError(f'ACT defense is not implemented for {model.loss}!')

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
    output_orig = model.forward(images)
    with th.no_grad():
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=model.config.advtrain_eps,
                                     alpha=model.config.advtrain_alpha,
                                     pgditer=model.config.advtrain_pgditer if pgditer is None else pgditer,
                                     device=model.device, metric=model.metric,
                                     verbose=False)
    # Collapsing positive and negative -- Anti-Collapse Triplet (ACT) defense.
    model.eval()
    model.wantsgrad = True
    if hasattr(model, 'is_advtrain_pnp_adapt') and model.is_advtrain_pnp_adapt:
        if re.match(r'ptriplet.*', model.loss):
            images_pnp = pnp.pncollapse(images, triplets)
        # adapt pnp/act for the specific loss function
        elif re.match(r'pcontrast.*', model.loss):
            assert(model.loss == 'pcontrastN')
            with th.no_grad():
                mask = F.pairwise_distance(output_orig[anc, :], output_orig[neg, :]) < configs.contrastive.margin_euclidean
                mask = mask.view(-1, 1)
            images_pnp = pnp.pncollapse(images, triplets)
            images_aps = pnp.apsplit(images, triplets)

            model.train()
            epnp = F.normalize(model.forward(images_pnp))
            eaps = F.normalize(model.forward(images_aps))
            model.wantsgrad = False
            N = len(anc)
            ep = th.where(mask, epnp[:N], eaps[:N])
            en = th.where(mask, epnp[N:], eaps[N:])
            model.train()
            loss = model.lossfunc.raw(output_orig[anc, :], ep, en).mean()
            #
            model.log('Train/loss_orig', loss_orig.item())
            model.log('Train/loss_adv', loss.item())
            return loss
        else:
            raise NotImplementedError(
                f'not implemeneted pnp/act for {model.loss}')
    else:
        # use pnp/act for triplet ignoring the loss type.
        images_pnp = pnp.pncollapse(images, triplets)
    # Adversarial Training
    model.train()
    pnemb = model.forward(images_pnp)
    if model.lossfunc._metric in ('C', 'N'):
        pnemb = F.normalize(pnemb)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(output_orig[anc, :],
                              pnemb[:len(pnemb) // 2], pnemb[len(pnemb) // 2:]).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss


def mmt_training_step(model: th.nn.Module, batch, batch_idx):
    '''
    min-max triplet
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
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=model.config.advtrain_eps,
                                     alpha=model.config.advtrain_alpha,
                                     pgditer=model.config.advtrain_pgditer,
                                     device=model.device, metric=model.metric,
                                     verbose=False)
    model.eval()
    model.wantsgrad = True
    images_pnp = pnp.minmaxtriplet(images, triplets)
    model.train()
    pnemb = model.forward(images_pnp)
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


def tbc_training_step(model: th.nn.Module, batch, batch_idx):
    '''
    Three body collapse. (a, p, n)

    moderate recall, a certain level of robustness.
    but the robustness level is insufficient.
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
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=model.config.advtrain_eps,
                                     alpha=model.config.advtrain_alpha,
                                     pgditer=model.config.advtrain_pgditer,
                                     device=model.device, metric=model.metric,
                                     verbose=False)
    model.eval()
    model.wantsgrad = True
    images_pnp = pnp.tribodycollapse(images, triplets)
    model.train()
    pnemb = model.forward(images_pnp)
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


def acap_training_step(model: th.nn.Module, batch, batch_idx):
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
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    model.eval()
    pnp = PositiveNegativePerplexing(model, eps=model.config.advtrain_eps,
                                     alpha=model.config.advtrain_alpha,
                                     pgditer=model.config.advtrain_pgditer,
                                     device=model.device, metric=model.metric,
                                     verbose=False)
    model.eval()
    model.wantsgrad = True
    images_pnp = pnp.anticolanchperplex(images, triplets)
    model.train()
    pnemb = model.forward(images_pnp)
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


def rest_training_step(model, batch, batch_idx):
    '''
    Revised EST
    we speculate that the misleading gradient (perturbed anchor) hinders
    the convergence of the ranking model. In this revised version, we pass
    in the not perturbed anchors.
    '''
    # prepare data batch in a proper shape
    images = batch[0].to(model.device)
    labels = batch[1].view(-1).to(model.device)
    if model.dataset in ('sop', 'cub', 'cars'):
        images = images.view(-1, 3, 224, 224)
        shape = (-1, 3, 224, 224)
    elif model.dataset in ('mnist', 'fashion'):
        images = images.view(-1, 1, 28, 28)
        shape = (-1, 1, 28, 28)
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
                     margin=configs.triplet.margin_euclidean if model.lossfunc._metric in ('E', 'N')
                     else configs.triplet.margin_cosine)
    anc, pos, neg = triplets
    advrank = AdvRank(model, eps=model.config.advtrain_eps,
            alpha=model.config.advtrain_alpha,
            pgditer=model.config.advtrain_pgditer,
            device=model.device,
            metric=model.metric, verbose=False)
    model.wantsgrad = True
    model.eval()
    advpn = advrank.embShift(th.stack([
        images[pos, :, :, :].view(*shape).clone().detach(),
        images[neg, :, :, :].view(*shape).clone().detach()]).view(*shape))
    batch = th.cat([
        images[anc, :, :, :].view(*shape), advpn], dim=0).view(*shape)
    model.train()
    embs = model.forward(batch)
    if model.lossfunc._metric in ('C', 'N'):
        embs = F.normalize(embs)
    model.wantsgrad = False
    # compute adversarial loss
    model.train()
    loss = model.lossfunc.raw(
        embs[:len(embs) // 3],
        embs[len(embs) // 3:2 * len(embs) // 3],
        embs[2 * len(embs) // 3:]).mean()
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    return loss
