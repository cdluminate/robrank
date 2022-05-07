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
# A. Madry Defense (AMD) adaptation for deep metric learning.
# It is originally designed for classification problems (cross-entropy loss).
# Here we adopt it for deep metric learning.
###############################################################################


from typing import Tuple
import re
import os
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
    Here we are in charge of the inner maximization problem, and provide
    the corresponding adversarial examples.
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
                 stopat: float = None):
        # where we stop attacking.
        if stopat is None:
            #stopat = 0.2
            # stopat = 0.2 * (model._amdsemi_last_state / 2))
            # stopat = max(min(model._amdsemi_last_state, 0.2), 0.0))
            stopat = np.sqrt(
                max(min(self.model._amdsemi_last_state, 0.2), 0.0) / 0.2) * 0.2
            # stopat = (1 - np.exp(-10.0 * max(min(model._amdsemi_last_state, 0.2), 0.0)/0.2))*0.2)
        # prepare
        anc, pos, neg = triplets
        imanc = images[anc, :, :, :].clone().detach().to(self.device)
        impos = images[pos, :, :, :].clone().detach().to(self.device)
        imneg = images[neg, :, :, :].clone().detach().to(self.device)
        images_orig = th.cat([imanc, impos, imneg]).clone().detach()
        images = th.cat([imanc, impos, imneg])
        images.requires_grad = True
        # start PGD
        state_for_saturate_stop: float = -1.0
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
                loss = (F.pairwise_distance(ea, en) -
                        F.pairwise_distance(ea, ep)).clamp(
                    min=stopat).mean()
            elif self.metric in ('C',):
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
            # stop when saturate in order to save time.
            if abs(state_for_saturate_stop - loss.item()) < 1e-7:
                break
            else:
                state_for_saturate_stop = loss.item()
        # note: it is very important to clear the junk gradients.
        optm.zero_grad()
        optx.zero_grad()
        images.requires_grad = False
        return images

    def HardnessManipulate(self,
                           images: th.Tensor,
                           output_orig: th.Tensor,
                           labels: th.Tensor,
                           sourcehardness: str,
                           destinationhardness: str,
                           *,
                           method: str = 'KL',
                           gradual: bool = False,
                           return_triplets: bool = False,
                           fix_anchor: bool = False):
        '''
        Hardness manipulation from source hardness to destination hardness.
        This is specific to triplet input.
        Method in {KL, L2, ET} for hardness alignment.
        Gradual Adversary is a bool argument.
        Fix-Anchor is a boolean argument.
        '''
        # Sanity check
        if self.model.metric != 'N':
            raise NotImplementedError("currently only impleted for N metric")
        if gradual:
            if not hasattr(self.model, '_hm_prev_loss'):
                raise AttributeError("HM/GradualAdversary not properly initialized")
        if fix_anchor:
            raise NotImplementedError
        # sample the source and destination triplets
        src_triplets = miner(output_orig, labels, method=sourcehardness,
                             metric=self.model.lossfunc._metric,
                             margin=configs.triplet.margin_euclidean
                             if self.model.lossfunc._metric in ('E',)
                             else configs.triplet.margin_cosine)
        sanc, spos, sneg = src_triplets
        dest_triplets = miner(output_orig, labels, method=destinationhardness,
                              metric=self.model.lossfunc._metric,
                              margin=configs.triplet.margin_euclidean
                              if self.model.lossfunc._metric in ('E',)
                              else configs.triplet.margin_cosine)
        danc, dpos, dneg = dest_triplets
        # calculate destination loss vector
        if self.model.metric in ('E', 'N'):
            def _d(x, y):
                return F.pairwise_distance(x, y)
        else:
            def _d(x, y):
                return 1 - F.cosine_similarity(x, y)
        with th.no_grad():
            # destH is a vector.
            destH = (_d(output_orig[danc, :], output_orig[dpos, :]) - \
                _d(output_orig[danc, :], output_orig[dneg, :])).view(-1)
            # gradually increase hardness
            if gradual:
                # parameters
                ul, uh = configs.triplet.margin_cosine, 0.1
                nrmloss = th.tensor(self.model._hm_prev_loss).clamp(
                        min=0.0, max=ul)/ul  # in [0,1]
                # switch
                _G = 4
                if _G == 0:
                    # linear addition (increase E[H] by uh)
                    inc = (1.0 - nrmloss) * uh  # in [0,uh]
                    destH = destH + inc
                elif _G == 1:
                    # linear addition with padding
                    pad = destH.clamp(max=0.0).abs()
                    inc = (1.0 - nrmloss) * (uh + pad)
                    destH = destH + inc
                elif _G == 2:
                    destH = destH.clamp(min=-ul) # at least provide grads.
                elif _G == 3:
                    # also scale max(0, HD-HS)
                    destH = ((destH - srcH).clamp(min=0.) + uh) * (1-nrmloss)
                elif _G == 4:
                    # pure GA (semihard)
                    # destH = -0.2 + 0.2 * (1-nrmloss)
                    destH = -0.2 * nrmloss
                else:
                    raise ValueError
                # least square approaching (min |E[H]-uh|)
                #inc = (1-(th.tensor(self.model._hm_prev_loss).clamp(min=0.0,
                #        max=2+configs.triplet.margin_cosine)/(2+configs.triplet.margin_cosine))
                #       )*(uh - destH).clamp(min=0.0)
                # destH = destH + inc
            if method == 'KL':
                #destH = F.softmax(destH, dim=0)
                destH = F.normalize(destH, p=1, dim=0)
            elif method == 'L2':
                pass
            elif method == 'ET':
                pass
            else:
                raise NotImplementedError
        # prepare the template for adversarial examples
        imanc = images[sanc, :, :, :].clone().detach().to(self.device)
        impos = images[spos, :, :, :].clone().detach().to(self.device)
        imneg = images[sneg, :, :, :].clone().detach().to(self.device)
        imgs_orig = th.cat([imanc, impos, imneg]).clone().detach()
        imgs = th.cat([imanc, impos, imneg])
        imgs.requires_grad = True
        # start creating adversarial examples
        state_for_saturate_stop: float = -1.0
        self.model.eval()
        for iteration in range(self.pgditer):
            # optimizer
            optm = th.optim.SGD(self.model.parameters(), lr=0.)
            optx = th.optim.SGD([imgs], lr=1.)
            optm.zero_grad()
            optx.zero_grad()
            # Do nothing if the destination equals source
            if sourcehardness == destinationhardness:
                break
            # forward and get the embeddings
            emb = self.model.forward(imgs)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # compute the source loss vector
            srcH = (_d(ea, ep) - _d(ea, en)).view(-1)
            if method == 'KL':
                # srcH = F.softmax(srcH, dim=0)
                srcH = F.normalize(srcH, p=1, dim=0)
            else:
                # L2, ET, no operation
                pass
            # compute the loss of loss for attack (meta adversarial attack?)
            if method == 'KL':
                loss = F.kl_div(srcH, destH, reduction='mean')
            elif method == 'L2':
                loss = F.mse_loss(srcH, destH, reduction='mean')
            elif method == 'ET':
                # loss = th.min(th.tensor(0).to(srcH.device), srcH - destH).square().mean()
                loss = (srcH - destH).clamp(max=0.).square().mean()
            else:
                raise NotImplementedError
            itermsg = {'destH': destH.sum().item(),
                    'srcH': srcH.sum().item(),
                    'metaloss': loss.item()}
            # projected gradient descent
            loss.backward()
            if self.pgditer > 1:
                imgs.grad.data.copy_(self.alpha * th.sign(imgs.grad))
            elif self.pgditer == 1:
                imgs.grad.data.copy_(self.eps * th.sign(imgs.grad))
            optx.step()
            imgs = th.min(imgs, imgs_orig + self.eps)
            imgs = th.max(imgs, imgs_orig - self.eps)
            imgs = th.clamp(imgs, min=0., max=1.)
            imgs = imgs.clone().detach()
            imgs.requires_grad = True
            # report for the current step
            if self.verbose:
                print(images.shape)
            # stop when saturate in order to save time.
            if abs(state_for_saturate_stop - loss.item()) < 1e-7:
                break
            else:
                state_for_saturate_stop = loss.item()
        # note: clear the junk gradients or it interferes with model training.
        optm.zero_grad()
        optx.zero_grad()
        imgs.requires_grad = False
        if return_triplets:
            return imgs, src_triplets
        return imgs

    def HardnessManipulate_alt(self,
                           images: th.Tensor,
                           output_orig: th.Tensor,
                           labels: th.Tensor,
                           sourcehardness: str,
                           destinationhardness: str,
                           *,
                           method: str = 'KL',
                           gradual: bool = False,
                           return_triplets: bool = False,
                           fix_anchor: bool = False):
        '''
        alternative implementation of HardnessManipulate(...)
        drops some historical burden but requires relatively new pytorch
        avoids optimizer trick.
        '''
        # Sanity check
        if self.model.metric != 'N':
            raise NotImplementedError("currently only impleted for N metric")
        if gradual:
            if not hasattr(self.model, '_hm_prev_loss'):
                raise AttributeError("HM/GradualAdversary not properly initialized")
        if fix_anchor:
            raise NotImplementedError
        # sample the source and destination triplets
        src_triplets = miner(output_orig, labels, method=sourcehardness,
                             metric=self.model.lossfunc._metric,
                             margin=configs.triplet.margin_euclidean
                             if self.model.lossfunc._metric in ('E',)
                             else configs.triplet.margin_cosine)
        sanc, spos, sneg = src_triplets
        dest_triplets = miner(output_orig, labels, method=destinationhardness,
                              metric=self.model.lossfunc._metric,
                              margin=configs.triplet.margin_euclidean
                              if self.model.lossfunc._metric in ('E',)
                              else configs.triplet.margin_cosine)
        danc, dpos, dneg = dest_triplets
        # calculate destination loss vector
        if self.model.metric in ('E', 'N'):
            def _d(x, y):
                return F.pairwise_distance(x, y)
        else:
            def _d(x, y):
                return 1 - F.cosine_similarity(x, y)
        with th.no_grad():
            # destH is a vector.
            destH = (_d(output_orig[danc, :], output_orig[dpos, :]) - \
                _d(output_orig[danc, :], output_orig[dneg, :])).view(-1)
            # gradually increase hardness
            if gradual:
                # parameters
                ul, uh = configs.triplet.margin_cosine, 0.1
                nrmloss = th.tensor(self.model._hm_prev_loss).clamp(
                        min=0.0, max=ul)/ul  # in [0,1]
                # switch
                _G = 4
                if _G == 4:
                    # pure GA (semihard)
                    # destH = -0.2 + 0.2 * (1-nrmloss)
                    destH = -0.2 * nrmloss
                else:
                    raise ValueError
                # least square approaching (min |E[H]-uh|)
                #inc = (1-(th.tensor(self.model._hm_prev_loss).clamp(min=0.0,
                #        max=2+configs.triplet.margin_cosine)/(2+configs.triplet.margin_cosine))
                #       )*(uh - destH).clamp(min=0.0)
                # destH = destH + inc
            if method == 'KL':
                #destH = F.softmax(destH, dim=0)
                destH = F.normalize(destH, p=1, dim=0)
            elif method == 'L2':
                pass
            elif method == 'ET':
                pass
            else:
                raise NotImplementedError
        # prepare the template for adversarial examples
        imanc = images[sanc, :, :, :].clone().detach().to(self.device)
        impos = images[spos, :, :, :].clone().detach().to(self.device)
        imneg = images[sneg, :, :, :].clone().detach().to(self.device)
        imgs_orig = th.cat([imanc, impos, imneg]).clone().detach()
        imgs = th.cat([imanc, impos, imneg])
        # start creating adversarial examples
        state_for_saturate_stop: float = -1.0
        self.model.eval()
        for iteration in range(self.pgditer):
            # Do nothing if the destination equals source
            if sourcehardness == destinationhardness:
                break
            imgs.requires_grad = True
            # forward and get the embeddings
            emb = self.model.forward(imgs)
            if self.metric in ('C', 'N'):
                emb = F.normalize(emb)
            ea = emb[:len(emb) // 3]
            ep = emb[len(emb) // 3:2 * len(emb) // 3]
            en = emb[2 * len(emb) // 3:]
            # compute the source loss vector
            srcH = (_d(ea, ep) - _d(ea, en)).view(-1)
            if method == 'KL':
                # srcH = F.softmax(srcH, dim=0)
                srcH = F.normalize(srcH, p=1, dim=0)
            else:
                # L2, ET, no operation
                pass
            # compute the loss of loss for attack (meta adversarial attack?)
            if method == 'KL':
                loss = F.kl_div(srcH, destH, reduction='mean')
            elif method == 'L2':
                loss = F.mse_loss(srcH, destH, reduction='mean')
            elif method == 'ET':
                # loss = th.min(th.tensor(0).to(srcH.device), srcH - destH).square().mean()
                loss = (srcH - destH).clamp(max=0.).square().mean()
            else:
                raise NotImplementedError
            itermsg = {'destH': destH.sum().item(),
                    'srcH': srcH.sum().item(),
                    'metaloss': loss.item()}
            # projected gradient descent
            grad = th.autograd.grad(loss, imgs,
                    retain_graph=False, create_graph=False)[0]
            if self.pgditer > 1:
                imgs = imgs.detach() + self.alpha * grad.sign()
            elif self.pgditer == 1:
                imgs = imgs.detach() + self.eps * grad.sign()
            delta = th.clamp(imgs - imgs_orig, min=-self.eps, max=+self.eps)
            imgs = th.clamp(imgs + delta, min=0, max=1).detach()
            # report for the current step
            if self.verbose:
                print(images.shape)
            # stop when saturate in order to save time.
            if abs(state_for_saturate_stop - loss.item()) < 1e-7:
                break
            else:
                state_for_saturate_stop = loss.item()
        if return_triplets:
            return imgs, src_triplets
        return imgs


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


def amdsemi_training_step(model: th.nn.Module, batch, batch_idx, *, aap=False):
    '''
    adaptation of madry defense to triplet loss.
    we purturb (a, p, n).
    Manual prototype for HM. Nonstandard
    '''
    # specific to amdsemi
    if not hasattr(model, '_amdsemi_last_state'):
        # initialize this variable.
        model._amdsemi_last_state: float = 2.0
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
    if aap:
        output_orig = model.forward(images)
        if model.lossfunc._metric in ('C', 'N'):
            output_orig = F.normalize(output_orig)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    else:
        with th.no_grad():
            output_orig = model.forward(images)
            model.train()
            loss_orig = model.lossfunc(output_orig, labels)
    # generate adversarial examples
    triplets = miner(output_orig, labels, method=model.lossfunc._minermethod,
                     metric=model.lossfunc._metric,
                     margin=configs.triplet.margin_euclidean
                     if model.lossfunc._metric in ('E',)
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
    if aap:
        loss = loss + 1.0 * model.lossfunc.raw(
            output_orig[anc, :],
            pnemb[:len(pnemb) // 3],
            output_orig[pos, :],
            override_margin=0.0)
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    # specific to amdsemi
    model._amdsemi_last_state = loss.item()
    # return
    return loss


def hm_training_step(model: th.nn.Module, batch, batch_idx, *,
                     srch: str, desth: str, hm: str = 'KL',
                     gradual: bool = False, ics: bool = False,
                     fix_anchor: bool = False):
    '''
    Hardness manipulation.

    hm, hm with fixed anchor (rh)

    gradual {,g}hm

    hm in {KL, L2, ET}
    -> hmkl, hml2, hmet

    srch and desth in
    {spc2-random (r), spc2-semihard (m), spc2-softhard (s),
    spc2-distance (d), spc2-hard (h)}
    -> hm{kl,l2,et}{r,m,s,d,h}{r,m,s,d,h}

    ics (intra class structure)
    -> {,i} postfix

    This will override the hardness selection from the loss side.
    e.g., hmklrm:pmtripletN will override 'm' in pmtripletN into 'r'.
    '''
    # Sanity check
    assert(isinstance(fix_anchor, bool))
    assert(isinstance(gradual, bool))
    assert(isinstance(ics, bool))
    assert(fix_anchor == False)
    if gradual:
        if not hasattr(model, '_hm_prev_loss'):
            model._hm_prev_loss: float = 2.0
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
        if model.metric in ('C', 'N'):
            output_orig = F.normalize(output_orig)
        model.train()
        loss_orig = model.lossfunc(output_orig, labels)
    # create adversarial examples
    model.eval()
    model.wantsgrad = True
    amd = MadryInnerMax(model, eps=model.config.advtrain_eps,
                        alpha=model.config.advtrain_alpha,
                        pgditer=model.config.advtrain_pgditer,
                        device=model.device, metric=model.metric,
                        verbose=False)
    images_amd, triplets = amd.HardnessManipulate(images, output_orig, labels,
    #images_amd, triplets = amd.HardnessManipulate_alt(images, output_orig, labels,
                                        sourcehardness=srch,
                                        destinationhardness=desth,
                                        method=hm, gradual=gradual,
                                        fix_anchor=fix_anchor,
                                        return_triplets=True)
    anc, pos, neg = triplets
    # get embeddings of anchors.
    model.train()
    if ics:
        apidx = th.cat([anc, pos]).view(-1)
        ap_orig = model.forward(images[apidx, :, :, :])
        if model.metric in ('C', 'N'):
            ap_orig = F.normalize(ap_orig)
    # train with adversarial examples
    pnemb = model.forward(images_amd)
    if model.lossfunc._metric in ('C', 'N'):
        pnemb = F.normalize(pnemb)
    model.wantsgrad = False
    # compute adversarial loss
    loss = model.lossfunc.raw(
        pnemb[:len(pnemb) // 3],
        pnemb[len(pnemb) // 3:2 * len(pnemb) // 3],
        pnemb[2 * len(pnemb) // 3:]).mean()
    if gradual:
        model._hm_prev_loss = loss.item()
    if ics:
        icslambda = 0.5
        if os.path.exists('override_lambda_005'):
            icslambda = 0.05
        loss = loss + icslambda * model.lossfunc.raw(
                    ap_orig[:len(ap_orig)//2],
                    pnemb[:len(pnemb) // 3],
                    ap_orig[len(ap_orig)//2:],
                    override_margin=0.0)
    # logging
    model.log('Train/loss_orig', loss_orig.item())
    model.log('Train/loss_adv', loss.item())
    # return
    return loss
