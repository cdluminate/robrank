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

from scipy import stats
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
from termcolor import cprint, colored
from tqdm import tqdm
import collections
import functools
import math
import numpy as np
import os, sys, re
import pylab as lab
import random
import statistics
import torch as th
import traceback
import json
from ..utils import IMmean, IMstd, renorm, denorm, xdnorm


def projGradDescent(model: th.nn.Module, images: th.Tensor, labels: th.Tensor,
        *, eps: float = 0.0, alpha: float = 2./255., pgditer: int = 1,
        verbose=False, device='cpu', targeted=False, unbound=False,
        rinit=False, B_UAP=False):
    '''
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/gradient.py
    https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/projected_gradient_descent.py
    This function implements BIM when rinit==False. It becomes PGD when rinit==True.
    B-UAP is the batch-wise universal (image-agnostic) adversarial perturbation
    '''
    assert(type(images) == th.Tensor)
    assert(eps is not None)
    # prepare
    images = images.to(device).clone().detach()
    images_orig = images.clone().detach()
    images.requires_grad = True
    labels = labels.to(device).view(-1)
    # random start?
    if bool(os.getenv('RINIT', False)):
        rinit = True
    if rinit:
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            if not B_UAP:
                images = images + eps*2*(0.5-th.rand(images.shape, device=device))
            else:
                images = images + eps*2*(0.5-th.rand([1,*images.shape[1:]], device=device))
            images = th.clamp(images, min=0., max=1.)
            images = images.clone().detach()
            images.requires_grad = True
        elif images_orig.min() < 0.:
            if not B_UAP:
                images = images + (eps/IMstd[:,None,None]).to(device)*2*(0.5-th.rand(images.shape, device=device))
            else:
                images = images + (eps/IMstd[:,None,None]).to(device)*2*(0.5-th.rand([1,*images.shape[1:]], device=device))
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
            images = images.clone().detach()
            images.requires_grad = True
        else:
            raise Exception
    # start attack
    model.eval() # NOTE: model.train() may incur problems
    for iteration in range(pgditer):
        # setup optimizers, and clear the gradient
        optim = th.optim.SGD(model.parameters(), lr=0.)
        optim.zero_grad()
        optimx = th.optim.SGD([images], lr=1.)
        optimx.zero_grad()
        # forward
        output = model.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        if not targeted:
            loss = -loss  # gradient ascent
        else:
            pass  # gradient descent
        loss.backward()

        # Batch UAP?
        if B_UAP:
            # batch-wise UAP. let's aggregate the adversarial perturbation
            with th.no_grad():
                aggrad = images.grad.mean(0, keepdim=True).detach().repeat([images.shape[0], 1, 1, 1])
                images.grad.data.copy_(aggrad)
        # iterative? or single-step?
        if pgditer > 1:
            if images_orig.min() >= 0. and images_orig.max() <= 1.:
                images.grad.data.copy_(alpha * th.sign(images.grad))
            elif images_orig.min() < 0.:
                images.grad.data.copy_((alpha/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                raise Exception
        else:
            if images.min() >= 0. and images.max() <= 1.:
                images.grad.data.copy_(eps * th.sign(images.grad))
            elif images.min() < 0.:
                images.grad.data.copy_((eps/IMstd[:,None,None]).to(device) * th.sign(images.grad))
            else:
                raise Exception
        # update the input
        optimx.step()
        # project the input (L_\infty bound)
        if not unbound:
            if images_orig.min() >= 0. and images_orig.max() <= 1.:
                images = th.min(images, images_orig + eps)
                images = th.max(images, images_orig - eps)
            elif images_orig.min() < 0.:
                images = th.min(images, images_orig + (eps/IMstd[:,None,None]).to(device))
                images = th.max(images, images_orig - (eps/IMstd[:,None,None]).to(device))
            else:
                raise Exception
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            images = th.clamp(images, min=0., max=1.)
        elif images_orig.min() < 0.:
            images = th.max(images, renorm(th.zeros(images.shape, device=device)))
            images = th.min(images, renorm(th.ones(images.shape, device=device)))
        else:
            raise Exception
        # detach from computation graph and prepare for the next round
        images = images.clone().detach()
        images.requires_grad = True
        if pgditer > 1 and verbose:
            cprint('  (PGD)' if not B_UAP else '  (B-UAP)', 'blue', end=' ')
            print(f'iter {iteration:3d}', f'\tloss= {loss.item():7.4f}',
                    f'\tL2m= {(images-images_orig).norm(2,dim=1).mean():7.4f}',
                    f'\tL0m= {(images-images_orig).abs().max(dim=1)[0].mean():7.4f}')
    if False: # visualization
        for i in range(images.shape[0]):
            npxo = images_orig[i].detach().cpu().squeeze().view(28,28).numpy()
            npx  = images[i].detach().cpu().squeeze().view(28,28).numpy()
            lab.subplot(121); lab.imshow(npxo, cmap='gray'); lab.colorbar()
            lab.subplot(122); lab.imshow(npx,  cmap='gray'); lab.colorbar()
            lab.show()
    xr = images.detach()
    xr.requires_grad = False
    r = (images - images_orig).detach()
    r.requires_grad = False
    if verbose:
        if images_orig.min() >= 0. and images_orig.max() <= 1.:
            tmp = r.view(r.shape[0], -1)
        elif images_orig.min() < 0.:
            tmp = r.mul(IMstd[:,None,None].to(r.device)).view(r.shape[0], -1)
        else:
            raise Exception

        cprint('r>', 'white', 'on_cyan', end=' ')
        print('Min', '%.3f'%tmp.min().item(),
                'Max', '%.3f'%tmp.max().item(),
                'Mean', '%.3f'%tmp.mean().item(),
                'L0', '%.3f'%tmp.norm(0, dim=1).mean().item(),
                'L1', '%.3f'%tmp.norm(1, dim=1).mean().item(),
                'L2', '%.3f'%tmp.norm(2, dim=1).mean().item())
    return (xr, r)
