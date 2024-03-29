'''
Copyright (C) 2019-2022, Mo Zhou <cdluminate@gmail.com>

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
# defenses/est.py
# Embedding-Shifted Triplet Defense (ECCV 2020) arXiv: 2002.11293
###############################################################################

from ..attacks import AdvRank
from ..models.svdreg import svdreg
import torch as th
import rich
c = rich.get_console()


def est_training_step(model, batch, batch_idx, *, pgditer: int = None):
    '''
    Do adversarial training using Mo's defensive triplet (2002.11293 / ECCV'20)
    Embedding-Shifted Triplet (EST)
    Confirmed for MNIST/Fashion-MNIST/CUB/CARS/SOP
    This defense is generic to network architecture and metric learning loss.

    Arguments:
        model: a pytorch_lightning.LightningModule instance. It projects
               input images into a embedding space.
        batch: see pytorch lightning protocol for training_step(...)
        batch_idx: see pytorch lightning protocol for training_step(...)
    '''
    images, labels = (batch[0].to(model.device), batch[1].to(model.device))
    # generate adversarial examples
    #advatk_metric = 'C' if model.dataset in ('mnist', 'fashion') else 'C'
    advrank = AdvRank(model, eps=model.config.advtrain_eps,
                      alpha=model.config.advtrain_alpha,
                      pgditer=model.config.advtrain_pgditer if pgditer is None else pgditer,
                      device=model.device,
                      metric=model.metric, verbose=False)
    # set shape
    if any(x in model.dataset for x in ('sop', 'cub', 'cars')):
        shape = (-1, 3, 224, 224)
    elif any(x in model.dataset for x in ('mnist', 'fashion')):
        shape = (-1, 1, 28, 28)
    else:
        raise ValueError(f'does not recognize dataset {model.dataset}')
    # eval orig
    with th.no_grad():
        output_orig = model.forward(images.view(*shape))
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adv examples
    model.wantsgrad = True
    model.eval()
    advimgs = advrank.embShift(images.view(*shape))
    model.train()
    output = model.forward(advimgs.view(*shape))
    model.wantsgrad = False
    # compute loss
    loss = model.lossfunc(output, labels)
    if hasattr(model, 'do_svd') and model.do_svd:
        loss += svdreg(model, output)
    model.log('Train/loss', loss)
    #tqdm.write(f'* OriLoss {loss_orig.item():.3f} | [AdvLoss] {loss.item():.3f}')
    model.log('Train/OriLoss', loss_orig.item())
    model.log('Train/AdvLoss', loss.item())
    return loss


def est_training_step_cosine_only(model, batch, batch_idx, *, pgditer:int = None,
        do_batcheff: bool = False):
    '''
    do not train the model. only measure the cosine similarty to reflect
    misleading gradients for figure 2 in pami
    '''
    images, labels = (batch[0].to(model.device), batch[1].to(model.device))
    advrank = AdvRank(model, eps=model.config.advtrain_eps,
                      alpha=model.config.advtrain_alpha,
                      pgditer=model.config.advtrain_pgditer if pgditer is None else pgditer,
                      device=model.device,
                      metric=model.metric, verbose=False)
    # set shape
    if any(x in model.dataset for x in ('sop', 'cub', 'cars')):
        shape = (-1, 3, 224, 224)
    elif any(x in model.dataset for x in ('mnist', 'fashion')):
        shape = (-1, 1, 28, 28)
    else:
        raise ValueError(f'does not recognize dataset {model.dataset}')
    # eval orig
    with th.no_grad():
        output_orig = model.forward(images.view(*shape))
        loss_orig = model.lossfunc(output_orig, labels)
    # generate adv examples
    model.wantsgrad = True
    model.eval()
    advimgs = advrank.embShift(images.view(*shape))
    model.train()
    output = model.forward(advimgs.view(*shape))
    model.wantsgrad = False
    # compute cosine similarty
    with th.no_grad():
        if not do_batcheff:
            cosine = model.lossfunc.cosine_only(output, output_orig, labels.view(-1))
        else:
            cosine = model.lossfunc.batcheff_only(output, output_orig, labels.view(-1))
        #print(cosine)
        if not hasattr(model, 'cosine_only_stat'):
            model.cosine_only_stat = []
        model.cosine_only_stat.extend(cosine)
        #print(len(model.cosine_only_stat))
    # compute fake loss
    loss = th.tensor(0.0, requires_grad=True, device=model.device)
    model.log('Train/loss', loss)
    #tqdm.write(f'* OriLoss {loss_orig.item():.3f} | [AdvLoss] {loss.item():.3f}')
    model.log('Train/OriLoss', loss_orig.item())
    model.log('Train/AdvLoss', loss.item())
    return loss
