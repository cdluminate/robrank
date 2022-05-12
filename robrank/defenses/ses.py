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
# defenses/ses.py
# Suppressing Embedding Shift (SES) Defense
# Described in the supplementary material of ECCV 2020, as well as the preprint
# paper arXiv:2106.03614.
###############################################################################

from ..attacks import AdvRank
import torch.nn.functional as F
import rich
c = rich.get_console()


def ses_training_step(model, batch, batch_idx):
    '''
    Adversarial training by directly supressing embedding shift (SES)
    max(*.es)->advimg, min(advimg->emb,oriimg->img;*.metric)
    Confirmed for MNIST/Fashion-MNIST
    [ ] for CUB/SOP

    This defense has been discussed in the supplementary material / appendix
    of the ECCV20 paper. (See arxiv: 2002.11293)
    '''
    images, labels = (batch[0].to(model.device), batch[1].to(model.device))
    # generate adversarial examples
    advatk_metric = model.metric
    advrank = AdvRank(model, eps=model.config.advtrain_eps,
                      alpha=model.config.advtrain_alpha,
                      pgditer=model.config.advtrain_pgditer,
                      device=model.device,
                      metric=advatk_metric, verbose=False)
    # setup shape
    if any(x in model.dataset for x in ('sop', 'cub', 'cars')):
        shape = (-1, 3, 224, 224)
    elif any(x in model.dataset for x in ('mnist', 'fashion')):
        shape = (-1, 1, 28, 28)
    else:
        raise ValueError('illegal dataset!')
    # find adversarial example
    model.wantsgrad = True
    model.eval()
    advimgs = advrank.embShift(images.view(*shape))
    model.train()
    model.watnsgrad = False
    # evaluate advtrain loss
    output_orig = model.forward(images.view(*shape))
    loss_orig = model.lossfunc(output_orig, labels)
    output_adv = model.forward(advimgs.view(*shape))
    # select defense method
    if model.metric == 'E':
        # this is a trick to normalize non-normed Euc embedding,
        # or the loss could easily diverge.
        nadv = F.normalize(output_adv)
        embshift = F.pairwise_distance(nadv, output_orig)
    elif model.metric == 'N':
        nori = F.normalize(output_orig)
        nadv = F.normalize(output_adv)
        embshift = F.pairwise_distance(nadv, nori)
    elif model.metric == 'C':
        embshift = 1 - F.cosine_similarity(output_adv, output_orig)
    # loss and log
    # method 1: loss_triplet + loss_embshift
    loss = loss_orig + 1.0 * embshift.mean()
    # method 2: loss_triplet + loss_embshiftp2
    #loss = loss_orig + 1.0 * (embshift ** 2).mean()
    if hasattr(model, 'do_svd') and model.do_svd:
        loss += svdreg(model, output_adv)
    model.log('Train/loss', loss)
    model.log('Train/OriLoss', loss_orig.item())
    model.log('Train/AdvLoss', embshift.mean().item())
    model.log('Train/embShift', embshift.mean().item())
    return loss
