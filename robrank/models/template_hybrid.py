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
import torch as th
import torchvision as vision
import pytorch_lightning as thl
from pytorch_lightning.utilities.enums import DistributedType
import os
import re
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .. import datasets
from .. import configs
from .. import utils
import multiprocessing as mp
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as __nmi
from ..attacks import AdvRank
from .svdreg import svdreg
from tqdm import tqdm
import functools
from .. import losses
#
try:
    import faiss
    faiss.omp_set_num_threads(4)
except ImportError:
    from sklearn.cluster import KMeans
#
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    pass
#
import rich
c = rich.get_console()


class Hc2f2(th.nn.Module):
    '''
    Hybrid C2F2
    '''

    def __init__(self):
        super(Hc2f2, self).__init__()
        self.repnet = th.nn.Sequential(
            th.nn.Conv2d(1, 32, kernel_size=5, padding=2),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            th.nn.ReLU(),
            th.nn.MaxPool2d(kernel_size=2, stride=2),
            th.nn.Flatten(),
            th.nn.Linear(64 * 7 * 7, 1024),
        )
        self.embnet = th.nn.Sequential(
            th.nn.ReLU(),
            th.nn.Linear(1024, 512),
        )
        self.clsnet = th.nn.Sequential(
            th.nn.ReLU(),
            th.nn.Dropout(p=0.4),
            th.nn.Linear(1024, 10),
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        repres = self.repnet(x)
        embeds = self.embnet(repres)
        logits = self.clsnet(repres)
        return (repres, embeds, logits)


class Hresnet18(th.nn.Module):
    '''
    Hybrid ResNet18
    '''

    def __init__(self, num_cls: int, emb_dim: int = 512):
        super(Hresnet18, self).__init__()
        self.repnet = vision.models.resnet18(pretrained=True)
        self.repnet.fc = th.nn.Identity()
        self.embnet = th.nn.Sequential(
            th.nn.ReLU(), th.nn.Linear(512, emb_dim))
        self.clsnet = th.nn.Sequential(
            th.nn.ReLU(), th.nn.Dropout(p=0.2),
            th.nn.Linear(512, num_cls))

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        repres = self.repnet(x)
        embeds = self.embnet(repres)
        logits = self.clsnet(repres)
        return (repres, embeds, logits)


class HybridModelBase(thl.LightningModule):
    '''
    Deep Metric Learning + Classification Hybrid Model
    '''
    BACKBONE = 'hc2f2'
    _valvecs = None
    _vellabs = None

    def __init__(self, *, dataset: str, loss: str):
        super(HybridModelBase, self).__init__()
        # configure
        assert(dataset in getattr(configs, self.BACKBONE).allowed_datasets)
        self.dataset = dataset
        assert(loss in getattr(configs, self.BACKBONE).allowed_losses)
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        self.config = getattr(configs, self.BACKBONE)(dataset, loss)
        # modules
        if self.BACKBONE == 'hc2f2':
            self.backbone = Hc2f2()
        elif self.BACKBONE == 'hres18':
            num_class = getattr(configs, dataset).num_class
            self.backbone = Hresnet18(num_class)
        else:
            raise Exception(f'!unknown BACKBONE {self.BACKBONE}')

    def setup(self, stage=None):
        train, test = getattr(
            datasets, self.dataset).getDataset(self.datasetspec)
        self.data_train = train
        self.data_val = test

    def train_dataloader(self):
        train_loader = DataLoader(self.data_train,
                                  batch_size=self.config.batchsize,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=self.config.loader_num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.data_val,
                                batch_size=self.config.valbatchsize,
                                pin_memory=True,
                                num_workers=self.config.loader_num_workers)
        return val_loader

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),
                              lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optim

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        if hasattr(self, 'is_advtrain') and self.is_advtrain:
            raise NotImplementedError

        # prepare and forward
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device).view(-1).long()
        if any(x in self.dataset for x in ('sop', 'cub', 'cars')):
            images = images.view(-1, 3, 224, 224)
        elif any(x in self.dataset for x in ('mnist', 'fashion')):
            images = images.view(-1, 1, 28, 28)
        else:
            raise ValueError('illegal dataset')
        repres, embeddings, logits = self.forward(images)

        # metric
        loss_dml = self.lossfunc(embeddings, labels)
        self.log('Train/loss_dml', loss_dml.item())
        if hasattr(self, 'do_svd') and self.do_svd:
            raise NotImplementedError

        # classification
        loss_ce = F.cross_entropy(logits, labels)
        self.log('Train/loss_ce', loss_ce.item())
        accuracy = logits.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Train/accuracy', accuracy)

        return loss_dml + loss_ce

    def _recompute_valvecs(self):
        with th.no_grad():
            c.print(
                '\n[yellow]Re-Computing Validation Set Representations ...',
                end=' ')
            valvecs, vallabs = [], []
            for i, (images, labels) in enumerate(self.val_dataloader()):
                images, labels = images.to(
                    self.device), labels.view(-1).to(self.device)
                repres, embs, logits = self.forward(images)
                if self.metric in ('C', 'N'):
                    nembs = F.normalize(embs, p=2, dim=-1)
                valvecs.append(nembs.detach())
                vallabs.append(labels.detach())
            valvecs, vallabs = th.cat(valvecs), th.cat(vallabs)
        self._valvecs = valvecs
        self._vallabs = vallabs
        return (valvecs, vallabs)

    def validation_step(self, batch, batch_idx):
        images = batch[0].to(self.device)
        labels = batch[1].to(self.device).view(-1).long()
        with th.no_grad():
            repres, embeddings, logits = self.forward(images)

        # deep metric learning
        if self._valvecs is None:
            self._recompute_valvecs()
        with th.no_grad():
            if self.metric == 'N':
                nembs = F.normalize(embeddings, p=2, dim=-1)
                dist = th.cdist(nembs, self._valvecs)
            else:
                raise NotImplementedError
            r, r_1, r_2, mAP = [], [], [], []
            for i in range(repres.size(0)):
                _r, _r1, _r2 = utils.metric_get_rank(dist[i], labels[i],
                                                     self._vallabs, ks=[1, 2])
                r.append(_r)
                r_1.append(_r1)
                r_2.append(_r2)
                mAP.append(
                    utils.metric_get_ap(
                        dist[i],
                        labels[i],
                        self._vallabs))
            r, r_1, r_2 = np.mean(r), np.mean(r_1), np.mean(r_2)
            mAP = np.mean(mAP)

        # classification
        with th.no_grad():
            loss_ce = F.cross_entropy(logits, labels)
            accuracy = logits.max(1)[1].eq(
                labels.view(-1)).sum().item() / labels.nelement()
        self.log('Validation/loss_ce', loss_ce.item())
        self.log('Validation/accuracy', accuracy)

        loss = loss_ce
        return {'loss': loss.item(), 'accuracy': accuracy,
                'r@1': r_1, 'r@2': r_2, 'r@M': r, 'mAP': mAP}

    def validation_epoch_end(self, outputs: list):
        # reduce:mean the summary of every process
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
        if str(self._distrib_type) in (
                'DistributedType.DDP', 'DistributedType.DDP2'):
            th.distributed.barrier()
            for key in summary.keys():
                tmp = th.tensor(summary[key]).to(self.device)
                th.distributed.all_reduce(tmp, op=th.distributed.ReduceOp.SUM)
                summary[key] = tmp.item() / th.distributed.get_world_size()

        # Calculate the rest scores
        nmi = utils.metric_get_nmi(
            self._valvecs,
            self._vallabs,
            self.config.num_class)
        summary['NMI'] = nmi

        # clean up
        self._valvecs = None
        self._vallabs = None

        # log and print
        self.log('Validation/r@M', summary['r@M'])
        self.log('Validation/r@1', summary['r@1'])
        self.log('Validation/r@2', summary['r@2'])
        self.log('Validation/mAP', summary['mAP'])
        self.log('Validation/NMI', summary['NMI'])
        self.log('Validation/accuracy', summary['accuracy'])
        c.print(f'\nValidation │ ' +
                f'r@M= {summary["r@M"]:.1f} ' +
                f'r@1= {summary["r@1"]:.3f} ' +
                f'r@2= {summary["r@2"]:.3f} ' +
                f'mAP= {summary["mAP"]:.3f} ' +
                f'NMI= {summary["NMI"]:.3f} ' +
                f'Acc= {summary["accuracy"]:.3f}')
