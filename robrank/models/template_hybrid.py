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
import torchvision as vision
import pytorch_lightning as thl
from pytorch_lightning.utilities.enums import DistributedType
import os
import re
import pytorch_lightning.metrics.functional
import torch.nn.functional as F
from .. import datasets
from .. import configs
from .. import utils
import multiprocessing as mp
from termcolor import cprint
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


class HybridModelBase(object):
    '''
    Deep Metric Learning + Classification Hybrid
    '''
    BACKBONE = 'resnet18'

    def __init__(self, *, dataset: str, loss: str):
        # FIXME
        super().__init__()
        # configure
        assert(dataset in getattr(configs, self.backbone).allowed_datasets)
        self.dataset = dataset
        assert(loss in getattr(configs, self.backbone).allowed_datasets)
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        self.config = getattr(configs, self.backbone)(dataset, loss)
        # modules
        if self.BACKBONE == 'c2f1':
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
                th.nn.Dropout(p=0.4, training=self.training),
                th.nn.Linear(1024, 10),
            )
        else:
            raise Exception(f'unknown BACKBONE {BACKBONE}')
        raise NotImplementedError

    def forward(self, x):
        if self.BACKBONE == 'c2f1':
            repres = self.repnet(x)
            embeddings = self.embnet(repres)
            logits = self.clsnet(repres)
            return (repres, embeddings, logits)
        else:
            raise Exception(f'unknown BACKBONE {BACKBONE}')

    def training_step(self, batch, batch_idx):
        if hasattr(self, 'is_advtrain') and self.is_advtrain:
            raise NotImplementedError

        # prepare and forward
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
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
        self.loss('Train/loss_ce', loss_ce.item())
        accuracy = logits.max(1)[1].eq(
                labels.view(-1)).sum().item() / labels.nelement()
        self.log('Train/accuracy', accuracy)

        return loss_dml + loss_ce

    def _recompute_valvecs(self):
        with th.no_grad():
            cprint('\nRe-Computing Validation Set Representations ...',
                   'yellow', end=' ')
            valvecs, vallabs = [], []
            dataloader = self.val_dataloader()
            #iterator = tqdm(enumerate(dataloader), total=len(dataloader))
            iterator = enumerate(dataloader)
            for i, (images, labels) in iterator:
                images, labels = images.to(
                    self.device), labels.view(-1).to(self.device)
                output = self.forward(images)
                if self.metric in ('C', 'N'):
                    output = th.nn.functional.normalize(output, p=2, dim=-1)
                valvecs.append(output.detach())
                vallabs.append(labels.detach())
            valvecs, vallabs = th.cat(valvecs), th.cat(vallabs)
        self._valvecs = valvecs
        self._vallabs = vallabs
        return (valvecs, vallabs)


    def validation_step(self, batch, batch_idx):
        # FIXME
        with th.no_grad():
            images, labels = (batch[0].to(self.device), batch[1].to(self.device))
            repres, embeddings, logits = self.forward(images)

        # metric
        if self._valvecs is None:
            self._recompute_valvecs()
        with th.no_grad():
            if self.metric == 'N':
                nembs = F.normalize(embeddings, p=2, dim=-1)
            else:
                raise NotImplementedError
            dist = th.cdist(nembs, self._valvecs)
            knnsearch = self._vallabs[dist.argsort(
                dim=1, descending=False)[:, 1]].flatten()
            recall1 = knnsearch.eq(labels).float().mean()
            knn2 = self._vallabs[dist.argsort(
                dim=1, descending=False)[:, 2]].flatten()
            recall2 = th.logical_or(knn2 == labels, knnsearch == labels).float(
            ).mean()
            # AP
            mAP = []
            for i in range(dist.size(0)):
                mAP.append(_get_ap(dist[i], labels[i], self._vallabs))
            mAP = np.mean(mAP)

        # classification
        loss_ce = F.cross_entropy(logits, labels)
        self.log('Validation/loss_ce', loss_ce.item())
        accuracy = logits.max(1)[1].eq(
                labels.view(-1)).sum().item() / labels.nelement()
        self.log('Validation/accuracy', accuracy)

        loss = loss_ce
        return {'loss': loss.item(), 'accuracy', accuracy,
                'r@1': recall1.item(), 'r@2': recall2.item(),
                'mAP': mAP}

    def validation_epoch_end(self, outputs: list):

        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}

        # metric
        nmi = _get_nmi(self._valvecs, self._vallabs, self.config.num_class)
        self._valvecs = None
        self._vallabs = None
                summary['NMI'] = nmi
        if str(self._distrib_type) in (
                'DistributedType.DDP', 'DistributedType.DDP2'):
            #print(self._distrib_type, th.distributed.get_rank())
            th.distributed.barrier()
            recall1 = th.tensor(summary['r@1']).to(self.device)
            th.distributed.all_reduce(recall1, op=th.distributed.ReduceOp.SUM)
            summary['r@1'] = recall1.item(
            ) / th.distributed.get_world_size()
            recall2 = th.tensor(summary['r@2']).to(self.device)
            th.distributed.all_reduce(recall2, op=th.distributed.ReduceOp.SUM)
            summary['r@2'] = recall2.item(
            ) / th.distributed.get_world_size()
            tmp = th.tensor(summary['mAP']).to(self.device)
            th.distributed.all_reduce(tmp, op=th.distributed.ReduceOp.SUM)
            summary['mAP'] = tmp.item() / th.distributed.get_world_size()
        # write into log
        self.log('Validation/NMI', summary['NMI'])
        self.log('Validation/r@1', summary['r@1'])
        self.log('Validation/r@2', summary['r@2'])
        self.log('Validation/mAP', summary['mAP'])

        # report
        c.print(f'Validation | loss= {summary["loss"]:.5f}  accuracy= {summary["accuracy"]:.5f}')
        c.print(
            f'\nValidation │ r@1= {summary["r@1"]:.5f}' +
            f' r@2= {summary["r@2"]:.5f}' +
            f' mAP= {summary["mAP"]:.3f}' +
            f' NMI= {summary["NMI"]:.3f}')

    def configure_optimizers(self):
        optim = th.optim.Adam([self.repnet.parameters(),
            self.embnet.parameters(), self.clsnet.parameters()],
            lr.self.config.lr, weight_decay=self.config.weight_decay)
        return optim

    def setup(self, stage=None):
        train, test = getattr(
                datasets, self.dataset).getDataset(self.datasetspec)
        self.data_train = train
        self.data_val = test

    def train_dataloader(self):
        train_loader = th.utils.data.DataLoader(self.data_train,
                batch_size=self.config.batchsize,
                shuffle=True,
                pin_memory=True,
                num_workers=self.config.loader_num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = th.utils.data.DataLoader(self.data_val,
                batch_size=self.config.valbatchsize,
                pin_memory=True,
                num_workers=self.config.loader_num_workers)
        return val_loader
