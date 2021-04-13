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


class ClassifierTemplate(object):

    def training_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Train/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Train/accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Validation/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Validation/accuracy', accuracy)
        return {'loss': loss.item(), 'accuracy': accuracy}

    def validation_epoch_end(self, outputs: list):
        if str(self._distrib_type) in (
                'DistributedType.DDP', 'DistributedType.DDP2'):
            if th.distributed.get_rank() != 0:
                return
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
        cprint(
            f'\tValidation |  loss= {summary["loss"]:.5f}  accuracy= {summary["accuracy"]:.5f}',
            'yellow', None, ['bold'])

    def test_step(self, batch, batch_idx):
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        output = self.forward(images)
        loss = th.nn.functional.cross_entropy(output, labels)
        self.log('Test/loss', loss.item())
        accuracy = output.max(1)[1].eq(
            labels.view(-1)).sum().item() / labels.nelement()
        self.log('Test/accuracy', accuracy)


class MetricBase(object):

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
        # XXX: in DDP mode the size of valvecs is 10000, looks correct
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    th.distributed.barrier()
        #    sizes_slice = [th.tensor(0).to(self.device)
        #                   for _ in range(th.distributed.get_world_size())]
        #    size_slice = th.tensor(len(valvecs)).to(self.device)
        #    th.distributed.all_gather(sizes_slice, size_slice)
        #    print(sizes_slice)
        #    print(f'[th.distributed.get_rank()]Shape:',
        #          valvecs.shape, vallabs.shape)
        self._valvecs = valvecs
        self._vallabs = vallabs
        return (valvecs, vallabs)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        if hasattr(self, 'is_advtrain') and self.is_advtrain:
            return self.adv_training_step(batch, batch_idx)
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        if any(x in self.dataset for x in ('sop', 'cub', 'cars')):
            output = self.forward(images.view(-1, 3, 224, 224))
        elif any(x in self.dataset for x in ('mnist', 'fashion')):
            output = self.forward(images.view(-1, 1, 28, 28))
        else:
            raise ValueError(f'illegal dataset')
        loss = self.lossfunc(output, labels)
        if hasattr(self, 'do_svd') and self.do_svd:
            loss += svdreg(self, output)
        self.log('Train/loss', loss)
        #tqdm.write(f'* OriLoss {loss.item():.3f}')
        self.log('Train/OriLoss', loss.item())
        return loss

    def adv_training_step(self, batch, batch_idx):
        '''
        Do adversarial training using Mo's defensive triplet (2002.11293)
        Confirmed for MNIST/Fashion-MNIST
        [ ] for CUB/SOP
        '''
        if hasattr(self, 'is_advtrain_embshift') and self.is_advtrain_embshift:
            return self.adv_training_step_embshift(batch, batch_idx)
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        # generate adversarial examples
        advatk_metric = 'C' if self.dataset in ('mnist', 'fashion') else 'C'
        advrank = AdvRank(self, eps=self.config.advtrain_eps,
                          alpha=3. / 255. if self.config.advtrain_eps > 0.1 else 1. / 255.,
                          pgditer=32 if self.config.advtrain_eps > 0.1 else 24,
                          device=self.device,
                          metric=advatk_metric, verbose=False)
        # set shape
        if any(x in self.dataset for x in ('sop', 'cub', 'cars')):
            shape = (-1, 3, 224, 224)
        elif any(x in self.dataset for x in ('mnist', 'fashion')):
            shape = (-1, 1, 28, 28)
        else:
            raise ValueError(f'does not recognize dataset {self.dataset}')
        # eval orig
        with th.no_grad():
            output_orig = self.forward(images.view(*shape))
            loss_orig = self.lossfunc(output_orig, labels)
        # generate adv examples
        self.wantsgrad = True
        self.eval()
        advimgs = advrank.embShift(images.view(*shape))
        self.train()
        output = self.forward(advimgs.view(*shape))
        self.wantsgrad = False
        # compute loss
        loss = self.lossfunc(output, labels)
        if hasattr(self, 'do_svd') and self.do_svd:
            loss += svdreg(self, output)
        self.log('Train/loss', loss)
        #tqdm.write(f'* OriLoss {loss_orig.item():.3f} | [AdvLoss] {loss.item():.3f}')
        self.log('Train/OriLoss', loss_orig.item())
        self.log('Train/AdvLoss', loss.item())
        return loss

    def adv_training_step_embshift(self, batch, batch_idx):
        '''
        Adversarial training by directly supressing embedding shift
        max(*.es)->advimg, min(advimg->emb,oriimg->img;*.metric)
        Confirmed for MNIST/Fashion-MNIST
        [ ] for CUB/SOP
        '''
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        # generate adversarial examples
        advatk_metric = self.metric
        advrank = AdvRank(self, eps=self.config.advtrain_eps,
                          alpha=3. / 255. if self.config.advtrain_eps > 0.1 else 1. / 255.,
                          pgditer=32 if self.config.advtrain_eps > 0.1 else 24,
                          device=self.device,
                          metric=advatk_metric, verbose=False)
        # setup shape
        if any(x in self.dataset for x in ('sop', 'cub', 'cars')):
            shape = (-1, 3, 224, 224)
        elif any(x in self.dataset for x in ('mnist', 'fashion')):
            shape = (-1, 1, 28, 28)
        else:
            raise ValueError('illegal dataset!')
        # find adversarial example
        self.wantsgrad = True
        self.eval()
        advimgs = advrank.embShift(images.view(*shape))
        self.train()
        self.watnsgrad = False
        # evaluate advtrain loss
        output_orig = self.forward(images.view(*shape))
        loss_orig = self.lossfunc(output_orig, labels)
        output_adv = self.forward(advimgs.view(*shape))
        # select defense method
        if self.metric == 'E':
            # this is a trick to normalize non-normed Euc embedding,
            # or the loss could easily diverge.
            nadv = F.normalize(output_adv)
            embshift = F.pairwise_distance(nadv, output_orig).mean()
        elif self.metric == 'N':
            nori = F.normalize(output_orig)
            nadv = F.normalize(output_adv)
            embshift = F.pairwise_distance(nadv, nori).mean()
        elif self.metric == 'C':
            embshift = (
                1 -
                F.cosine_similarity(
                    output_adv,
                    output_orig)).mean()
        # loss and log
        loss = loss_orig + embshift * 1.0
        if hasattr(self, 'do_svd') and self.do_svd:
            loss += svdreg(self, output_adv)
        self.log('Train/loss', loss)
        self.log('Train/OriLoss', loss_orig.item())
        self.log('Train/AdvLoss', embshift.item())
        self.log('Train/embShift', embshift.item())
        return loss


def _get_f1(dist: th.Tensor, label: th.Tensor,
            vallabels: th.Tensor, ks: list):
    pass


def _get_nmi(valvecs: th.Tensor, vallabs: th.Tensor, ncls: int) -> float:
    '''
    Compute the NMI score
    '''
    use_cuda: bool = th.cuda.is_available() and hasattr(faiss, 'StandardGpuResources')
    if bool(os.getenv('FAISS_CPU', 0)):
        use_cuda = False
    npvecs = valvecs.detach().cpu().numpy().astype(np.float32)
    nplabs = vallabs.detach().cpu().view(-1).numpy().astype(np.float32)
    # a weird dispatcher but it works.
    if 'faiss' in globals():
        if use_cuda:
            gpu_resource = faiss.StandardGpuResources()
            cluster_idx = faiss.IndexFlatL2(npvecs.shape[1])
            if not th.distributed.is_initialized():
                cluster_idx = faiss.index_cpu_to_gpu(
                    gpu_resource, 0, cluster_idx)
            else:
                cluster_idx = faiss.index_cpu_to_gpu(gpu_resource,
                                                     th.distributed.get_rank(), cluster_idx)
            kmeans = faiss.Clustering(npvecs.shape[1], ncls)
            kmeans.verbose = False
            kmeans.train(npvecs, cluster_idx)
            _, pred = cluster_idx.search(npvecs, 1)
            pred = pred.flatten()
        else:
            kmeans = faiss.Kmeans(
                npvecs.shape[1], ncls, seed=123, verbose=False)
            kmeans.train(npvecs)
            _, pred = kmeans.index.search(npvecs, 1)
            pred = pred.flatten()
        nmi = __nmi(nplabs, pred)
    elif 'KMeans' in globals():
        kmeans = KMeans(n_clusters=ncls, random_state=0).fit(npvecs)
        nmi = __nmi(nplabs, kmeans.labels_)
    else:
        raise NotImplementedError(
            'please provide at leaste one kmeans implementation for the NMI metric.')
    return nmi


def _get_rank(dist: th.Tensor, label: th.Tensor,
              vallabels: th.Tensor, ks: list) -> tuple:
    '''
    Flexibly get the rank of the topmost item in the same class
    dist = [dist(anchor,x) for x in validation_set]
    '''
    assert(dist.nelement() == vallabels.nelement())
    # XXX: [important] argsort(...)[:,1] for skipping the diagonal (R@1=1.0)
    # we skip the smallest value as it's exactly for the anchor itself
    argsort = dist.argsort(descending=False)[1:]
    rank = th.where(vallabels[argsort] == label)[0].min().item()
    return (rank,) + tuple(rank < k for k in ks)


def _get_ap(dist: th.Tensor, label: th.Tensor, vallabels: th.Tensor) -> float:
    '''
    Get the overall average precision
    '''
    assert(dist.nelement() == vallabels.nelement())
    # we skip the smallest value as it's exectly for the anchor itself
    argsort = dist.argsort(descending=False)[1:]
    argwhere1 = th.where(vallabels[argsort] == label)[0] + 1
    ap = ((th.arange(len(argwhere1)).float() + 1).to(argwhere1.device) /
          argwhere1).sum().item() / len(argwhere1)
    return ap


###############################################################################
class MetricTemplate28(MetricBase):
    '''
    Deep Metric Learning with MNIST-compatible Network.
    '''
    is_advtrain = False
    do_svd = False
    backbone = 'c2f1'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in getattr(configs, self.backbone).allowed_datasets)
        assert(loss in getattr(configs, self.backbone).allowed_losses)
        self.dataset = dataset
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        # configuration
        self.config = getattr(configs, self.backbone)(dataset, loss)
        # modules
        if self.backbone == 'c2f1':
            '''
            A 2-Conv Layer 1-FC Layer Network For Ranking
            See [Madry, advrank] for reference.
            '''
            self._model = th.nn.Sequential(
                th.nn.Conv2d(1, 32, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(64 * 7 * 7, 1024),
            )
        elif self.backbone == 'lenet':
            self._model = th.nn.Sequential(
                th.nn.Conv2d(1, 20, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(20, 50, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(800, 500),
                th.nn.ReLU(),
                th.nn.Linear(500, 128),
            )
        else:
            raise ValueError('unknown backbone')
        # validation
        self._valvecs = None
        self._vallabs = None
        # summary
        c.print('[green]Model Meta Information[/green]', {
            'dataset': self.dataset,
            'datasestspec': self.datasetspec,
            'lossfunc': self.lossfunc,
            'metric': self.metric,
            'config': {k: v for (k, v) in self.config.__dict__.items()
                       if k not in ('allowed_losses', 'allowed_datasets')},
        })

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self._model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.lossfunc, 'getOptim'):
            optim2 = self.lossfunc.getOptim()
            return optim, optim2
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

    def validation_step(self, batch, batch_idx):
        #print('[', th.distributed.get_rank(), ']', batch_idx, '\n')
        if self._valvecs is None:
            self._recompute_valvecs()
        with th.no_grad():
            images, labels = (batch[0].to(self.device),
                              batch[1].to(self.device))
            output = self.forward(images)
            if self.metric == 'C':
                output = th.nn.functional.normalize(output, p=2, dim=-1)
                dist = 1 - th.mm(output, self._valvecs.t())
            elif self.metric in ('E', 'N'):
                if self.metric == 'N':
                    output = th.nn.functional.normalize(output, p=2, dim=-1)
                #X, D = self._valvecs.size(0), output.size(1)
                dist = th.cdist(output, self._valvecs)
                # dist = th.cat(list(map(lambda x: (x.view(x.size(0), 1, D).expand(
                #     x.size(0), X, D) - self._valvecs.view(1, X, D).expand(x.size(0),
                #     X, D)).norm(p=2, dim=2), output.split(16))))
            # XXX: [important] argsort(...)[:,1] for skipping the diagonal
            # (R@1=1.0)
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
        return {'r@1': recall1.item(), 'r@2': recall2.item(),
                'mAP': mAP}

    def validation_epoch_end(self, outputs: list):
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    #if th.distributed.get_rank() != 0:
        #    if self.local_rank != 0:
        #        return
        # NMI
        nmi = _get_nmi(self._valvecs, self._vallabs, self.config.num_class)
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    th.distributed.barrier()
        #    sizes_slice = [th.tensor(0).to(self.device)
        #                   for _ in range(th.distributed.get_world_size())]
        #    size_slice = th.tensor(self._valvecs.size(0)).to(self.device)
        #    th.distributed.all_gather(sizes_slice, size_slice)
        #    print(sizes_slice)
        self._valvecs = None
        self._vallabs = None
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
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
        #
        cprint(
            f'\nValidation │ r@1= {summary["r@1"]:.5f}' +
            f' r@2= {summary["r@2"]:.5f}' +
            f' mAP= {summary["mAP"]:.3f}' +
            f' NMI= {summary["NMI"]:.3f}',
            'yellow')


###############################################################################
class MetricTemplate224(MetricBase):
    '''
    Deep Metric Learning with Imagenet compatible network (2002.08473)

    Overload the backbone vairable to switch to resnet50, mnasnet,
    or even the efficientnet.
    '''
    BACKBONE = 'resnet18'
    is_advtrain = False
    do_svd = False

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # configuration
        if self.BACKBONE == 'resnet18':
            self.config = configs.res18(dataset, loss)
            self.backbone = vision.models.resnet18(pretrained=True)
        elif self.BACKBONE == 'resnet50':
            self.config = configs.res50(dataset, loss)
            self.backbone = vision.models.resnet50(pretrained=True)
        elif self.BACKBONE == 'resnet101':
            self.config = configs.res50(dataset, loss)
            self.backbone = vision.models.resnet101(pretrained=True)
        elif self.BACKBONE == 'resnet152':
            self.config = configs.res50(dataset, loss)
            self.backbone = vision.models.resnet152(pretrained=True)
        elif self.BACKBONE == 'mnas05':
            self.config = configs.mnas(dataset, loss)
            self.backbone = vision.models.mnasnet0_5(pretrained=True)
        elif self.BACKBONE == 'mnas10':
            self.config = configs.mnas(dataset, loss)
            self.backbone = vision.models.mnasnet1_0(pretrained=True)
        elif self.BACKBONE == 'mnas13':
            self.config = configs.mnas(dataset, loss)
            self.backbone = vision.models.mnasnet1_3(pretrained=True)
        elif self.BACKBONE == 'efficientnet-b0':
            self.config = configs.enb0(dataset, loss)
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.BACKBONE == 'efficientnet-b4':
            self.config = configs.enb4(dataset, loss)
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        elif self.BACKBONE == 'efficientnet-b7':
            self.config = configs.enb7(dataset, loss)
            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            raise ValueError()
        assert(dataset in self.config.allowed_datasets)
        self.dataset = dataset
        assert(loss in self.config.allowed_losses)
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        # modules
        if re.match(r'resnet',
                    self.BACKBONE) and self.config.embedding_dim > 0:
            if '18' in self.BACKBONE:
                emb_dim = 512
            elif '50' in self.BACKBONE:
                emb_dim = 2048
            else:
                emb_dim = 2048
            self.backbone.fc = th.nn.Linear(emb_dim, self.config.embedding_dim)
        elif re.match(r'resnet', self.BACKBONE):
            self.backbone.fc = th.nn.Identity()
        elif re.match(r'mnas', self.BACKBONE) and self.config.embedding_dim > 0:
            self.backbone.classifier = th.nn.Linear(
                1280, self.config.embedding_dim)
        elif re.match(r'mnas', self.BACKBONE):
            self.backbone.classifier = th.nn.Identity()
        elif re.match(r'efficientnet', self.BACKBONE) and self.config.embedding_dim > 0:
            if 'b0' in self.BACKBONE:
                emb_dim = 1280
            elif 'b7' in self.BACKBONE:
                emb_dim = 2560
            self.backbone._modules['_dropout'] = th.nn.Identity()
            self.backbone._modules['_fc'] = th.nn.Linear(
                emb_dim, self.config.embedding_dim)
            # self.backbone._modules['_swish'] = th.nn.Identity() # XXX: don't
            # override swish.
        elif re.match(r'efficientnet', self.BACKBONE):
            self.backbone._modules['_dropout'] = th.nn.Identity()
            self.backbone._modules['_fc'] = th.nn.Identity()
            # self.backbone._modules['_swish'] = th.nn.Identity() # XXX: don't
            # override swish.
        # Freeze BatchNorm2d
        if self.config.freeze_bn and not self.is_advtrain:
            def __freeze(mod):
                if isinstance(mod, th.nn.BatchNorm2d):
                    mod.eval()
                    mod.train = lambda _: None
            # self.backbone.apply(__freeze)
            for mod in self.backbone.modules():
                __freeze(mod)
        # validation
        self._valvecs = None
        self._vallabs = None
        # adversarial attack
        self.wantsgrad = False
        # Dump configurations
        c.print('[green]Model Meta Information[/green]', {
            'dataset': self.dataset,
            'datasestspec': self.datasetspec,
            'lossfunc': self.lossfunc,
            'metric': self.metric,
            'config': {k: v for (k, v) in self.config.__dict__.items()
                       if k not in ('allowed_losses', 'allowed_datasets')},
        })

    def forward(self, x):
        if self.wantsgrad:
            return self.forward_wantsgrad(x)
        x = x.view(-1, 3, 224, 224)  # incase of datasetspec in ('p', 't')
        with th.no_grad():
            x = utils.renorm(x)
        x = self.backbone(x)
        return x

    def forward_wantsgrad(self, x):
        x = utils.renorm(x.view(-1, 3, 224, 224))
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        optim = th.optim.Adam(
            self.backbone.parameters(),
            lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.config, 'milestones'):
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim,
                                                          milestones=self.config.milestones, gamma=0.1)
            return [optim], [scheduler]
        if hasattr(self.lossfunc, 'getOptim'):
            optim2 = self.lossfunc.getOptim()
            return optim, optim2
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

    def validation_step(self, batch, batch_idx):
        if self._valvecs is None:
            self._recompute_valvecs()
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        with th.no_grad():
            output = self.forward(images)
            r, r_1, r_2 = [], [], []
            mAP = []
            if self.metric in ('C', 'N'):
                output = th.nn.functional.normalize(output, p=2, dim=-1)
                if self.metric == 'C':
                    dists = 1 - output.mm(self._valvecs.t())
            for i in range(output.size(0)):
                if self.metric == 'C':
                    dist = dists[i]
                elif self.metric in ('E', 'N'):
                    dist = (self._valvecs - output[i]).norm(p=2, dim=-1)
                _r, _r1, _r2 = _get_rank(
                    dist, labels[i], self._vallabs, ks=[1, 2])
                r.append(_r)
                r_1.append(_r1)
                r_2.append(_r2)
                mAP.append(_get_ap(dist, labels[i], self._vallabs))
            r_1, r_2 = np.mean(r_1), np.mean(r_2)
            r = np.mean(r)
            mAP = np.mean(mAP)
        self.log('Validation/r@M', r)
        self.log('Validation/r@1', r_1)
        self.log('Validation/r@2', r_2)
        self.log('Validation/mAP', mAP)
        return {'r@M': r, 'r@1': r_1, 'r@2': r_2, 'mAP': mAP}

    def validation_epoch_end(self, outputs: list):
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    #if th.distributed.get_rank() != 0:
        #    if self.local_rank != 0:
        #        return
        # BEGIN: Calculate the rest scores
        nmi = _get_nmi(self._valvecs, self._vallabs, self.config.num_class)
        self.log('Validation/NMI', nmi)
        #   END: Calculate the rest scores
        self._valvecs = None
        self._vallabs = None
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
        if str(self._distrib_type) in (
                'DistributedType.DDP', 'DistributedType.DDP2'):
            th.distributed.barrier()
            for key in summary.keys():
                tmp = th.tensor(summary[key]).to(self.device)
                th.distributed.all_reduce(tmp, op=th.distributed.ReduceOp.SUM)
                summary[key] = tmp.item() / th.distributed.get_world_size()
        summary['NMI'] = nmi
        cprint(
            f'\nValidation │ r@M= {summary["r@M"]:.1f} ' +
            f'r@1= {summary["r@1"]:.3f} r@2= {summary["r@2"]:.3f} ' +
            f'mAP= {summary["mAP"]:.3f} ' +
            f'NMI= {summary["NMI"]:.3f}',
            'yellow')
