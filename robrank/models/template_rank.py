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

# pylint: disable=no-member
from packaging import version
import torch as th
import torchvision as vision
from torch.utils.data import DataLoader
import pytorch_lightning as thl
if version.parse(thl.__version__) >= version.parse("1.6.0"):
    import pytorch_lightning.strategies as thlstra
import pytorch_lightning
from pytorch_lightning.utilities.enums import DistributedType
import os
import re
import torch.nn.functional as F
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
import timm
from .. import losses
from .. import defenses
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
try:
    import pretrainedmodels as ptm
except ImportError:
    pass
#
try:
    import ujson as json
except ModuleNotFoundError:
    import json
#
import rich
c = rich.get_console()


class MetricBase(thl.LightningModule):
    '''
    This is a standard PyTorch-Lightning Module. See
    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    for the corresponding API documentation.
    '''

    _valvecs = None
    _vallabs = None

    def post_init_hook(self):
        '''
        A customizable function that should be overriden by child classes.
        This function is runed at the end of child.__init__(...)
        '''
        pass

    def _recompute_valvecs(self):
        '''
        Compute embedding vectors for the whole validation dataset,
        in order to do image retrieval and evaluate the Recall@K scores,
        etc.
        '''
        with th.no_grad():
            c.print('[yellow]\nComputing Val Set Repres ...', end=' ')
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

    def setup(self, stage=None):
        '''
        Initialize datasets (incl. training and validation), and register them
        as attributes of the current model object.
        '''
        train, test = getattr(
            datasets, self.dataset).getDataset(self.datasetspec)
        self.data_train = train
        self.data_val = test

    def train_dataloader(self):
        '''
        [after self.setup] create training dataset loader from training dataset.
        '''
        train_loader = DataLoader(self.data_train,
                                  batch_size=self.config.batchsize,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=self.config.loader_num_workers)
        return train_loader

    def val_dataloader(self):
        '''
        [after self.setup] create validation dataset loader
        '''
        val_loader = DataLoader(self.data_val,
                                batch_size=self.config.valbatchsize,
                                pin_memory=True,
                                num_workers=self.config.loader_num_workers)
        return val_loader

    def forward(self, x):
        '''
        The generic forward pass function. This forward pass function will
        be used by all different architectures, including resnet and c2f2.
        So handling inputs in different shapes is necessary.
        '''
        if any(x in self.dataset for x in ('sop', 'cub', 'cars')):
            x = x.view(-1, 3, 224, 224)
            # this is used for adversarial attack / adversarial training
            # we have to track the gradient for the very initial input
            if hasattr(self, 'wantsgrad') and self.wantsgrad:
                if hasattr(self, 'is_inceptionbn') and self.is_inceptionbn:
                    x = utils.renorm_ibn(x)
                else:
                    x = utils.renorm(x)
                x = self.backbone(x)
                return x
            else:
                # we don't want to track the gradients for the preprocessing
                # step
                with th.no_grad():
                    if hasattr(self, 'is_inceptionbn') and self.is_inceptionbn:
                        x = utils.renorm_ibn(x)
                    else:
                        x = utils.renorm(x)
                x = self.backbone(x)
                return x
        elif any(x in self.dataset for x in ('mnist', 'fashion')):
            x = x.view(-1, 1, 28, 28)
            x = self.backbone(x)
            return x
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        '''
        configuring optimizers. This is also a generic function used by
        all child classes (networks in different architectures)
        '''
        optim = getattr(th.optim, getattr(self.config, 'optimizer', 'Adam'))
        optim = optim(self.backbone.parameters(),
                      lr=self.config.lr, weight_decay=self.config.weight_decay)
        if hasattr(self.config, 'milestones'):
            scheduler = th.optim.lr_scheduler.MultiStepLR(optim,
                                                          milestones=self.config.milestones, gamma=0.1)
            return [optim], [scheduler]
        if hasattr(self.lossfunc, 'getOptim'):
            optim2 = self.lossfunc.getOptim()
            return optim, optim2
        '''
        if we draw figures, don't change the parameters at all
        '''
        if hasattr(self, 'is_advtrain_est_cosine_only') or hasattr(self,
                'is_advtrain_pnp_cosine_only') or hasattr(self,
                'is_advtrain_est_batcheff_only') or hasattr(self,
                'is_advtrain_pnp_batcheff_only'):
            optim = th.optim.SGD(self.backbone.parameters(), lr=0.0, weight_decay=0.0)
        return optim

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        '''
        Generic training step function shared between network models in
        different architectures. Be careful if you want to override this
        in any child class.

        Manual optimization process may make this function look a little
        bit of complicated:
        https://pytorch-lightning.readthedocs.io/en/stable/common/optimizers.html#manual-optimization
        '''
        # Dispatcher for different defense methods, if specified.
        # When no defense is specified, we fallback to regular training.
        if getattr(self, 'is_advtrain', False):
            # not recommended to use the ambiguous attribute.
            # will be deprecated in the future.
            return defenses.est_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_est', False):
            return defenses.est_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_est_cosine_only', False):
            return defenses.est_training_step_cosine_only(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_est_batcheff_only', False):
            return defenses.est_training_step_cosine_only(self, batch, batch_idx, do_batcheff=True)
        elif getattr(self, 'is_advtrain_estf', False):
            return defenses.est_training_step(
                self, batch, batch_idx, pgditer=1)
        elif getattr(self, 'is_advtrain_ses', False):
            return defenses.ses_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_pnp', False):
            use_barlow_twins = getattr(self, 'use_barlow_twins', False)
            if use_barlow_twins:
                c.print('[white on red]>_< Barlow-Twins is enabled.')
            return defenses.pnp_training_step(self, batch, batch_idx,
                            use_barlow_twins=use_barlow_twins)
        elif getattr(self, 'is_advtrain_pnp_cosine_only', False):
            return defenses.pnp_training_step_cosine_only(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_pnp_batcheff_only', False):
            return defenses.pnp_training_step_cosine_only(self, batch, batch_idx, do_batcheff=True)
        elif getattr(self, 'is_advtrain_pnpf', False):
            return defenses.pnp_training_step(
                self, batch, batch_idx, pgditer=1)
        elif getattr(self, 'is_advtrain_pnp_adapt', False):
            return defenses.pnp_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_pnpx', False):
            '''
            Benign + adversarial training mode (PNP/Augment, postfix=px)
            '''
            if np.random.random() > 0.5:
                return defenses.pnp_training_step(self, batch, batch_idx)
            else:
                pass  # do the normal training step
        elif getattr(self, 'is_advtrain_mmt', False):
            return defenses.mmt_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_tbc', False):
            return defenses.tbc_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_acap', False):
            return defenses.acap_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_rest', False):
            return defenses.rest_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_amd', False):
            return defenses.amd_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_ramd', False):
            return defenses.ramd_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_amdsemi', False):
            return defenses.amdsemi_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_amdsemiaap', False):
            return defenses.amdsemi_training_step(
                self, batch, batch_idx, aap=True)
        elif getattr(self, 'is_advtrain_amdsemiact', False):
            '''
            Mixed training with AMD Semi + ACT training
            '''
            if np.random.random() > 0.5:
                return defenses.pnp_training_step(self, batch, batch_idx)
            else:
                return defenses.amdsemi_training_step(self, batch, batch_idx)
        elif getattr(self, 'is_freeat_none', False):
            return defenses.none_freeat_step(self, batch, batch_idx)
        elif getattr(self, 'is_freeat_amd', False):
            return defenses.amd_freeat_step(self, batch, batch_idx)
        elif getattr(self, 'is_freeat_amdsemi', False):
            return defenses.amdsemi_freeat_step(self, batch, batch_idx)
        elif getattr(self, 'is_advtrain_hm', False):
            if not hasattr(self, 'hm_spec'):
                raise ValueError('''you should register a dictionary 'hm_spec'
                        as attribute of a model with several keys defined
                        in the following code in the .py file. Or just
                        generate a default set of configurations with
                        robrank/models/autogen/autogen.py''')
            use_barlow_twins = getattr(self, 'use_barlow_twins', False)
            if use_barlow_twins:
                c.print('[white on red]>_< Barlow-Twins is enabled.')
            return defenses.hm_training_step(self, batch, batch_idx,
                                             srch=self.hm_spec['srch'],
                                             desth=self.hm_spec['desth'],
                                             hm=self.hm_spec['hm'],
                                             gradual=self.hm_spec['gradual'],
                                             ics=self.hm_spec['ics'],
                                             fix_anchor=self.hm_spec['fix_anchor'],
                                             use_barlow_twins=use_barlow_twins,
                                             )
        elif getattr(self, 'is_advtrain_hmix', False):
            if np.random.random() > 0.5:
                return defenses.hm_training_step(self, batch, batch_idx,
                        srch='spc2-random', desth='spc2-semihard',
                        hm='ET', gradual=False, ics=False, fix_anchor=False)
            else:
                return defenses.hm_training_step(self, batch, batch_idx,
                        srch='spc2-softhard', desth='spc2-semihard',
                        hm='ET', gradual=False, ics=False, fix_anchor=False)
        else:
            pass
        # else: normal training.
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

    def training_epoch_end(self, training_step_outputs):
        '''
        dump cosine stat for tpami figure 2.
        '''
        # process flags
        dump_cosine_stat = False
        dump_batcheff_stat = False
        if getattr(self, 'is_advtrain_pnp_cosine_only', False):
            dump_cosine_stat = True
        elif getattr(self, 'is_advtrain_est_cosine_only', False):
            dump_cosine_stat = True
        elif getattr(self, 'is_advtrain_pnp_batcheff_only', False):
            dump_batcheff_stat = True
        elif getattr(self, 'is_advtrain_est_batcheff_only', False):
            dump_batcheff_stat = True
        # cosine only stat
        if dump_cosine_stat:
            fname = 'cosine_only_stat.json'
            with open(fname, 'wt') as f:
                json.dump(self.cosine_only_stat, f)
            print(f'>_< self.cosine_only_stat has been dumped into {fname}')
        if dump_batcheff_stat:
            fname = 'batcheff_only_stat.json'
            with open(fname, 'wt') as f:
                # this is not bug. we re-use the list cosine_only_stat
                json.dump(self.cosine_only_stat, f)
            print(f'>_< self.batcheff_only_stat has been dumped into {fname}')
        # just exit after one epoch
        if dump_cosine_stat or dump_batcheff_stat:
            exit()


    def validation_step(self, batch, batch_idx):
        '''
        Generic validation step for all ranking / dml models.
        This will be ran in data parallel mode if we have multiple GPUs.
        Before validation a batch, we have to first recalculate
        the embedding vectors for the whole validation dataset.
        See self._recompute_valvecs(...).
        '''
        #print('[', th.distributed.get_rank(), ']', batch_idx, '\n')
        if self._valvecs is None:
            self._recompute_valvecs()
        images, labels = (batch[0].to(self.device), batch[1].to(self.device))
        with th.no_grad():
            # calculate pairwise distance
            output = self.forward(images)
            if self.metric in ('C', 'N'):
                output = F.normalize(output, p=2, dim=-1)
            if self.metric == 'C':
                dist = 1 - th.mm(output, self._valvecs.t())
            elif self.metric in ('E', 'N'):
                dist = th.cdist(output, self._valvecs)
            # metrics
            r, r_1, r_2, mAP, mAPR = [], [], [], [], []
            for i in range(output.size(0)):
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
                mAPR.append(
                    *utils.metric_get_ap_r(dist[i], labels[i],
                                           self._vallabs, rs=[10]))
            r, r_1, r_2 = np.mean(r), np.mean(r_1), np.mean(r_2)
            mAP = np.mean(mAP)
            mAPR = np.mean(mAPR)
        return {'r@M': r, 'r@1': r_1, 'r@2': r_2, 'mAP': mAP, 'mAP@R': mAPR}

    def validation_epoch_end(self, outputs: list):
        '''
        Aggregate and summarize the validation results from multiple GPUs
        and multiple validation iterations.
        '''
        # only the process of rank 0 has to report this summary.
        # TODO: figure out why this part results in deadlock / gets stuck
        # if str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
        #    #if th.distributed.get_rank() != 0:
        #    if self.local_rank != 0:
        #        return

        # reduce:mean the summary of every process
        summary = {key: np.mean(tuple(
            x[key] for x in outputs)) for key in outputs[0].keys()}
        if version.parse(thl.__version__) >= version.parse('1.6.0') and \
                hasattr(self.trainer, 'strategy') and \
                isinstance(self.trainer.strategy, thlstra.DDPStrategy):
            '''
            This branch is for pytorch-lightining >= 1.6.0
            '''
            #th.distributed.barrier()
            for key in summary.keys():
                tmp = th.tensor(summary[key]).to(self.device)
                th.distributed.all_reduce(tmp, op=th.distributed.ReduceOp.SUM)
                summary[key] = tmp.item() / th.distributed.get_world_size()
            #th.distributed.barrier()  # rank specific operation needs a barrier
            #if th.distributed.get_rank() != 0:
            #    return None
        elif hasattr(self, '_distrib_type') and \
                str(self._distrib_type) in ('DistributedType.DDP', 'DistributedType.DDP2'):
            '''
            This branch is for pytorch-lightining < 1.6.0, e.g. 1.5.9
            '''
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
        self.log('Validation/mAP@R', summary['mAP@R'])
        c.print(f'\nValidation │ ' +
                f'r@M= {summary["r@M"]:.1f} ' +
                f'r@1= {summary["r@1"]:.3f} ' +
                f'r@2= {summary["r@2"]:.3f} ' +
                f'mAP= {summary["mAP"]:.3f} ' +
                f'mAP@R = {summary["mAP@R"]:.3f} ' +
                f'NMI= {summary["NMI"]:.3f}')

###############################################################################


class MetricTemplate28(MetricBase):
    '''
    Deep Metric Learning with MNIST-compatible Network.
    '''
    is_advtrain = False
    do_svd = False
    BACKBONE = 'rc2f2'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in getattr(configs, self.BACKBONE).allowed_datasets)
        assert(loss in getattr(configs, self.BACKBONE).allowed_losses)
        self.dataset = dataset
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()
        # configuration
        self.config = getattr(configs, self.BACKBONE)(dataset, loss)
        # modules
        if self.BACKBONE == 'rc2f2':
            '''
            A 2-Conv Layer 1-FC Layer Network For Ranking
            See [Madry, advrank] for reference.
            '''
            self.backbone = th.nn.Sequential(
                th.nn.Conv2d(1, 32, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(32, 64, kernel_size=5, padding=2),
                th.nn.ReLU(),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(64 * 7 * 7, 1024),
                th.nn.ReLU(),
                th.nn.Linear(1024, self.config.embedding_dim)
            )
        elif self.BACKBONE == 'rlenet':
            self.backbone = th.nn.Sequential(
                th.nn.Conv2d(1, 20, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Conv2d(20, 50, kernel_size=5, stride=1),
                th.nn.MaxPool2d(kernel_size=2, stride=2),
                th.nn.Flatten(),
                th.nn.Linear(800, 500),
                th.nn.ReLU(),
                th.nn.Linear(500, self.config.embedding_dim),
            )
        else:
            raise ValueError('unknown backbone')
        # summary
        c.print('[green]Model Meta Information[/green]', {
            'dataset': self.dataset,
            'datasestspec': self.datasetspec,
            'lossfunc': self.lossfunc,
            'metric': self.metric,
            'config': {k: v for (k, v) in self.config.__dict__.items()
                       if k not in ('allowed_losses', 'allowed_datasets')},
        })
        self.post_init_hook()


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
    is_inceptionbn = False

    def __create_config_backbone(self, name: str, dataset: str, loss: str):
        '''
        input: specified backbone name
        result: populate self.config and self.backbone instances
        '''
        if name == 'rres18':
            self.config = configs.rres18(dataset, loss)
            self.backbone = vision.models.resnet18(pretrained=True)
        elif name == 'rres50':
            self.config = configs.rres50(dataset, loss)
            self.backbone = vision.models.resnet50(pretrained=True)
        elif name == 'rres101':
            self.config = configs.rres50(dataset, loss)
            self.backbone = vision.models.resnet101(pretrained=True)
        elif name == 'rres152':
            self.config = configs.rres50(dataset, loss)
            self.backbone = vision.models.resnet152(pretrained=True)
        elif name == 'rmnas05':
            self.config = configs.rmnas(dataset, loss)
            self.backbone = vision.models.mnasnet0_5(pretrained=True)
        elif name == 'rmnas10':
            self.config = configs.rmnas(dataset, loss)
            self.backbone = vision.models.mnasnet1_0(pretrained=True)
        elif name == 'rmnas13':
            self.config = configs.rmnas(dataset, loss)
            self.backbone = vision.models.mnasnet1_3(pretrained=True)
        elif name == 'reffb0':
            self.config = configs.reffb0(dataset, loss)
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif name == 'reffb4':
            self.config = configs.reffb4(dataset, loss)
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        elif name == 'ribn':
            self.config = configs.ribn(dataset, loss)
            self.backbone = ptm.__dict__['bninception'](
                    num_classes=1000, pretrained='imagenet')
        elif name == 'rswint':
            # SwinT-Tiny 224 ImageNet1k
            self.config = configs.rswint(dataset, loss)
            self.backbone = timm.create_model(
                    'swin_tiny_patch4_window7_224', pretrained=True)
        elif name == 'rswins':
            self.config = configs.rswint(dataset, loss)
            self.backbone = timm.create_model(
                    'swin_small_patch4_window7_224', pretrained=True)
        elif name == 'rswinb':
            self.config = configs.rswint(dataset, loss)
            self.backbone = timm.create_model(
                    'swin_base_patch4_window7_224', pretrained=True)
        elif name == 'rswinl':
            self.config = configs.rswint(dataset, loss)
            self.backbone = timm.create_model(
                    'swin_large_patch4_window7_224', pretrained=True)
        else:
            raise ValueError(f'unrecognized backbone {name}')


    def __perform_backbone_surgery(self, name: str):
        '''
        perform surgery for deep metric learning
        because the networks are originally for classification

        needs self.config to be populated
        '''
        if re.match(r'rres.*', name):
            emb_dim = 512 if '18' in name else 2048
            if self.config.embedding_dim > 0:
                self.backbone.fc = th.nn.Linear(
                    emb_dim, self.config.embedding_dim)
            else:
                self.backbone.fc = th.nn.Identity()
        elif re.match(r'rmnas.*', name):
            if self.config.embedding_dim > 0:
                self.backbone.classifier = th.nn.Linear(
                    1280, self.config.embedding_dim)
            else:
                self.backbone.classifier = th.nn.Identity()
        elif re.match(r'reff.*', name):
            if self.config.embedding_dim > 0:
                if 'b0' in name:
                    emb_dim = 1280
                elif 'b7' in name:
                    emb_dim = 2560
                self.backbone._modules['_dropout'] = th.nn.Identity()
                self.backbone._modules['_fc'] = th.nn.Linear(
                    emb_dim, self.config.embedding_dim)
                # note: don't override swish.
                # self.backbone._modules['_swish'] = th.nn.Identity()
            else:
                self.backbone._modules['_dropout'] = th.nn.Identity()
                self.backbone._modules['_fc'] = th.nn.Identity()
        elif re.match(r'ribn.*', name):
            assert(self.config.embedding_dim > 0)
            self.backbone.global_pool = th.nn.AdaptiveAvgPool2d(1)
            self.backbone.last_linear = th.nn.Linear(
                self.backbone.last_linear.in_features,
                self.config.embedding_dim)
        elif re.match(r'rswin.*', name):
            # swin transformer family
            assert(self.config.embedding_dim > 0)
            self.backbone.head = th.nn.Linear(self.backbone.head.in_features,
                    self.config.embedding_dim)
        else:
            raise NotImplementedError('how to perform surgery for such net?')
        # Freeze BatchNorm2d (ICML20: revisiting ... in DML)
        if self.config.freeze_bn and not self.is_advtrain:
            def __freeze(mod):
                if isinstance(mod, th.nn.BatchNorm2d):
                    mod.eval()
                    mod.train = lambda _: None
            # self.backbone.apply(__freeze)
            for mod in self.backbone.modules():
                __freeze(mod)
        # directly remove BatchNorm2d instead
        if hasattr(self, 'remove_bn') and self.remove_bn:
            # Removing Batch Normalization Boosts Adversarial Training
            def __remove_bn(mod):
                if isinstance(mod, th.nn.BatchNorm2d):
                    mod = th.nn.Identity()
            for mod in self.backbone.modules():
                __remove_bn(mod)


    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        self.__create_config_backbone(self.BACKBONE, dataset, loss)

        # add attributes
        assert(dataset in self.config.allowed_datasets)
        self.dataset = dataset
        assert(loss in self.config.allowed_losses)
        self.loss = loss
        self.lossfunc = getattr(losses, loss)()
        self.metric = self.lossfunc.determine_metric()
        self.datasetspec = self.lossfunc.datasetspec()

        # perform surgery for deep metric learning
        self.__perform_backbone_surgery(self.BACKBONE)

        # for adversarial attack
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
        self.post_init_hook()
