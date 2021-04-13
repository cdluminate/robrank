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

from termcolor import cprint, colored
from torchvision import transforms
import argparse
import os
import pytorch_lightning as thl
import re
import robrank as rr
import sys
import torch as th
import torch.utils.data
import torchvision as V
import torchvision as vision
import gc
import psutil
import json
import itertools as it
import glob
import rich
c = rich.get_console()


class TFdump:
    '''
    Dump a tensorboard tfevent binary file
    '''

    def __init__(self, *args):
        ag = argparse.ArgumentParser()
        ag.add_argument('-f', '--file', type=str, required=True)
        ag = ag.parse_args(*args)

        from tensorflow.python.summary.summary_iterator import summary_iterator
        for s in summary_iterator(ag.file):
            # print(s)
            if len(s.summary.value) > 0:
                if 'Valid' in s.summary.value[0].tag:
                    cprint(s.summary.value[0].tag, 'red', end=' ')
                    cprint(f'{s.step}', 'yellow', end=' ')
                    cprint(s.summary.value[0].simple_value, 'blue')


class Swipe:
    '''
    Conduct a batch of advrank attack
    it stucks if we do not kill the child processes
    '''
    profile_eccv20_mnist = (
        #'ES:eps=0.3:alpha=(math 2/255):pgditer=32',
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'),
            ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.1:alpha=0.01:pgditer=20',
                'eps=0.3:alpha=0.01:pgditer=30',
            ))],
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'),
            ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.1:alpha=0.01:pgditer=20',
                'eps=0.3:alpha=0.01:pgditer=30',
            ))],
    )
    profile_pami_mnist = (
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'), ('M=10', 'M=1'),
            (
                'eps=0.007843:alpha=0.003922:pgditer=24',
                'eps=0.300000:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=10', 'W=1'),
            (
                'eps=0.007843:alpha=0.003922:pgditer=24',
                'eps=0.300000:alpha=0.011764:pgditer=32',
            ))],
    )
    profile_eccv20_imagenet = (
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'),
            ('W=1', 'W=2', 'W=5', 'W=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=24',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.06:alpha=0.006:pgditer=20',
            ))],
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'),
            ('M=1', 'M=2', 'M=5', 'M=10'),
            (
                'eps=0.01:alpha=0.003922:pgditer=10',
                'eps=0.03:alpha=0.003922:pgditer=15',
                'eps=0.06:alpha=0.006:pgditer=20',
            ))],
    )
    profile_pami_imagenet = (
        *[':'.join(x) for x in it.product(
            ('SPQA',), ('pm=+', 'pm=-'), ('M=10', 'M=1'),
            (
                'eps=0.007843:alpha=0.003922:pgditer=24',
                'eps=0.062745:alpha=0.011764:pgditer=32',
            ))],
        *[':'.join(x) for x in it.product(
            ('CA',), ('pm=+', 'pm=-'), ('W=10', 'W=1'),
            (
                'eps=0.007843:alpha=0.003922:pgditer=24',
                'eps=0.062745:alpha=0.011764:pgditer=32',
            ))],
    )

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        g = ag.add_mutually_exclusive_group(required=True)
        g.add_argument('-C', '--checkpoint', type=str, default=None,
                       help='example: logs_mnist-c2f1-ptripletE/lightning_logs/'
                       + 'version_0/checkpoints/epoch=7.ckpt')
        g.add_argument('-c', '--checkpointdir', type=str, default=None,
                       help='example: logs_mnist-c2f1-ptripletE')
        ag.add_argument('-D', '--device', type=str, default='cuda'
                        if th.cuda.is_available() else 'cpu')
        ag.add_argument('-p', '--profile', type=str, required=True,
                        choices=('pami_mnist', 'pami_imagenet',
                                 'eccv20_mnist', 'eccv20_imagenet'))
        ag.add_argument('-b', '--batchsize', type=int, default=-1)
        ag.add_argument('-m', '--maxiter', type=int, default=None)
        ag.add_argument('-v', '--verbose', action='store_true')
        ag = ag.parse_args(argv)
        print(ag)
        profile = getattr(self, 'profile_' + ag.profile)
        print(profile)

        # shorthand usage
        if ag.checkpointdir is not None:
            path = os.path.join(ag.checkpointdir, 'lightning_logs/version_*')
            ndir = rr.utils.nsort(glob.glob(path), r'.*version_(\d+)')[0]
            path = os.path.join(ndir, 'checkpoints/epoch=*')
            nchk = rr.utils.nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
            cprint(
                f'* Automatically discovered the latest checkpoint .. {nchk}',
                'cyan')
            ag.checkpoint = nchk

        instances = {}
        for atk in profile:
            argv = ['-D', ag.device, '-C', ag.checkpoint, '-A', atk]
            if ag.batchsize > 0:
                argv.extend(['-b', str(ag.batchsize)])
            if ag.maxiter is not None:
                argv.extend(['-m', str(ag.maxiter)])
            if ag.verbose:
                argv.append('-v')
            print('Calling AdvRank with', argv)
            instance = AdvRank(argv)
            instances[atk] = instance.stats
            del instance
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
            gc.collect()
        results = {key: val for (key, val) in instances.items()}
        cprint('=== Final Swipe Results ====================================',
               'white', 'on_red')
        code = json.dumps(results, indent=2)
        print(rr.utils.rjson(code))
        with open(ag.checkpoint + f'.{ag.profile}.json', 'wt') as f:
            f.write(code)


class AdvRank:
    '''
    Conduct adversarial ranking attack
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--checkpoint', type=str, required=True,
                        help='example: logs_mnist-c2f1-ptripletE/lightning_logs/'
                        + 'version_0/checkpoints/epoch=7.ckpt')
        ag.add_argument('-A', '--attack', type=str, required=True)
        ag.add_argument('-D', '--device', type=str, default='cuda'
                        if th.cuda.is_available() else 'cpu')
        ag.add_argument('-v', '--verbose', action='store_true')
        ag.add_argument('-m', '--maxiter', type=int, default=None)
        ag.add_argument('-b', '--batchsize', type=int, default=-1,
                        help='override batchsize')
        ag = ag.parse_args(argv)
        ag.dataset, ag.model, ag.loss = re.match(
            r'logs_(\w+)-(\w+)-(\w+)/.*\.ckpt', ag.checkpoint).groups()
        c.print(rich.panel.Panel(' '.join(argv), title='RobRank::AdvRank',
                                 style='bold magenta'))
        c.print(vars(ag))

        c.rule('[white on magenta]>_< Restoring Model from Checkpoint ...')
        model = getattr(rr.models, ag.model).Model.load_from_checkpoint(
            checkpoint_path=ag.checkpoint,
            dataset=ag.dataset, loss=ag.loss)
        model = model.to(ag.device)
        if ag.batchsize > 0:
            model.config.valbatchsize = ag.batchsize

        c.rule('[white on magenta]>_< Initializing Attack Launcher ...')
        atker = rr.attacks.AdvRankLauncher(ag.attack, ag.device, ag.verbose)
        print(atker)

        c.rule('[white on magenta]>_< Getting Validation Loader ...')
        model.setup()
        val_dataloader = model.val_dataloader()
        sorig, sadv = atker(model, val_dataloader, maxiter=ag.maxiter)
        self.stats = (sorig, sadv)


class Validate:
    '''
    Validate a trained model
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--checkpoint', type=str, default=None)
        ag.add_argument(
            '-g',
            '--gpus',
            type=int,
            default=th.cuda.device_count())
        ag = ag.parse_args(argv)
        ag.dataset, ag.model, ag.loss = re.match(
            r'logs_(\w+)-(\w+)-(\w+)/', ag.checkpoint).groups()
        print(vars(ag))
        model = getattr(rr.models, ag.model).Model(
            dataset=ag.dataset, loss=ag.loss)
        #model.load_from_checkpoint(ag.checkpoint, dataset=ag.dataset, loss=ag.loss)
        trainer = thl.Trainer(gpus=ag.gpus,
                              num_sanity_val_steps=-1,
                              resume_from_checkpoint=ag.checkpoint)
        cprint('>_< Start Validating ...', 'white', 'on_magenta')
        trainer.fit(model)
        cprint('>_< Pulling Down ...', 'white', 'on_red')


class Train:
    '''
    Train a ranking model
    '''

    def __init__(self, argv):
        ag = argparse.ArgumentParser()
        ag.add_argument('-C', '--config', type=str, required=True,
                        help='example: "sop:res18:ptripletE".')
        ag.add_argument('-g', '--gpus', type=int, default=th.cuda.device_count(),
                        help='number of GPUs to use')
        ag.add_argument('--dp', action='store_true',
                        help='use th.nn.DataParallel instead of distributed.')
        ag.add_argument('--do_test', action='store_true')
        ag.add_argument('-m', '--monitor', type=str, default='Validation/r@1')
        ag.add_argument('-r', '--resume', action='store_true')
        ag.add_argument('--svd', action='store_true')
        ag = ag.parse_args(argv)
        c.print(rich.panel.Panel(' '.join(argv), title='RobRank::Train',
                                 style='bold magenta'))
        c.print(vars(ag))
        ag.dataset, ag.model, ag.loss = re.match(
            r'(\w+):(\w+):(\w+)', ag.config).groups()

        # find the latest checkpoint
        if ag.resume:
            checkpointdir = 'logs_' + re.sub(r':', '-', ag.config)
            path = os.path.join(checkpointdir, 'lightning_logs/version_*')
            ndir = rr.utils.nsort(glob.glob(path), r'.*version_(\d+)')[0]
            path = os.path.join(ndir, 'checkpoints/epoch=*')
            nchk = rr.utils.nsort(glob.glob(path), r'.*epoch=(\d+)')[0]
            cprint(f'* Discovered the latest ckpt .. {nchk}', 'cyan')
            ag.checkpoint = nchk

        c.rule('[white on magenta]>_< Initializing Model & Arguments ...')
        model = getattr(rr.models, ag.model).Model(
            dataset=ag.dataset, loss=ag.loss)

        # experimental features
        if ag.svd:
            model.do_svd = True

        c.rule('[white on magenta]>_< Initializing Optimizer ...')
        if ag.dp:
            accel = 'dp'
        elif ag.gpus > 1:
            accel = 'ddp'
        else:
            accel = None
        # checkpoint_callback = thl.callbacks.ModelCheckpoint(
        #        monitor=ag.monitor,
        #        mode='max')
        trainer = thl.Trainer(
            max_epochs=model.config.maxepoch,
            gpus=ag.gpus,
            log_every_n_steps=1,
            val_check_interval=1.0,
            check_val_every_n_epoch=model.config.validate_every,
            default_root_dir='logs_' + re.sub(r':', '-', ag.config),
            accelerator=accel,
            resume_from_checkpoint=ag.checkpoint if ag.resume else None,
        )
        # checkpoint_callback=checkpoint_callback)
        # print(checkpoint_callback.best_model_path)

        c.rule('[white on magenta]>_< Start Training ...')
        trainer.fit(model)
        if ag.do_test:
            trainer.test(model)

        cprint('>_< Pulling Down ...', 'white', 'on_red')


class Download:
    '''
    Download MNIST and Fashion-MNIST datasets
    '''

    def __init__(self):
        cprint('>_< MNIST', 'white', 'on_blue')
        V.datasets.MNIST('~/.torch/', download=True)

        cprint('>_< FashionMNIST', 'white', 'on_blue')
        V.datasets.FashionMNIST('~/.torch/', download=True)

        cprint('>_< Done!', 'white', 'on_green')
