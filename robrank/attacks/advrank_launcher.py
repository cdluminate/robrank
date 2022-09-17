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
import os
import re
import json
from tqdm import tqdm
import numpy as np
import torch as th
import torchvision as V
from .advrank import AdvRank
from ..utils import rjson
import rich
c = rich.get_console()

_LEGAL_ATTAKS_ = ('ES', 'QA', 'CA', 'SPQA', 'GTM', 'GTT', 'TMA', 'LTM')


class AdvRankLauncher(object):
    '''
    Entrace Class for adversarial ranking attack [ArXiv:2002.11293]

    attack: str - describing the attack type
    device: str - torch device description
    dumpaxd: str - path to the directory for dumped adversarial examples
    verbose: bool - print additional information
    '''

    def __init__(self, attack: str, device: str = 'cpu',
                 dumpaxd: str = '',
                 verbose: bool = False,
                 *,
                 nes_mode: bool = False):
        self.device = device
        self.verbose = verbose
        self.kw = {}
        self.dumpaxd: str = dumpaxd
        self.dumpax_counter: int = 0
        self.nes_mode : bool = nes_mode
        # parse the attack
        self.kw['device'] = device
        self.kw['verbose'] = verbose
        attack_type, atk_arg = re.match(r'(\w+?):(.*)', attack).groups()
        self.kw['attack_type'] = attack_type
        self.kw.update(dict(re.findall(r'(\w+)=([\-\+\.\w]+)', atk_arg)))
        # sanity check and type conversion
        print('* Attack', self.kw)
        assert(attack_type in _LEGAL_ATTAKS_)
        if attack_type == 'ES':
            pass
        elif attack_type in ('GTM', 'GTT', 'TMA', 'LTM'):
            pass
        elif attack_type == 'CA':
            assert('W' in self.kw)
            assert('pm' in self.kw)
        elif attack_type in ('QA', 'SPQA'):
            assert('M' in self.kw)
            assert('pm' in self.kw)
        for key in ('eps', 'alpha'):
            if key in self.kw:
                self.kw[key] = float(self.kw[key])
        for key in ('pgditer', 'W', 'M'):
            if key in self.kw:
                self.kw[key] = int(self.kw[key])

    def __call__(self, model: object, loader: object, *, maxiter: int = None):
        '''
        The model should be a ranking model. required methods:
            model._recompute_valvecs
        '''
        # pylint: disable=unused-variable
        # calculate the embeddings for the whole validation set
        model.eval()
        candi = model._recompute_valvecs()
        # XXX: this is tricky, but we need it.
        model.wantsgrad = True
        print('\n* Validation Embeddings', candi[0].shape,
              'Labels', candi[1].shape)

        # initialize attacker
        advrank = AdvRank(model, metric=model.metric, **self.kw)
        if self.nes_mode:
            advrank.set_mode('NES')

        # let's start!
        Sumorig, Sumadv = [], []
        iterator = tqdm(enumerate(loader),
                        total=len(loader) if maxiter is None else maxiter)
        for N, (images, labels) in iterator:
            if maxiter is not None and N >= maxiter:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            xr, r, sumorig, sumadv = advrank(images, labels, candi)
            Sumorig.append(sumorig)
            Sumadv.append(sumadv)
            if self.dumpaxd:
                if not os.path.exists(self.dumpaxd):
                    os.mkdir(self.dumpaxd)
                #oxpath = os.path.join(self.dumpaxd, f'ox-{self.dumpax_counter:07d}.pth')
                #axpath = os.path.join(self.dumpaxd, f'ax-{self.dumpax_counter:07d}.pth')
                #th.save(xr, axpath)

                if len(xr.shape) == 4:
                    for Ni in range(xr.size(0)):
                        oxpath = os.path.join(self.dumpaxd,
                                              f'ox-{self.dumpax_counter:07d}.jpg')
                        axpath = os.path.join(self.dumpaxd,
                                              f'ax-{self.dumpax_counter:07d}.jpg')
                        V.utils.save_image(images[Ni, :, :, :].squeeze().cpu(),
                                           oxpath)
                        V.utils.save_image(xr[Ni, :, :, :].squeeze().cpu(),
                                           axpath)
                        self.dumpax_counter += 1
                else:
                    raise NotImplementedError

        # let's summarize!
        c.print('[yellow]=== Summary ========================================')
        c.print('[bold green]Original Example:')
        sorig = {key: np.mean([y[key] for y in Sumorig])
                 for key in Sumorig[0].keys()}
        print(rjson(json.dumps(sorig, indent=2)))
        c.print('[bold red]Adversarial Example:')
        sadv = {key: np.mean([y[key] for y in Sumadv])
                for key in Sumadv[0].keys()}
        print(rjson(json.dumps(sadv, indent=2)))
        c.print('[bold yello]Difference:')
        ckeys = set(sorig.keys()).intersection(set(sadv.keys()))
        diff = {key: np.mean([y[key] for y in Sumadv]) -
                np.mean([y[key] for y in Sumorig]) for key in ckeys}
        print(rjson(json.dumps(diff, indent=2)))
        c.print('[yellow]====================================================')
        return (sorig, sadv)
