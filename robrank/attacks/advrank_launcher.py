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

import os
import sys
import re
import json
import functools
import torch as th
import collections
from tqdm import tqdm
import pylab as lab
import traceback
import math
import statistics
from scipy import stats
import numpy as np
import random
from .advrank_qcselector import QCSelector
from .advrank import AdvRank
from termcolor import cprint
from ..utils import rjson


class AdvRankLauncher(object):
    '''
    Entrace Class for adversarial ranking attack [ArXiv:2002.11293]
    '''

    def __init__(self, attack: str, device: str = 'cpu',
                 verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.kw = {}
        # parse the attack
        self.kw['device'] = device
        self.kw['verbose'] = verbose
        attack_type, tmp = re.match(r'(\w+?):(.*)', attack).groups()
        self.kw['attack_type'] = attack_type
        self.kw.update(dict(re.findall(r'(\w+)=([\-\+\.\w]+)', tmp)))
        # sanity check and type conversion
        print('* Attack', self.kw)
        assert(attack_type in ('ES', 'QA', 'CA', 'SPQA'))
        if attack_type == 'ES':
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
        # calculate the embeddings for the whole validation set
        model.eval()
        candi = model._recompute_valvecs()
        # XXX: this is tricky, but we need it.
        model.wantsgrad = True
        print()
        print(
            '* Validation Embeddings',
            candi[0].shape,
            'Labels',
            candi[1].shape)

        # initialize attacker
        advrank = AdvRank(model, metric=model.metric, **self.kw)

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

        # let's summarize!
        cprint(
            '=== Summary ==============================================',
            'yellow')
        cprint('Original Example:', 'green', None, ['bold'])
        sorig = {key: np.mean([y[key] for y in Sumorig])
                 for key in Sumorig[0].keys()}
        print(rjson(json.dumps(sorig, indent=2)))
        cprint('Adversarial Example:', 'red', None, ['bold'])
        sadv = {key: np.mean([y[key] for y in Sumadv])
                for key in Sumadv[0].keys()}
        print(rjson(json.dumps(sadv, indent=2)))
        cprint('Difference:', 'yellow', None, ['bold'])
        ckeys = set(sorig.keys()).intersection(set(sadv.keys()))
        diff = {key: np.mean([y[key] for y in Sumadv]) -
                np.mean([y[key] for y in Sumorig]) for key in ckeys}
        print(rjson(json.dumps(diff, indent=2)))
        cprint(
            '==========================================================',
            'yellow')
        return (sorig, sadv)
