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
import re
from .advclass import *
import rich
c = rich.get_console()

_LEGAL_ATTACKS_ = ('PGD', )


class AdvClassLauncher(object):
    '''
    Entrance class for classification attack
    '''

    def __init__(self, attack: str, device: str = 'cpu',
                 verbose: bool = False):
        self.device = device
        self.verbose = verbose
        self.kw = {}
        # parse the attack
        self.kw['device'] = device
        self.kw['verbose'] = verbose
        attack_type, atk_arg = re.match(r'(\w+?):(.*)', attack).groups()
        self.kw['attack_type'] = attack_type
        self.kw.update(dict(re.findall(r'(\w+)=([\-\+\.\w]+)', atk_arg)))
        # sanity check
        assert(attack_type in _LEGAL_ATTACKS_)
        for key in ('eps', 'alpha'):
            if key in self.kw:
                self.kw[key] = float(self.kw[key])
        for key in ('pgditer', ):
            if key in self.kw:
                self.kw[key] = int(self.kw[key])
        print('* Attack', self.kw)

    def __call__(self, model: object, loader: object, *, maxiter: int = None):
        '''
        The model should be a classification model.
        '''
        model.eval()
        Sorig, Sadv = [], []
        for N, (images, labels) in tqdm(enumerate(loader)):
            if maxiter is not None and N >= maxiter:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            xr, r, sorig, sadv = projGradDescent(model, images, labels,
                    **self.kw)
            Sorig.append(sorig)
            Sadv.append(sadv)
        return Sorig, Sadv
