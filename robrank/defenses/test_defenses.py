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
import itertools as it
import torch as th
from ..losses import ptripletN
from .test_common import TestNet
from .pnp import *
from .amd import *
from .est import *
from .ses import *
import pytest

@pytest.mark.skip(reason='this is test helper')
def __test_xxx_training_step(training_step: callable):
    model = TestNet()
    images = th.rand(10, 1, 28, 28)
    labels = th.stack([th.arange(5), th.arange(5)]).T.flatten()
    loss = training_step(model, (images, labels), 0)
    loss.backward()

@pytest.mark.skip(reason='this is test helper')
def __test_hm_training_step(g, r, hm, srch, dsth, ics):
    model = TestNet()
    images = th.rand(10, 1, 28, 28)
    labels = th.stack([th.arange(5), th.arange(5)]).T.flatten()
    loss = hm_training_step(model, (images, labels), 0,
            gradual=g, fix_anchor=r, hm=hm, srch=srch, desth=dsth,
            ics=ics)
    loss.backward()

@pytest.mark.parametrize('ts', [
    pnp_training_step,
    est_training_step,
    rest_training_step,
    ses_training_step,
    ])
def test_xxx_training_step(ts: callable):
    __test_xxx_training_step(ts)

H_MAP = {'r': 'spc2-random',
         'm': 'spc2-semihard',
         's': 'spc2-softhard',
         'd': 'spc2-distance',
         'h': 'spc2-hard'}

@pytest.mark.parametrize('g, r, hm, srch, dsth, ics',
    it.product(
        (False, True),
        (False,),
        ('KL', 'L2', 'ET'),
        'rmsdh',
        'rmsdh',
        (False, True),
        ))
def test_hm_training_step(g, r, hm, srch, dsth, ics):
    __test_hm_training_step(g, r, hm, H_MAP[srch], H_MAP[dsth], ics)
