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
from ..losses import ptripletN
from . import test_common
from .pnp import *
from .amd import *
from .est import *
from .ses import *
import pytest


@pytest.mark.parametrize('ts', [
    pnp_training_step,
    est_training_step,
    rest_training_step,
    ses_training_step,
    ])
def test_xxx_training_step(ts: callable):
    return test_common.test_xxx_training_step(ts)
