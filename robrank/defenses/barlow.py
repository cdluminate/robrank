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

__doc__ = '''
Barlow Twins: Self-Supervised Learning via Redundancy Reduction
https://arxiv.org/pdf/2103.03230.pdf
Let's see if this helps reducing collapse issue a little bit
'''

import torch as th
import torch.nn.functional as F
import rich
console = rich.get_console()


def _off_diagonal(matrix: th.Tensor):
    '''
    https://discuss.pytorch.org/t/most-efficient-way-to-get-just-the-off-diagonal-elements-of-a-tensor/131065
    '''
    assert matrix.dim() == 2
    res = matrix.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res


def _diagonal(matrix: th.Tensor):
    assert matrix.dim() == 2
    return matrix.diagonal(dim1=-1, dim2=-2)


def _barlow_twins(z_a: th.Tensor, z_b: th.Tensor, *, lam: float = 5e-3):
    '''
    computation kernel... the most core operations
    we follow the pseudo-code of https://arxiv.org/pdf/2103.03230.pdf
    input:
      z_a: th.Tensor shape(N, D)
      z_b: th.Tensor shape(N, D)
    output:
      loss: th.Tensor shape(,) scalar
    parameter:
      lambda: 5e-3 is the param setting in the paper.
    '''
    assert z_a.shape == z_b.shape
    N, D = z_a.shape
    # normalize repr. along the batch dimension
    z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # (N,D)
    z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # (N,D)
    # cross-correlation matrix
    c = th.mm(z_a_norm.T, z_b_norm) / N  # (D,D)
    # loss
    c_diff = (c - th.eye(D).to(z_a.device)).pow(2)  # (D,D)
    invariance_term = _diagonal(c_diff).sum()
    redundancy_term = (_off_diagonal(c_diff) * lam).sum()
    loss = invariance_term + redundancy_term
    return loss
