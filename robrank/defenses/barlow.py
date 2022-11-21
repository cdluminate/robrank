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

from typing import List
import torch as th
import torch.nn.functional as F
from collections import Counter
import rich
console = rich.get_console()
__lambda__ = 5e-3


def _off_diagonal(matrix: th.Tensor):
    '''
    https://discuss.pytorch.org/t/most-efficient-way-to-get-just-the-off-diagonal-elements-of-a-tensor/131065
    '''
    assert matrix.dim() == 2
    res = matrix.clone()
    res.diagonal(dim1=-1, dim2=-2).zero_()
    return res


@th.no_grad()
def test__off_diagonal():
    x = th.rand(10, 10)
    ref = _off_diagonal(x).sum()
    x.diagonal().zero_()
    res = x.sum()
    assert abs(res - ref) < 1e-5, 'implementation bug'


def _diagonal(matrix: th.Tensor):
    assert matrix.dim() == 2
    return matrix.diagonal(dim1=-1, dim2=-2)


@th.no_grad()
def test__diagonal():
    x = th.rand(10, 10)
    ref = x.trace()
    res = _diagonal(x).sum()
    assert abs(res - ref) < 1e-5, 'implementation bug'


def _barlow_twins(z_a: th.Tensor, z_b: th.Tensor, *, lam: float = __lambda__):
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


@th.no_grad()
def test__barlow_twins():
    a = th.rand(10, 10)
    b = th.rand(10, 10)
    bt = _barlow_twins(a, b)
    assert not th.isnan(bt), 'implementation bug'


def assemble_same_class(tensors: List[th.Tensor], labels: List[th.Tensor], *,
                        thres: int = 3) -> List[th.Tensor]:
    '''
    assemble vectors of the same class together
    all the given labels should be the same, i.e., labels[0] == labels[1]

    input:
      tensors: list of tensors
        [z_a_1, z_b_1, z_a_2, z_b_2, ...]
      labels:
        labels corresponding to the above tensors
    output:
      asm:
        [[z_a_cls1, z_b_cls1], [z_a_cls2, z_b_cls2], ...]
      we select the classes that appeared at least `thres` times on a single
      side for A/B.
    '''
    counter = Counter()
    for i in range(0, len(labels), 2):
        counter.update(labels[i].tolist())
    selections = [k for (k, v) in counter.items() if v >= thres]
    #print('asm: selections', selections)
    asm = []
    for cls in selections:
        acls = [x[l == cls] for (x, l) in zip(tensors[0::2], labels[0::2])]
        #print('acls', cls, 'sizes', [x.shape for x in acls])
        acls = th.vstack(acls)
        bcls = [x[l == cls] for (x, l) in zip(tensors[1::2], labels[1::2])]
        #print('bcls', cls, 'sizes', [x.shape for x in bcls])
        bcls = th.vstack(bcls)
        asm.append([acls, bcls])
    return asm


@th.no_grad()
def test_assemble_same_class():
    '''
    note, use the following command to enable stdout printing
    $ pytest -v -x -k barlow -s
    '''
    xa = th.rand(100, 10)
    xb = th.rand(100, 10)
    la = th.randint(0, 10, (100,))
    lb = la.clone()
    thres = 3
    asm = assemble_same_class([xa, xb], [la, lb], thres=thres)
    for (xa, xb) in asm:
        assert xa.dim() == 2
        assert xa.size(0) >= 3
        assert xb.dim() == 2
        assert xb.size(0) >= 3
        assert xa.shape == xb.shape
        print('xa.shape', xa.shape, 'xb.shape', xb.shape)


def barlow_twins(z_a: th.Tensor, l_a: th.Tensor,
                 z_b: th.Tensor, l_b: th.Tensor,
                 *, lam: float = __lambda__):
    '''
    wrapped version of barlow twins
    input:
    z_a: a batch of arbitrary class
    l_a: labels of z_a
    z_b: another batch of arbitrary class
    note, class(z_a) should be aligned with class(z_b)
    '''
    # re-assemble vectors in the same class
    asm = assemble_same_class([z_a, z_b], [l_a, l_b])
    losses = []
    for (za, zb) in asm:
        loss = _barlow_twins(za, zb)
        losses.append(loss)
    #print(losses)
    return sum(losses)


#th.no_grad()  # XXX: don't use no_grad -- here we test backward pass as well
def test_barlow_twins():
    xa = th.rand(100, 10)
    xa.requires_grad = True
    xb = th.rand(100, 10)
    xb.requires_grad = True
    la = th.randint(0, 10, (100,))
    lb = la.clone()
    loss = barlow_twins(xa, la, xb, lb)
    print(loss)
    loss.backward()
