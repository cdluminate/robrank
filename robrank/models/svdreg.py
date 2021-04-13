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
import torch.nn.functional as F
import functools as ft
import operator as op
import rich
c = rich.get_console()

Constraints = {
    1: lambda svd: (svd.S.max() - 1e+3).relu(),
    2: lambda svd: (svd.S[0] / svd.S[1] - 8.0).relu(),
    3: lambda svd: (1.2 - svd.S[1:5] / svd.S[2:6]).relu().mean(),
    4: lambda svd: (1e-3 / svd.S.min() - 1.0).relu(),
    5: lambda svd: th.exp(svd.S.max() / (8192 * svd.S.min()) - 1),
    6: lambda svd: th.log(svd.S.max() / 2e+2).relu(),
    7: lambda svd: th.log(1e-5 / svd.S.min()).relu(),
    8: lambda svd: th.log(svd.S.max() / (1e5 * svd.S.min())).relu()
}


def svdreg(
        model: th.nn.Module,
        repres: th.Tensor,
        *,
        constraints: list = [6, 7],
        verbose: bool = True,
) -> th.Tensor:
    '''
    Perform Singular Value Regularization

    Prevents the model from collapsing but may also hamper the robustness
    for adversarial training.
    '''
    # Pass on the Singular Value Decomposition
    svd = th.svd(repres)
    model.log('Train/SVD.S.MAX', svd.S.max().item())
    c.log('S.max', svd.S.max().item(), 'min', svd.S.min().item())
    c.log('S.top5', svd.S.cpu().tolist()[:5])
    penalties = [Constraints[i](svd) for i in constraints]
    return ft.reduce(op.add, penalties) if penalties else 0.0


def test_svdreg():
    model = th.nn.Sequential(th.nn.Linear(8, 8))
    repres = th.rand(4, 8)
    model.log = lambda *args: 0
    loss = svdreg(model, repres)
    assert(isinstance(loss, th.Tensor))
