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

import rich
import torch as th
import torch.nn.functional as F
c = rich.get_console()
margin = 0.2
N, D = 10, 32

c.rule('Dot')
a = th.rand(D, requires_grad=True)
b = th.rand(D, requires_grad=True)
l = th.dot(a, b)
l.backward()
with th.no_grad():
    print('dl/da', (a.grad - b).abs().sum())
    print('dl/db', (b.grad - a).abs().sum())

c.rule('Batch Dot')
a = th.rand(N, D, requires_grad=True)
b = th.rand(N, D, requires_grad=True)
l = (a * b).sum(dim=1).sum()
l.backward()


def _grad_dot(a, b):
    return b, a


with th.no_grad():
    ga, gb = _grad_dot(a, b)
    print('dl/da', (a.grad - ga).abs().sum())
    print('dl/db', (b.grad - gb).abs().sum())

c.rule('Norm')
a = th.rand(N, D, requires_grad=True)
l = a.norm(p=2, dim=1).sum()
l.backward()


def _grad_norm(a):
    return a / a.norm(p=2, dim=1).view(N, 1)


with th.no_grad():
    ga = _grad_norm(a)
    print('dl/da', (a.grad - ga).abs().sum())

c.rule('Cosine')
a = th.rand(N, D, requires_grad=True)
b = th.rand(N, D, requires_grad=True)
l = F.cosine_similarity(a, b).sum()
l.backward()


def _grad_cosine(a, b):
    ga = b / (a.norm(p=2, dim=1) * b.norm(p=2, dim=1)).view(N, 1) \
        - (a * b).sum(dim=1).view(N, 1) \
        * a / ((a.norm(p=2, dim=1)**3) * b.norm(p=2, dim=1)).view(N, 1)
    gb = a / (b.norm(p=2, dim=1) * a.norm(p=2, dim=1)).view(N, 1) \
        - (a * b).sum(dim=1).view(N, 1) \
        * b / ((b.norm(p=2, dim=1)**3) * a.norm(p=2, dim=1)).view(N, 1)
    return ga, gb


with th.no_grad():
    ga, gb = _grad_cosine(a, b)
    print('dl/da', (a.grad - ga).abs().sum())
    print('dl/db', (b.grad - gb).abs().sum())

c.rule('Euclidean')
a = th.rand(N, D, requires_grad=True)
b = th.rand(N, D, requires_grad=True)
l = F.pairwise_distance(a, b).sum()
l.backward()


def _grad_euclidean(a, b):
    ga = (a - b) / F.pairwise_distance(a, b).view(N, 1)
    gb = (b - a) / F.pairwise_distance(a, b).view(N, 1)
    return ga, gb


with th.no_grad():
    ga, gb = _grad_euclidean(a, b)
    print('dl/da', (a.grad - ga).abs().sum())
    print('dl/db', (b.grad - gb).abs().sum())


def _grad_triplet(a, p, n, metric):
    if metric == 'C':
        ga1, gp = _grad_cosine(a, p)
        ga1, gp = -ga1, -gp
        ga2, gn = _grad_cosine(a, n)
    else:
        ga1, gp = _grad_euclidean(a, p)
        ga2, gn = _grad_euclidean(a, n)
        ga2, gn = -ga2, -gn
    return ga1 + ga2, gp, gn


for metric in ('C', 'N', 'E'):
    c.rule(f'Triplet / Metric {metric}')

    # prepare
    if metric in ('C', 'N'):
        xa = th.rand(N, D, requires_grad=True)
        xp = th.rand(N, D, requires_grad=True)
        xn = th.rand(N, D, requires_grad=True)
        a = F.normalize(xa)
        a.retain_grad()
        p = F.normalize(xp)
        p.retain_grad()
        n = F.normalize(xn)
        n.retain_grad()
    else:
        a = th.rand(N, D, requires_grad=True)
        p = th.rand(N, D, requires_grad=True)
        n = th.rand(N, D, requires_grad=True)

    # forward + backward
    if metric in ('E', 'N'):
        dap = F.pairwise_distance(a, p)
        dan = F.pairwise_distance(a, n)
    else:
        dap = 1 - F.cosine_similarity(a, p)
        dan = 1 - F.cosine_similarity(a, n)
    l = (dap - dan + margin).sum()
    l.backward()

    # validate gradient
    with th.no_grad():
        ga, gp, gn = _grad_triplet(a, p, n, metric=metric)
        print('dl/da', (a.grad - ga).abs().sum())
        print('dl/dp', (p.grad - gp).abs().sum())
        print('dl/dn', (n.grad - gn).abs().sum())
