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
import json
import fcntl
import time
import contextlib
import re
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters.terminal import TerminalFormatter


IMmean = th.tensor([0.485, 0.456, 0.406])
IMstd = th.tensor([0.229, 0.224, 0.225])


def renorm(im): return im.sub(IMmean[:, None, None].to(
    im.device)).div(IMstd[:, None, None].to(im.device))


def denorm(im): return im.mul(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def xdnorm(im): return im.div(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def chw2hwc(im): return im.transpose((0, 2, 3, 1)) if len(
    im.shape) == 4 else im.transpose((1, 2, 0))


def rjson(j: object) -> str:
    '''
    Render/Highlight the JSON code for pretty print
    '''
    if isinstance(j, str):
        '''
        let's assume it's a json string
        '''
        code = j
    elif any(isinstance(j, x) for x in (str, list, dict, float, int)):
        '''
        let's first serialize it into json then render
        '''
        code = json.dumps(j)
    else:
        raise ValueError('does not know how to deal with such datatype')
    return highlight(code, PythonLexer(), TerminalFormatter())


def pdist(repres: th.Tensor, metric: str) -> th.Tensor:
    '''
    Helper: compute pairwise distance matrix.
    https://github.com/pytorch/pytorch/issues/48306
    '''
    assert(len(repres.shape) == 2)
    with th.no_grad():
        if metric == 'C':
            # 1. th.nn.functional.cosine_similarity(x[:,:,None],
            # x.t()[None,:,:])
            repres = th.nn.functional.normalize(repres, dim=-1)
            pdist = 1.0 - th.mm(repres, repres.t())
        elif metric in ('E', 'N'):
            if metric == 'N':
                repres = th.nn.functional.normalize(repres, dim=-1)
            # Memory efficient pairwise euclidean distance matrix
            # 1. th.nn.functional.pairwise_distance(x[:,:,None], x.t()[None,:,:])
            # 2. th.cdist(x,x)
            prod = th.mm(repres, repres.t())
            norm = prod.diag().unsqueeze(1).expand_as(prod)
            pdist = (norm + norm.t() - 2 * prod).sqrt()
        else:
            raise ValueError(f'illegal metric {metric}')
    return pdist


def orthogonalRegularization(model, loss):
    losses = []
    for m in model.modules():
        if isinstance(m, th.nn.Linear):
            w = m.weight
            mat = th.matmul(w, w.t())
            diff = mat - th.diag(th.diag(mat))
            loss = th.mean(th.pow(diff, 2))
            losses.append(loss)
    return th.sum(losses)


@contextlib.contextmanager
def openlock(*args, **kwargs):
    lock = open(*args, **kwargs)
    fcntl.lockf(lock, fcntl.LOCK_EX)
    try:
        yield lock
    finally:
        fcntl.lockf(lock, fcntl.LOCK_UN)
        lock.close()


def nsort(L: list, R: str):
    '''
    sort list L by the key:int matched from regex R, descending.
    '''
    assert(all(re.match(R, item) for item in L))
    nL = [(int(re.match(R, item).groups()[0]), item) for item in L]
    nL = sorted(nL, key=lambda x: x[0], reverse=True)
    return [x[-1] for x in nL]


def test_nsort():
    x = [x.strip() for x in '''
    version_0
    version_2
    version_10
    version_3
    version_1
    '''.strip().split('\n')]
    y = [y.strip() for y in '''
    epoch=0.ckpt
    epoch=10.ckpt
    epoch=2.ckpt
    epoch=7.ckpt
    '''.strip().split('\n')]
    assert(nsort(x, r'version_(\d+)')[0] == 'version_10')
    print(nsort(x, r'.*sion_(\d+)')[0] == 'version_10')
    assert(nsort(y, r'epoch=(\d+)')[0] == 'epoch=10.ckpt')
    print(nsort(y, r'.*ch=(\d+)')[0] == 'epoch=10.ckpt')


if __name__ == '__main__':
    test_nsort()
