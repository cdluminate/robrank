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
###############################################################################
# utils.py
# Really, these are merely some miscellaneous helper functions.
###############################################################################

# pylint: disable=no-member
import torch as th
import json
import fcntl
import contextlib
import os
import re
import numpy as np
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters.terminal import TerminalFormatter
from sklearn.metrics.cluster import normalized_mutual_info_score as __nmi
import rich
c = rich.get_console()
#
try:
    import faiss
    faiss.omp_set_num_threads(4)
except ImportError:
    from sklearn.cluster import KMeans


IMmean = th.tensor([0.485, 0.456, 0.406])  # pylint: disable=not-callable
IMstd = th.tensor([0.229, 0.224, 0.225])   # pylint: disable=not-callable
IMmean_ibn = th.tensor([0.502, 0.4588, 0.4078])
IMstd_ibn = th.tensor([0.0039, 0.0039, 0.0039])


def renorm(im): return im.sub(IMmean[:, None, None].to(
    im.device)).div(IMstd[:, None, None].to(im.device))


def renorm_ibn(im): return im.sub(IMmean_ibn[:, None, None].to(
    im.device)).div(IMstd_ibn[:, None, None].to(im.device))[:, range(3)[::-1], :, :]


def denorm(im): return im.mul(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def denorm_ibn(im): return im[:, range(3)[::-1], :, :].mul(IMmean_ibn[:, None, None].to(
    im.device)).add(IMstd_ibn[:, None, None].to(im.device))


def xdnorm(im): return im.div(IMstd[:, None, None].to(
    im.device)).add(IMmean[:, None, None].to(im.device))


def chw2hwc(im): return im.transpose((0, 2, 3, 1)) if len(
    im.shape) == 4 else im.transpose((1, 2, 0))


def metric_get_nmi(valvecs: th.Tensor, vallabs: th.Tensor, ncls: int) -> float:
    '''
    wrapper with a CUDA-OOM (out of memory) guard
    '''
    try:
        nmi = __metric_get_nmi(valvecs, vallabs, ncls)
    except RuntimeError as e:
        print('! FAISS(GPU) Triggered CUDA OOM. Falling back to CPU clustering...')
        os.putenv('FAISS_CPU', '1')
        nmi = __metric_get_nmi(valvecs, vallabs, ncls, use_cuda=False)
    return nmi


def __metric_get_nmi(valvecs: th.Tensor, vallabs: th.Tensor,
                     ncls: int, use_cuda: bool = True) -> float:
    '''
    Compute the NMI score
    '''
    use_cuda: bool = use_cuda and th.cuda.is_available() \
        and hasattr(faiss, 'StandardGpuResources')
    if int(os.getenv('FAISS_CPU', 0)) > 0:
        use_cuda = False
    npvecs = valvecs.detach().cpu().numpy().astype(np.float32)
    nplabs = vallabs.detach().cpu().view(-1).numpy().astype(np.float32)
    # a weird dispatcher but it works.
    if 'faiss' in globals():
        if use_cuda:
            gpu_resource = faiss.StandardGpuResources()
            cluster_idx = faiss.IndexFlatL2(npvecs.shape[1])
            if not th.distributed.is_initialized():
                cluster_idx = faiss.index_cpu_to_gpu(
                    gpu_resource, 0, cluster_idx)
            else:
                cluster_idx = faiss.index_cpu_to_gpu(gpu_resource,
                                                     th.distributed.get_rank(), cluster_idx)
            kmeans = faiss.Clustering(npvecs.shape[1], ncls)
            kmeans.verbose = False
            kmeans.train(npvecs, cluster_idx)
            _, pred = cluster_idx.search(npvecs, 1)
            pred = pred.flatten()
        else:
            kmeans = faiss.Kmeans(
                npvecs.shape[1], ncls, seed=123, verbose=False)
            kmeans.train(npvecs)
            _, pred = kmeans.index.search(npvecs, 1)
            pred = pred.flatten()
        nmi = __nmi(nplabs, pred)
    elif 'KMeans' in globals():
        kmeans = KMeans(n_clusters=ncls, random_state=0).fit(npvecs)
        nmi = __nmi(nplabs, kmeans.labels_)
    else:
        raise NotImplementedError(
            'please provide at leaste one kmeans implementation for the NMI metric.')
    return nmi


def metric_get_rank(dist: th.Tensor, label: th.Tensor,
                    vallabels: th.Tensor, ks: list) -> tuple:
    '''
    Flexibly get the rank of the topmost item in the same class
    dist = [dist(anchor,x) for x in validation_set]

    dist (1 x len(vallabels)): pairwise distance vector between a single query
        to the validation set.
    label: int label for the query
    vallabels: label array for the validation set
    '''
    assert(dist.nelement() == vallabels.nelement())
    # [important] argsort(...)[:,1] for skipping the diagonal (R@1=1.0)
    # we skip the smallest value as it's exactly for the anchor itself
    argsort = dist.argsort(descending=False)[1:]
    rank = th.where(vallabels[argsort] == label)[0].min().item()
    return (rank,) + tuple(rank < k for k in ks)


def test_metric_get_rank():
    N = 32
    dist = th.arange(N) / N
    label = 1
    labels = th.zeros(N)
    labels[[0, 1]] = 1
    recall = metric_get_rank(dist, label, labels, [1, 2])
    assert(recall == (0, True, True))
    labels = th.zeros(N)
    labels[[0, 2]] = 1
    recall = metric_get_rank(dist, label, labels, [1, 2])
    assert(recall == (1, False, True))


def metric_get_ap(dist: th.Tensor, label: th.Tensor,
                  vallabels: th.Tensor) -> float:
    '''
    Get the overall average precision

    dist (1 x len(vallabels)): pairwise distance vector between a single query
        to the validation set.
    label: int label for the query
    vallabels: label array for the validation set
    '''
    assert(dist.nelement() == vallabels.nelement())
    # we skip the smallest value as it's exectly for the anchor itself
    argsort = dist.argsort(descending=False)[1:]
    argwhere1 = th.where(vallabels[argsort] == label)[0] + 1
    ap = ((th.arange(len(argwhere1)).float() + 1).to(argwhere1.device) /
          argwhere1).sum().item() / len(argwhere1)
    return ap


def metric_get_ap_r(dist: th.Tensor, label: th.Tensor,
                    vallabels: th.Tensor, rs: list) -> float:
    '''
    computes the mAP@R metric following
    "A metric learning reality check", eccv 2020

    dist (1 x len(vallabels)): pairwise distance vector between a single query
        to the validation set.
    label: int label for the query
    vallabels: label array for the validation set
    '''
    assert(dist.nelement() == vallabels.nelement())
    # we skip the smallest value as it's exactly for the anchor itself
    argsort = dist.argsort(descending=False)[1:].cpu()
    mask = (vallabels[argsort] == label).cpu()
    cmask = mask.cumsum(dim=0)
    mapr = []
    for r in rs:
        tmp = (cmask[:r] / (th.arange(r) + 1))[mask[:r]].sum() / r
        mapr.append(tmp.item())
    return tuple(mapr)


def test_metric_get_ap_r():
    def et1e_4(a, b):
        assert(abs(a - b) < 1e-4)
    N = 101
    dist = th.arange(N) / N
    label = 1
    #
    labels = th.zeros(N)
    labels[[0, 1]] = 1
    mapr = metric_get_ap_r(dist, label, labels, [10])
    et1e_4(mapr[0], 0.1)
    #
    labels = th.zeros(N)
    labels[[0, 1, 10]] = 1
    mapr = metric_get_ap_r(dist, label, labels, [10])
    et1e_4(mapr[0], 0.12)
    #
    labels = th.zeros(N)
    labels[[0, 1, 2]] = 1
    mapr = metric_get_ap_r(dist, label, labels, [10])
    et1e_4(mapr[0], 0.20)
    #
    labels = th.zeros(N)
    labels[th.arange(11)] = 1
    mapr = metric_get_ap_r(dist, label, labels, [10])
    et1e_4(mapr[0], 1.00)


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


def warn(message: str):
    c.print(f'[bold yellow underline]W: {message}[/bold yellow underline]')


def info(message: str):
    c.print(f'[bold cyan underline]I: {message}[/hold cyan underline]')


if __name__ == '__main__':
    test_nsort()
