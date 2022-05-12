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
import torch as th
import rich
c = rich.get_console()

EPS = 8./255.
ALPHA = 2./255.

images = th.rand(10, 3, 224, 224)
c.print('images > shape', images.shape, 'min', images.min(), 'max', images.max())
ptb = (th.rand(10, 3, 224, 224)*2 - 1) * ALPHA
c.print('ptb > shape', ptb.shape, 'min', ptb.min(), 'max', ptb.max())

def proj1(x, y):
    # ref1
    x = x.clone().detach()
    x = th.min(x, y + EPS)
    x = th.max(x, y - EPS)
    x = th.clamp(x, min=0., max=1.)
    x = x.clone().detach()
    return x

def proj2(x, y):
    # ref2
    x = x.clone().detach()
    delta = th.clamp(x - y, min=-EPS, max=EPS)
    #x = th.clamp(x + delta, min=0., max=1.).detach()  # this looks wrong
    x = th.clamp(y + delta, min=0., max=1.).detach()  # this seems to fix the issue
    return x

def proj3(x, y):
    # prob1, seems equiv to ref1
    x = x.clone().detach()
    x = th.clamp(x, min=y-EPS, max=y+EPS)
    x = th.clamp(x, min=0, max=1)
    x = x.clone().detach()
    return x

PROJ = [proj1, proj2, proj3]

E1 = -th.ones(len(PROJ), len(PROJ))
E2 = -th.ones(len(PROJ), len(PROJ))
E3 = -th.ones(len(PROJ), len(PROJ))
for (i, proji) in enumerate(PROJ):
    for (j, projj) in enumerate(PROJ):
        # case 1
        pimages = images.clone().detach()
        a1 = proji(pimages, images)
        a2 = projj(pimages, images)
        e = th.norm(a1 - a2, p=1)
        E1[i, j] = e

        # case 2
        pimages = (images + ptb).clone().detach()
        a1 = proji(pimages, images)
        a2 = projj(pimages, images)
        e = th.norm(a1 - a2, p=1)
        E2[i, j] = e

        # case 3
        pimages = (images + EPS*ptb/ALPHA).clone().detach()
        a1 = proji(pimages, images)
        a2 = projj(pimages, images)
        e = th.norm(a1 - a2, p=1)
        E3[i, j] = e

c.print('case 1 diff 1-norm\n', E1)
c.print('case 2 diff 1-norm\n', E2)
c.print('case 3 diff 1-norm\n', E3)
