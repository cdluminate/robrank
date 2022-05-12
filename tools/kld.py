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

import rich
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
c = rich.get_console()

N = 10
maxiter = 100

x = th.rand(10)
print(x)
x.requires_grad = True

y = th.rand(10)
print(y)
y.requires_grad = True

opt = th.optim.SGD([x, y], lr=1e-1)

curve = []
for iteration in range(maxiter):
    opt.zero_grad()
    #loss = F.kl_div(F.softmax(x, dim=0), F.softmax(y, dim=0), reduction='sum')
    #loss = F.kl_div(x, y, reduction='sum')
    #loss = F.kl_div(F.normalize(x, p=1, dim=0), F.normalize(y, p=1, dim=0), reduction='sum')
    loss = F.mse_loss(x, y, reduction='mean')
    curve.append(loss.item())
    c.print(f'iteration {iteration} loss', loss.item())
    loss.backward()
    opt.step()
    c.print('x', x)
    c.print('y', y)

plt.plot(curve)
plt.show()
