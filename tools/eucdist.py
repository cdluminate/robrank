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

N = 9
D = 512

embs = th.rand(N, D)

pdistA = (embs.view(N, 1, D).expand(N, N, D)
          - embs.view(1, N, D).expand(N, N, D)).norm(2, dim=2)
print(pdistA)

prod = th.mm(embs, embs.t())
norm = prod.diag().unsqueeze(1).expand_as(prod)
pdist = (norm + norm.t() - 2 * prod).sqrt()
print(pdist)

# we can use th.cdist / th.pdist

print((pdistA - pdist).abs().sum())
