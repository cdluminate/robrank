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
