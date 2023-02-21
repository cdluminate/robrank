import argparse
import time
import rich
from rich.progress import track
import torch
import torchvision.models as models
console = rich.get_console()

ag = argparse.ArgumentParser()
ag.add_argument('--compile', '-c', action='store_true')
ag.add_argument('--compile-mode', '-m', type=str, default='default',
        choices=('default', 'max-autotune', 'reduce-overhead'))
ag.add_argument('--N-iteration', '-N', type=int, default=10)
ag = ag.parse_args()

with console.status('initialize model'):
    model = models.resnet18().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with console.status('compile'):
    ts = time.time()
    if ag.compile:
        compiled_model = torch.compile(model, mode=ag.compile_mode)
    else:
        compiled_model = model
    te = time.time()
    console.print('compile', te - ts)

with console.status('warmup'):
    x = torch.randn(16, 3, 224, 224).cuda()
    optimizer.zero_grad()
    ts = time.time()
    out = compiled_model(x)
    out.sum().backward()
    optimizer.step()
    te = time.time()
    console.print('forward-backward (warmup)', te - ts)

t = []
for i in track(range(ag.N_iteration), description='benchmarking ...'):
    x = torch.randn(16, 3, 224, 224).cuda()
    optimizer.zero_grad()
    ts = time.time()
    out = compiled_model(x)
    out.sum().backward()
    optimizer.step()
    te = time.time()
    t.append(te - ts)
console.print('mean:', sum(t)/len(t), 'all:', t)
