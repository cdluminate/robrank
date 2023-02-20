import time
import rich
import torch
import torchvision.models as models
console = rich.get_console()

with console.status('init model'):
    model = models.resnet18().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

with console.status('compile'):
    ts = time.time()
    compiled_model = torch.compile(model, mode='max-autotune')
    te = time.time()
    console.print('compile', te - ts)

with console.status('benchmark'):
    x = torch.randn(16, 3, 224, 224).cuda()
    optimizer.zero_grad()
    ts = time.time()
    out = compiled_model(x)
    out.sum().backward()
    optimizer.step()
    te = time.time()
    console.print('forward backward', te - ts)
