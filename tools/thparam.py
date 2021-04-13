import torch as th


class MyModule(th.nn.Module):
    beta = th.nn.Parameter(th.tensor(1.0))

    def __init__(self):
        super(MyModule, self).__init__()
        self.gamma = th.nn.Parameter(th.tensor(2.0))


my = MyModule()
print(my)
print(list(my.parameters()))
print('parameters should be registered in the __init__ function')
