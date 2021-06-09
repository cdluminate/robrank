from .. import configs
import pytorch_lightning as thl
import torch as th
import torchvision as vision
from .template_classify import ClassifierTemplate


class Model(ClassifierTemplate, thl.LightningModule):
    BACKBONE = 'cres18'

    def __init__(self, *, dataset: str, loss: str):
        super().__init__()
        # dataset setup
        assert(dataset in configs.cres18.allowed_datasets)
        assert(loss in configs.cres18.allowed_losses)
        self.dataset = dataset
        self.loss = loss
        self.config = configs.cres18(dataset, loss)
        self.backbone = vision.models.resnet18(False)
        # perform surgery
        self.backbone.fc = th.nn.Linear(
            512, getattr(configs, dataset).num_class)
