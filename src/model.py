import torch
import lightning as lit
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from src.models.pretrained_resnet import ResnetPT
from configs.config import config


class ModelLit(lit.LightningModule):
    def __init__(self):
        super(ModelLit, self).__init__()
        self.save_hyperparameters()
        self.model = ResnetPT()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, label = batch
        output = self.model(x)
        loss = F.cross_entropy(output, label)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=config["training"]["learning_rate"])
        return optimizer
