import torch
import numpy as np
from typing import List

from .base_model import BaseCertifierModel
from .networks import FFN


class ConstantClassifier(BaseCertifierModel):
    def __init__(self, model, class_weights: List[int], max_epochs: int = 10, learning_rate: float = 0.001,
                 optimizer: str = 'adam', epsilon: float = 0.00, alpha: float = 0.00):
        """
        Constant classifier
        :param model: torch.nn.Module implementing model
        :param max_epochs: maximum training epochs, which is required for scheduling epsilon
        :param learning_rate: learning rate for optimization
        :param epsilon: scalar value to control the scaling of self.fair_interval
        :param alpha: a flat value in 0, 1 which is the relative weight of fairness loss term
        """
        super().__init__(class_weights=class_weights, max_epochs=max_epochs, learning_rate=learning_rate,
                         optimizer=optimizer)
        self.save_hyperparameters()
        self.num_classes = model.num_classes
        self.EPSILON = epsilon
        self.fair_interval = None

        input_dim = model.input_dim
        num_classes = model.num_classes
        self.model = FFN(input_dim, num_classes, hidden_lay=0)

        _param = list(self.model.model.children())[0].weight
        _param.data = torch.zeros_like(_param)
        _param.requires_grad = False

    def set_fair_interval(self, interval):
        interval = torch.Tensor(interval).float()
        self.fair_interval = interval

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)

        loss = self.loss(probs, y)
        return {'acc': acc, 'loss': loss}

    def validation_epoch_end(self, outputs) -> None:
        all_accs = [o["acc"] for o in outputs]
        all_loss = [o["loss"] for o in outputs]
        self.log("val_acc", torch.stack(all_accs).mean(), prog_bar=True)
        self.log("val_quality", torch.stack(all_accs).mean(), prog_bar=True)
        self.log("val_loss", torch.stack(all_loss).mean(), prog_bar=True)
