import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from typing import List
import os
import numpy as np

import FairCertModule


class BaseCertifierModel(pl.LightningModule):
    def __init__(self, class_weights: List[int], max_epochs: int = 10, learning_rate: float = 0.001,
                 optimizer: str = None):
        super().__init__()
        self.LEARN_RATE = learning_rate      # Learning Rate Hyperparameter
        self.MAX_EPOCHS = max_epochs         # Maximum Epochs to Train the Model for
        self.class_weights = class_weights
        self.optimizer_name = optimizer
        self.macro_accuracy = torchmetrics.Accuracy(num_classes=len(self.class_weights), average='macro')

    def classify(self, x):
        outputs = self(x)
        return F.softmax(outputs, dim=1), torch.max(outputs, 1)[1]

    def loss(self, x, y):
        if type(self.class_weights) != torch.Tensor:
            return F.cross_entropy(x, y)
        else:
            return F.cross_entropy(x, y, weight=self.class_weights)

    @property
    def layers(self):
        return self.model.layers

    @property
    def activations(self):
        return self.model.activations

    def set_fair_interval(self, interval):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.macro_accuracy(logits, y)
        fairness_delta = FairCertModule.fairness_delta(self, inp=x, lab=y, vec=self.fair_interval, eps=self.EPSILON,
                                                       nclasses=self.num_classes)
        robustness_delta = FairCertModule.fairness_delta(self, inp=x, lab=y, vec=torch.ones_like(self.fair_interval),
                                                         eps=self.EPSILON, nclasses=self.num_classes)
        return {'acc': acc, 'delta': fairness_delta, 'robustness_delta': robustness_delta}

    @staticmethod
    def accuracy(logits, y):
        acc = torch.sum(torch.eq(torch.argmax(logits, -1), y).to(torch.float32)) / len(y)
        return acc

    def test_epoch_end(self, outputs) -> None:
        all_deltas = [o["delta"] for o in outputs]
        all_rdeltas = [o["robustness_delta"] for o in outputs]
        test_deltas = np.concatenate(all_deltas)
        test_rdeltas = np.concatenate(all_rdeltas)
        self.log("test_acc", self.macro_accuracy.compute())
        self.log("test_mean_delta", test_deltas.mean())
        self.log("test_max_delta", test_deltas.max())
        self.log("test_mean_rdelta", test_rdeltas.mean())
        self.log("test_max_rdelta", test_rdeltas.max())
        self.macro_accuracy.reset()

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.LEARN_RATE)
        elif self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.LEARN_RATE, momentum=0.9)

    def save(self, trainer):
        directory = "Models"
        if not os.path.exists(directory):
            os.makedirs(directory)
        SCHEDULED = self.EPSILON_LINEAR
        MODEL_ID = "FCN_e=%s_a=%s_s=%s" % (self.EPSILON, self.ALPHA, SCHEDULED)
        trainer.save_checkpoint("Models/%s.ckpt" % MODEL_ID)
        torch.save(self.state_dict(), "Models/%s.pt" % MODEL_ID)
