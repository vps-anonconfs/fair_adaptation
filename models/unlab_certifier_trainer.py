import torch
import copy
import FairCertModule as FCM
from .certifier_trainer import FairnessCertifier

from typing import List


class UnlabFairnessCertifier(FairnessCertifier):
    def __init__(self, model, class_weights: List[int], max_epochs: int = 10, learning_rate: float = 0.001,
                 optimizer: str = 'adam', epsilon: float = 0.00, alpha: float = 0.00):
        """
        Fairness Certifier that trains using only unlab data
        :param model: torch.nn.Module implementing model
        :param max_epochs: maximum training epochs, which is required for scheduling epsilon
        :param learning_rate: learning rate for optimization
        :param epsilon: scalar value to control the scaling of self.fair_interval
        :param alpha: a flat value in 0, 1 which is the relative weight of fairness loss term
        """
        super().__init__(class_weights=class_weights, max_epochs=max_epochs, learning_rate=learning_rate,
                         optimizer=optimizer)
        self.model = model
        self.save_hyperparameters()

        self.ALPHA = alpha            # Regularization Parameter (Weights the Reg. Term)
        self.EPSILON = epsilon        # Input Perturbation Budget at Training Time

        self.EPSILON_LINEAR = True   # Put Epsilon on a Linear Schedule?
        self.fair_interval = None    # this shall be set with set_fair_interval routine

        self.num_classes = model.num_classes
        if self.EPSILON_LINEAR:
            self.eps = 0.0
        else:
            self.eps = self.EPSILON

        # initial_model ensures that the trained model predictions do not diverge from the starting model
        self.initial_model = copy.deepcopy(self.model)

    def training_step(self, batch, batch_idx):
        # pretending to not see y
        x, _ = batch
        y_hat = self(x)
        with torch.no_grad():
            logits_ = self.initial_model(x)
            y = torch.argmax(logits_, dim=-1)
        regval = 0.0
        if self.ALPHA > 0:
            regval = FCM.fairness_regularizer(self, x, y, self.fair_interval, self.eps, nclasses=self.num_classes)
        loss = ((1-self.ALPHA) * self.loss(y_hat, y)) + (self.ALPHA * regval)
        return loss