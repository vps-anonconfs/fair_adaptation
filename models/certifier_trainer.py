import torch
import FairCertModule
from .base_model import BaseCertifierModel

from typing import List


class FairnessCertifier(BaseCertifierModel):
    def __init__(self, model, class_weights: List[int], max_epochs: int = 10, learning_rate: float = 0.001,
                 optimizer: str = 'adam', epsilon: float = 0.00, alpha: float = 0.00):
        """
        :param model: torch.nn.Module implementing model
        :param max_epochs: maximum training epochs, which is required for scheduling epsilon
        :param learning_rate: learning rate for optimization
        :param epsilon: scalar value to control the scaling of self.fair_interval 
        :param alpha: a flat value in 0, 1 which is the relative weight of fairness loss term
        
        @deprecated
        :param gamma: has no use
        :param mode: no need for mode, same as bool(alpha>0)
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
            
    def __init__(self, model, class_weights: List[int] = [1, 1], max_epochs: int = 10, learning_rate: float = 0.001,
                 optimizer: str = 'adam', epsilon: float = 0.00, alpha: float = 0.00):
        """
        :param model: torch.nn.Module implementing model
        :param max_epochs: maximum training epochs, which is required for scheduling epsilon
        :param learning_rate: learning rate for optimization
        :param epsilon: scalar value to control the scaling of self.fair_interval 
        :param alpha: a flat value in 0, 1 which is the relative weight of fairness loss term
        
        @deprecated
        :param gamma: has no use
        :param mode: no need for mode, same as bool(alpha>0)
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
            

    def get_param_string(self):
        return f"alpha:{self.ALPHA}-epsilon:{self.EPSILON}-schedule:{self.EPSILON_LINEAR}-epochs:{self.MAX_EPOCHS}" \
               f"-lr:{self.LEARN_RATE}-dim:{self.in_dim}"
            
    def set_fair_interval(self, interval):
        interval = torch.Tensor(interval).float()
        self.fair_interval = interval
                
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        regval = 0.0
        if self.ALPHA > 0:
            regval = FairCertModule.fairness_regularizer(self, x, y, self.fair_interval, self.eps, nclasses=self.num_classes)
        loss = ((1-self.ALPHA) * self.loss(y_hat, y)) + (self.ALPHA * regval)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        probs = self(x)
        acc = self.accuracy(probs, y)

        self.macro_accuracy(probs, y)
        regval = torch.tensor(0.0)
        if self.ALPHA > 0:
            regval = FairCertModule.fairness_regularizer(self, x, y, self.fair_interval, self.EPSILON,
                                                         nclasses=self.num_classes)
        loss = ((1-self.ALPHA) * self.loss(probs, y)) + (self.ALPHA * regval)
        return {'acc': acc, 'loss': loss, 'regval': regval/len(x)}

    def validation_epoch_end(self, outputs) -> None:
        all_accs = [o["acc"] for o in outputs]
        all_loss = [o["loss"] for o in outputs]

        all_regval = [o["regval"] for o in outputs]
        regval = torch.stack(all_regval).mean()
        macro_acc = self.macro_accuracy.compute()
        val_quality = macro_acc - 0.5*regval
        self.macro_accuracy.reset()

        self.log("val_quality", val_quality, prog_bar=True)
        self.log("val_acc", torch.stack(all_accs).mean(), prog_bar=True)
        self.log("val_loss", torch.stack(all_loss).mean(), prog_bar=True)
        if self.EPSILON_LINEAR:
            self.eps += self.EPSILON/max(1e-3, self.MAX_EPOCHS)