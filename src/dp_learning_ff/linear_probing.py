import warnings
from typing import Any, Optional

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import (
    STEP_OUTPUT,
)

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader


class LinearProbingModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, activation=torch.nn.Identity()):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class LinearProbingClassifier(LightningModule):
    def __init__(
        self,
        num_features,
        num_classes,
        epochs: int,
        epsilon,
        delta,
        max_grad_norm,
        lr: Optional[float] = None,
    ):
        super().__init__()
        self.model = LinearProbingModel(num_features, num_classes)
        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.lr = 1e-3 if lr is None else lr
        self.privacy_engine = PrivacyEngine()
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0)
        self.trainer.fit_loop.setup_data()
        data_loader = self.trainer.train_dataloader
        # transform (model, optimizer, dataloader) to DP-versions
        if hasattr(self, "dp"):
            self.dp["model"].remove_hooks()
        if not isinstance(data_loader, DPDataLoader):
            warnings.warn(
                "Dataloader is not DPDataLoader. Manually adjust sampling or privacy guarantees are violated."
            )
        dp_model, optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
            module=self,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=self.epochs,
            max_grad_norm=self.max_grad_norm,
            poisson_sampling=isinstance(data_loader, DPDataLoader),
        )
        self.dp = {"model": dp_model}

        return optimizer

    def training_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = torch.tensor(y_hat.argmax(dim=1) == y, dtype=float).mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = torch.tensor(y_hat.argmax(dim=1) == y, dtype=float).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        acc = torch.tensor(y_hat.argmax(dim=1) == y, dtype=float).mean()
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        # Logging privacy spent: (epsilon, delta)
        epsilon = self.privacy_engine.get_epsilon(self.delta)
        self.log("epsilon", epsilon, on_epoch=True, prog_bar=True)
        return super().on_train_batch_end(outputs, batch, batch_idx)
