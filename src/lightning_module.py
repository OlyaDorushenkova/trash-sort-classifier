import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl


class TrashClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        num_classes: int = 6,
        log_mlflow: bool = True,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.log_mlflow = log_mlflow

        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        acc = self.train_acc(preds, y)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        self.log(
            "train_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        acc = self.val_acc(preds, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
        )

        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        acc = self.test_acc(preds, y)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
        )

        self.log(
            "test_acc",
            acc,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_epoch_end(self):
        """
        Optional MLflow logging hook (if enabled externally).
        """
        pass