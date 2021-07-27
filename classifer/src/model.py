import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, ConfusionMatrix
from torchvision.models import resnet50
from torchvision.utils import save_image

from .style_transfer.model import AdaInModel


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adam",
        style_checkpoint: str = None,
        n_classes: int = 4,
        smoothing: float = 0,
    ):
        """Scanner Classifer

        Args:
            lr: learning rate
            optimizer: name of optimizer (adam | sgd)
            style_checkpoint: checkpoint of style transfer model
            n_classes: number of classes
            smoothing: label smoothing factor
        """
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.smoothing = smoothing

        self.net = resnet50(pretrained=False)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_confusion = ConfusionMatrix(num_classes=n_classes)

        if style_checkpoint:
            self.style_net = AdaInModel().load_from_checkpoint(style_checkpoint)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        y = F.one_hot(y, num_classes=self.hparams.n_classes).float()

        # Apply label smoothing
        if self.hparams.smoothing > 0:
            y = self._smooth(y)

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.train_acc(pred.max(1).indices, y)

        # Log
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred.max(1).indices, y)

        # Log
        self.log("val_loss", loss)
        self.log("val_acc", acc)

        return {
            "val_acc": acc,
            "val_loss": loss,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        print(f"Validation Loss: {avg_loss} Validation Accuracy: {avg_acc}")

    def test_step(self, batch, _):
        x, y = batch["content"]
        img_s = batch["style"]

        # Apply style transfer
        x_s, _, _ = self.style_net(x, img_s)
        # x_s = x

        # Pass through model
        pred = self(x_s)

        max_prob = F.softmax(pred, dim=1).max(1).values

        # Calculate loss and accuracy
        loss = F.cross_entropy(pred, y)
        acc = self.val_acc(pred.max(1).indices, y)
        conf = self.test_confusion(pred.max(1).indices, y)

        # Log
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {
            "confs": conf,
            "max_probs": max_prob,
        }

    def test_epoch_end(self, outputs):
        confs = torch.stack([x["confs"] for x in outputs], axis=0).sum(dim=0)
        max_probs = torch.stack([x["max_probs"] for x in outputs], axis=0).mean(dim=0)

        print("\nTest Confusion Matrix:")
        print(confs)
        print(f"Average Confidence: {max_probs.item():.3}")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = Adam(self.net.parameters(), lr=self.lr)
        else:
            optimizer = SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def _smooth(self, y):
        smoothing = self.hparams.smoothing
        confidence = 1.0 - smoothing  # Confidence for target class
        label_shape = y.size()
        other = smoothing / (label_shape[1] - 1)  # Confidence for non-target classes

        # Create new smoothed target vector
        smooth_y = torch.empty(size=label_shape, device=self.device)
        smooth_y.fill_(other)
        smooth_y.add_(y * confidence).sub_(y * other)

        return smooth_y
