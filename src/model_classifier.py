import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
from torchvision.models import resnet18

from .model_style_transfer import AdaInModel


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        optimizer: str = "adam",
        style_checkpoint: str = None,
        n_classes: int = 4,
    ):
        """Scanner Classifer

        Args:
            lr: learning rate
            optimizer: name of optimizer (adam | sgd)
            style_checkpoint: checkpoint of style transfer model
            n_classes: number of classes
        """
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.optimizer = optimizer
        self.n_classes = n_classes

        # Initalize network
        self.net = resnet18(pretrained=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

        # Metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.confusion = ConfusionMatrix(num_classes=n_classes)

        if style_checkpoint:
            self.style_net = AdaInModel().load_from_checkpoint(style_checkpoint)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.train_acc(pred.max(1).indices, y.max(1).indices)

        # Log
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through model
        pred = self(x)

        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.val_acc(pred.max(1).indices, y.max(1).indices)

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
        if isinstance(batch, dict):
            x_cont, y = batch["content"]
            img_s = batch["style"]
            x, _, _ = self.style_net(x_cont, img_s)  # Apply style transfer
        else:
            x, y = batch

        # Convert to 1-hot targets
        y = F.one_hot(y, num_classes=self.n_classes).float()

        # Pass through model
        pred = self(x)

        max_prob = F.softmax(pred, dim=1).max(1).values

        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(pred, y)
        acc = self.val_acc(pred.max(1).indices, y.max(1).indices)
        conf = self.confusion(pred.max(1).indices, y.max(1).indices)

        # Log
        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return {
            "confs": conf,
            "max_probs": max_prob,
            "test_loss": loss,
            "test_acc": acc,
        }

    def test_epoch_end(self, outputs):
        confs = torch.stack([x["confs"] for x in outputs], axis=0).sum(dim=0)
        max_probs = torch.cat([x["max_probs"] for x in outputs], axis=0).mean()
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        print("\nTest Confusion Matrix:")
        print(confs)
        print(f"Test Accuracy: {avg_acc}")
        print(f"Test Loss: {avg_loss}")
        print(f"Average Confidence: {max_probs.item():.3}")

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = Adam(self.net.parameters(), lr=self.lr)
        else:
            optimizer = SGD(self.net.parameters(), lr=self.lr, momentum=0.9)

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
