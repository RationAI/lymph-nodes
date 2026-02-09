import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryRecall,
    BinarySpecificity,
)


class LymphNodesMIL(LightningModule):
    def __init__(
        self, input_dim: int = 1536, lr: float = 1e-4, pos_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.attention_V = nn.Sequential(nn.Linear(input_dim, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)

        self.classifier = nn.Sequential(nn.Linear(input_dim, 1))

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        metrics = {
            "AUC": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "sensitivity": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "F1": BinaryF1Score(),
        }

        self.train_metrics = MetricCollection(metrics, prefix="train/")
        self.val_metrics = MetricCollection(metrics, prefix="val/")
        self.test_metrics = MetricCollection(metrics, prefix="test/")

    def forward(self, x):
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)

        a = self.attention_weights(a_v * a_u)

        mask = x.abs().sum(dim=-1, keepdim=True) > 1e-6
        a = a.masked_fill(~mask, float("-inf"))

        A = torch.softmax(a, dim=1)
        M = torch.sum(A * x, dim=1)

        logits = self.classifier(M)

        return logits.squeeze(1), A.squeeze(2)

    def training_step(self, batch, batch_idx):
        features, label, _ = batch
        logits, _ = self(features)

        loss = self.criterion(logits, label.float())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        probs = torch.sigmoid(logits)
        self.train_metrics.update(probs, label)

        self.log_dict(self.train_metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, label, _ = batch
        logits, _ = self(features)
        loss = self.criterion(logits, label.float())

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, label)
        self.log("val/loss", loss, prog_bar=True)
        self.log_dict(self.val_metrics, on_epoch=True)

    def test_step(self, batch, batch_idx):
        features, label = batch
        logits, _ = self(features)
        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, label)
        self.log_dict(self.test_metrics, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        features, label, _ = batch
        logits, attention_weights = self(features)
        probs = torch.sigmoid(logits)

        return {"probs": probs, "attention_weights": attention_weights, "labels": label}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
