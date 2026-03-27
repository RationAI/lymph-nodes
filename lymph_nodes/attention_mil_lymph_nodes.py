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

from lymph_nodes.modeling.aggregators import (
    ABMILAggregator,
    CLAMAggregator,
    MaxPoolingAggregator,
    TransMILAggregator,
)
from lymph_nodes.typing import TileEmbeddingsInput


_MIL_TYPES = ("max_pooling", "abmil", "transmil", "clam")


def _input_dim_for_foundation(foundation: str) -> int:
    match foundation:
        case "prov-gigapath" | "uni2-h":
            return 1536
        case "uni":
            return 1024
        case "virchow" | "virchow2":
            return 2560
        case _:
            raise ValueError(f"Unknown foundation model: {foundation}")


class LymphNodesMIL(LightningModule):
    def __init__(
        self,
        mil_type: str = "abmil",
        foundation: str = "prov-gigapath",
        lr: float = 1e-4,
        pos_weight: float = 1.0,
        bag_weight: float = 0.7,
        input_dim: int | None = None,
    ):
        """Initializes the attention MIL module for lymph nodes.

        Args:
            mil_type: Aggregation strategy. One of: max_pooling, abmil, transmil, clam.
            foundation: Name of the tile encoder; used to infer input_dim when not set.
            lr: Learning rate for Adam.
            pos_weight: Positive-class weight for BCEWithLogitsLoss.
            bag_weight: CLAM only — weight of the instance loss relative to the bag loss.
                        Total loss = bag_loss + bag_weight * instance_loss.
            input_dim: Override the embedding dimensionality (inferred from foundation
                        by default).
        """
        super().__init__()
        self.save_hyperparameters()

        if mil_type not in _MIL_TYPES:
            raise ValueError(f"mil_type must be one of {_MIL_TYPES}, got '{mil_type}'")

        if input_dim is None:
            input_dim = _input_dim_for_foundation(foundation)

        self.lr = lr
        self.mil_type = mil_type
        self.bag_weight = bag_weight

        match mil_type:
            case "max_pooling":
                self.aggregator = MaxPoolingAggregator()
            case "abmil":
                self.aggregator = ABMILAggregator(input_dim)
            case "transmil":
                self.aggregator = TransMILAggregator(input_dim)
            case "clam":
                self.aggregator = CLAMAggregator(input_dim)

        self.classifier = nn.Linear(input_dim, 1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

        metrics = MetricCollection(
            {
                "AUC": BinaryAUROC(),
                "accuracy": BinaryAccuracy(),
                "sensitivity": BinaryRecall(),
                "specificity": BinarySpecificity(),
                "F1": BinaryF1Score(),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        mask = x.abs().sum(dim=-1) > 1e-6  # (B, N)
        M, A = self.aggregator(x, mask)
        logits = self.classifier(M).squeeze(-1)
        return logits, A

    def training_step(self, batch: TileEmbeddingsInput, batch_idx: int):
        features, label, _ = batch
        logits, A = self(features)

        bag_loss = self.criterion(logits, label.float())

        if self.mil_type == "clam":
            inst_loss = self.aggregator.compute_instance_loss(features, A, label)
            loss = bag_loss + self.bag_weight * inst_loss
            self.log(
                "train/instance_loss",
                inst_loss,
                on_step=True,
                on_epoch=True,
                batch_size=len(label),
            )
        else:
            loss = bag_loss

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(label),
        )

        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        self.train_metrics.update(probs, label.long())
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: TileEmbeddingsInput, batch_idx: int):
        features, label, _ = batch
        logits, _ = self(features)
        loss = self.criterion(logits, label.float())

        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        self.val_metrics.update(probs, label.long())
        self.log("val/loss", loss, prog_bar=True, batch_size=len(label))
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: TileEmbeddingsInput, batch_idx: int):
        features, label, _ = batch
        logits, _ = self(features)
        probs = torch.sigmoid(logits)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        self.test_metrics.update(probs, label.long())
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    def predict_step(self, batch: TileEmbeddingsInput, batch_idx: int):
        features, label, _ = batch
        logits, attention_weights = self(features)
        probs = torch.sigmoid(logits)

        return {"probs": probs, "attention_weights": attention_weights, "labels": label}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
