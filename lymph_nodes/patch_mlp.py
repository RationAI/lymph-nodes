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


class PatchMLP(LightningModule):
    """Patch-level MLP classifier for tile embeddings.

    Every tile is classified independently.  Per-tile probabilities together
    with spatial coordinates (tile_x, tile_y) from the metadata can be used
    directly to build heatmaps over the whole slide.
    """

    def __init__(
        self,
        foundation: str = "prov-gigapath",
        hidden_dims: list[int] | None = None,
        dropout: float = 0.25,
        lr: float = 1e-4,
        pos_weight: float = 1.0,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        if input_dim is None:
            input_dim = _input_dim_for_foundation(foundation)

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.lr = lr

        # Build MLP layers
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))

        self.mlp = nn.Sequential(*layers)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        embeddings, labels, _ = batch
        logits = self(embeddings)
        loss = self.criterion(logits, labels)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )

        probs = torch.sigmoid(logits)
        self.train_metrics.update(probs, labels.long())
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        embeddings, labels, _ = batch
        logits = self(embeddings)
        loss = self.criterion(logits, labels)
        self.log("val/loss", loss, prog_bar=True, batch_size=len(labels))

        probs = torch.sigmoid(logits)
        self.val_metrics.update(probs, labels.long())
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        embeddings, labels, _ = batch
        logits = self(embeddings)
        probs = torch.sigmoid(logits)
        self.test_metrics.update(probs, labels.long())
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

    # ------------------------------------------------------------------
    # Predict — returns per-tile probs + coordinates for heatmaps
    # ------------------------------------------------------------------

    def predict_step(self, batch: tuple, batch_idx: int) -> dict:
        embeddings, labels, metadatas = batch
        logits = self(embeddings)
        probs = torch.sigmoid(logits)
        return {"probs": probs, "labels": labels, "metadatas": metadatas}

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
