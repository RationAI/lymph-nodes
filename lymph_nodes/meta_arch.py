import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC


class MetaArch(LightningModule):
    """Attention-Based Multiple Instance Learning (ABMIL) model for slide-level classification.

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning" (ICML 2018).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Gated attention mechanism
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1))
        self.criterion = nn.BCEWithLogitsLoss()

        metrics = MetricCollection({"AUROC": BinaryAUROC()})
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: tile embeddings for a single bag, shape [n_tiles, embed_dim]

        Returns:
            logit: slide-level logit, scalar
            attention: per-tile attention weights, shape [n_tiles]
        """
        v = self.attention_V(x)  # [n_tiles, hidden_dim]
        u = self.attention_U(x)  # [n_tiles, hidden_dim]
        a = self.attention_weights(v * u)  # [n_tiles, 1]
        a = torch.softmax(a, dim=0)  # [n_tiles, 1]
        z = (a * x).sum(dim=0, keepdim=True)  # [1, embed_dim]
        return self.classifier(z).squeeze(), a.squeeze(1)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> Tensor:
        inputs, targets = batch
        inputs = inputs.squeeze(0)  # [n_tiles, embed_dim]
        targets = targets.squeeze(0)  # scalar

        logits, _ = self(inputs)
        loss = self.criterion(logits, targets.float())
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.train_metrics.update(
            torch.sigmoid(logits).unsqueeze(0), targets.unsqueeze(0)
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> None:
        inputs, targets = batch
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)

        logits, _ = self(inputs)
        loss = self.criterion(logits, targets.float())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(
            torch.sigmoid(logits).unsqueeze(0), targets.unsqueeze(0)
        )

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int = 0) -> None:
        inputs, targets = batch
        inputs = inputs.squeeze(0)
        targets = targets.squeeze(0)

        logits, _ = self(inputs)
        self.test_metrics.update(
            torch.sigmoid(logits).unsqueeze(0), targets.unsqueeze(0)
        )

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
