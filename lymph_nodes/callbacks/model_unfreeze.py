import pytorch_lightning as pl
import torch


class ModelUnfreeze(pl.Callback):
    def __init__(self, monitor="val_loss", patience=3, min_delta=1e-4):
        """
        Args:
            monitor: Metric to monitor.
            patience: Number of validation checks with no improvement after which encoder is unfrozen.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait_count = 0
        self.best_score = None
        self.frozen = True

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is None or not self.frozen:
            return

        current = current.item() if isinstance(current, torch.Tensor) else current

        if self.best_score is None or current < self.best_score - self.min_delta:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            print(f"Unfreezing encoder after {self.wait_count} bad epochs.")
            pl_module.unfreeze()
            self.frozen = False

    def on_fit_start(self, trainer, pl_module):
        # Initially freeze
        pl_module.freeze()
