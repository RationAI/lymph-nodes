import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


log = logging.getLogger(__name__)


class MILTrainer:
    def __init__(self, model, train_dataset, cfg):
        self.cfg = cfg
        self.device = torch.device(
            cfg.training.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)

        # --- COMMENT: BATCH SIZE ---
        # Note batch_size=1. In MIL, every bag has a different number of instances.
        # Standard PyTorch DataLoaders cannot stack tensors of different sizes
        # into a single batch without a complex 'collate_fn'.
        # Using batch_size=1 is the standard workaround.
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.criterion = nn.BCELoss()

    def run(self):
        log.info(f"Starting training on {self.device}")
        for epoch in range(self.cfg.training.epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for data, label in self.train_loader:
                data, label = data.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                # Forward pass
                # We ignore the 3rd return value (Attention weights) during training
                Y_prob, Y_hat, _ = self.model(data)

                loss = self.criterion(Y_prob, label)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                correct += (Y_hat == label).sum().item()
                total += 1

            acc = correct / total
            avg_loss = train_loss / len(self.train_loader)
            log.info(
                f"Epoch {epoch + 1}/{self.cfg.training.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}"
            )
