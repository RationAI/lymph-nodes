"""MIL aggregation modules.

Each aggregator takes (x, mask) and returns (M, A):
  x    : (B, N, D)  — tile embeddings (zero-padded bags)
  mask : (B, N)     — True for valid (non-padding) tiles
  M    : (B, D)     — slide-level bag representation
  A    : (B, N)     — per-tile attention / importance weights (for visualisation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaxPoolingAggregator(nn.Module):
    """Baseline: bag label determined by the single most informative tile.

    No learned attention; the max operation is the implicit selector.
    Returns uniform attention over valid tiles for visualisation.
    """

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # Replace padding with -inf so max ignores it
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        M = x_masked.max(dim=1).values  # (B, D)
        # Uniform attention over valid tiles for downstream heatmaps
        valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        A = mask.float() / valid_counts  # (B, N)
        return M, A


class ABMILAggregator(nn.Module):
    """Gated attention-based MIL (Ilse et al., 2018).

    Two parallel branches (tanh + sigmoid) gate each other before producing
    scalar attention scores that are softmax-normalised and used for weighted
    sum pooling.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)
        a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = torch.softmax(a, dim=1)  # (B, N, 1)
        M = torch.sum(A * x, dim=1)  # (B, D)
        return M, A.squeeze(-1)  # (B, D), (B, N)


class TransMILAggregator(nn.Module):
    """Transformer-based MIL (Shao et al., 2021).

    A learnable CLS token is prepended to the bag; the transformer encodes
    interactions between all tiles; the CLS output is used as the bag
    representation.  Padding tiles are hidden via src_key_padding_mask.

    Note: standard O(N²) attention is used here.  For very large bags this
    can be a memory bottleneck — reduce num_layers or num_heads if needed.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B, _, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x_cls = torch.cat([cls, x], dim=1)  # (B, N+1, D)

        # CLS is always valid; extend mask accordingly
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_valid, mask], dim=1)  # (B, N+1)

        # PyTorch convention: True in src_key_padding_mask means "ignore"
        out = self.transformer(x_cls, src_key_padding_mask=~full_mask)  # (B, N+1, D)
        M = out[:, 0, :]  # CLS token output (B, D)

        # Transformer has no scalar attention output; return uniform for viz
        valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        A = mask.float() / valid_counts  # (B, N)
        return M, A


class CLAMAggregator(nn.Module):
    """Clustering-constrained Attention MIL — single-branch variant (Lu et al., 2021).

    Uses the same gated attention as ABMIL for bag-level pooling, but adds a
    small instance classifier trained with a pseudo-label clustering loss:
      - positive bags: top-k tiles → label 1, bottom-k tiles → label 0
      - negative bags: top-k tiles → label 0, bottom-k tiles → label 0

    The instance loss is ONLY used during training and must be requested
    explicitly by calling compute_instance_loss().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        k_sample: int = 8,
    ):
        super().__init__()
        self.k_sample = k_sample
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)
        self.instance_classifier = nn.Linear(input_dim, 1)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights(a_v * a_u)  # (B, N, 1)
        a = a.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        A = torch.softmax(a, dim=1)  # (B, N, 1)
        M = torch.sum(A * x, dim=1)  # (B, D)
        return M, A.squeeze(-1)  # (B, D), (B, N)

    def compute_instance_loss(self, x: Tensor, a: Tensor, labels: Tensor) -> Tensor:
        """Compute per-bag instance loss and return the batch mean.

        Args:
            x:      (B, N, D) tile embeddings
            a:      (B, N)    attention weights (post-softmax, from forward())
            labels: (B,)      bag-level binary labels
        """
        B = x.size(0)
        inst_losses: list[Tensor] = []

        for i in range(B):
            a_i = a[i]  # (N,)
            x_i = x[i]  # (N, D)
            label_i = int(labels[i].item())

            # Number of valid instances in this bag
            n_valid = int((a_i > 0).sum().item())
            k = max(1, min(self.k_sample, n_valid))

            sorted_idx = torch.argsort(a_i, descending=True)
            top_k = sorted_idx[:k]
            bot_k = sorted_idx[n_valid - k : n_valid]

            top_logits = self.instance_classifier(x_i[top_k]).squeeze(-1)
            bot_logits = self.instance_classifier(x_i[bot_k]).squeeze(-1)

            ones = torch.ones(k, device=x.device)
            zeros = torch.zeros(k, device=x.device)

            if label_i == 1:
                # Positive bag: high-attention → positive, low-attention → negative
                loss = 0.5 * F.binary_cross_entropy_with_logits(top_logits, ones)
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                    bot_logits, zeros
                )
            else:
                # Negative bag: all tiles should be negative
                loss = 0.5 * F.binary_cross_entropy_with_logits(top_logits, zeros)
                loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
                    bot_logits, zeros
                )

            inst_losses.append(loss)

        return torch.stack(inst_losses).mean()
