"""PyTorch LSTM models for rally prefix classification."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
        prob = log_prob.exp()
        target = target.long()
        focal = (1.0 - prob.gather(1, target.unsqueeze(1)).squeeze(1)).pow(self.gamma)
        ce = torch.nn.functional.nll_loss(log_prob, target, weight=self.alpha, reduction="none")
        loss = focal * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class RallyLSTMClassifier(nn.Module):
    def __init__(
        self,
        cat_cardinalities: Sequence[int],
        num_numeric_features: int,
        manual_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        output_dim: int = 19,
        num_proj_dim: int = 32,
        mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, min(32, max(4, (cardinality + 1) // 2)), padding_idx=0) for cardinality in cat_cardinalities]
        )
        emb_dim = sum(emb.embedding_dim for emb in self.embeddings)
        self.num_proj = nn.Linear(num_numeric_features, num_proj_dim) if num_numeric_features > 0 and num_proj_dim > 0 else None
        lstm_input_dim = emb_dim + (num_proj_dim if self.num_proj is not None else num_numeric_features)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        head_in = hidden_size + manual_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor, lengths: torch.Tensor, x_manual: torch.Tensor) -> torch.Tensor:
        cat_parts = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        seq_in = torch.cat(cat_parts, dim=-1) if cat_parts else x_num
        if self.num_proj is not None:
            seq_in = torch.cat([seq_in, self.num_proj(x_num)], dim=-1)
        elif x_num.shape[-1] > 0:
            seq_in = torch.cat([seq_in, x_num], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(seq_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        seq_repr = h_n[-1]
        # x_manual is already standardized per training fold in train_lstm.py.
        # Avoid BatchNorm here: early validation can be unstable because running
        # statistics are poorly estimated after only one or two epochs.
        fused = torch.cat([seq_repr, x_manual], dim=-1) if x_manual.shape[-1] > 0 else seq_repr
        return self.head(fused)


class RallyLSTMServerBinary(RallyLSTMClassifier):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["output_dim"] = 1
        super().__init__(*args, **kwargs)
