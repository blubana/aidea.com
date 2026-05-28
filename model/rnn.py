from __future__ import annotations

import torch
import torch.nn as nn


class RallyRNN(nn.Module):
    """Shared-encoder GRU/LSTM baseline with the same heads as RallyTransformer."""

    def __init__(
        self,
        num_tokens_per_feature: list[int],
        n_act: int,
        n_pt: int,
        d_model: int = 128,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        emb_dim: int = 32,
        rnn_type: str = "gru",
        bidirectional: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type={rnn_type!r}")

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        num_features = len(num_tokens_per_feature)

        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=0)
            for n in num_tokens_per_feature
        ])
        self.input_norm = nn.LayerNorm(num_features * emb_dim)
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=num_features * emb_dim,
            hidden_size=d_model,
            num_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout if num_encoder_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = d_model * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.action_head = nn.Linear(out_dim, n_act)
        self.point_head = nn.Linear(out_dim, n_pt)
        self.rally_head = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, X: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_parts = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(emb_parts, dim=-1)
        x = self.input_norm(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.detach().cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=X.size(1))
        out = self.drop(out)

        logits_action = self.action_head(out)
        logits_point = self.point_head(out)

        B, T, _ = out.shape
        positions = torch.arange(T, device=X.device).unsqueeze(0)
        pad_mask = positions >= lengths.unsqueeze(1)
        non_pad = (~pad_mask).float().unsqueeze(-1)
        mean_h = (out * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1.0)
        last_idx = (lengths - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, out.size(-1))
        last_h = out.gather(1, last_idx).squeeze(1)
        logit_rally = self.rally_head(torch.cat([last_h, mean_h], dim=-1)).squeeze(-1)
        return logits_action, logits_point, logit_rally

    def get_layer_wise_lr_params(self, base_lr: float, lr_decay: float = 0.9) -> list[dict]:
        return [
            {
                "params": list(self.embs.parameters()) + list(self.input_norm.parameters()),
                "lr": base_lr * lr_decay,
                "name": "embeddings",
            },
            {"params": list(self.rnn.parameters()), "lr": base_lr, "name": "rnn"},
            {
                "params": list(self.action_head.parameters())
                + list(self.point_head.parameters())
                + list(self.rally_head.parameters()),
                "lr": base_lr,
                "name": "heads",
            },
        ]
