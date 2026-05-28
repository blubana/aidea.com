"""Shared-encoder Multi-task Transformer with RoPE for table tennis rally prediction.

Architecture:
  - Feature embedding layer (one Embedding per categorical feature, projected to d_model)
  - Shared Transformer Encoder (RoPETransformerEncoderLayer × num_layers)
  - Three task heads:
      * action_head  → per-timestep logits [B, T, n_act]
      * point_head   → per-timestep logits [B, T, n_pt]
      * rally_head   → rally-level scalar  [B]  (masked mean-pool)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_emb


class RoPETransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with RoPE-augmented self-attention.

    Uses pre-norm (LayerNorm before attention and FFN) for training stability,
    and injects Rotary Position Embeddings into Q and K before computing
    scaled dot-product attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        # RoPE — one per layer, applied on head_dim
        self.rotary_emb = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        # Pre-norm layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, d_model]
            src_key_padding_mask: [batch, seq_len] bool, True = padding position

        Returns:
            [batch, seq_len, d_model]
        """
        # --- Self-attention (pre-norm) ---
        residual = x
        x = self.norm1(x)

        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(T, x.device)
        Q, K = apply_rotary_emb(Q, K, cos, sin)

        # Build attention bias from padding mask
        attn_bias: torch.Tensor | None = None
        if src_key_padding_mask is not None:
            # [B, 1, 1, T] — broadcast over (heads, query-positions)
            attn_bias = torch.zeros(B, 1, 1, T, device=x.device, dtype=x.dtype)
            attn_bias = attn_bias.masked_fill(
                src_key_padding_mask[:, None, None, :],
                float("-inf"),
            )
        if causal:
            causal_bias = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
            causal_bias = torch.triu(causal_bias, diagonal=1)[None, None, :, :]
            attn_bias = causal_bias if attn_bias is None else attn_bias + causal_bias

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        attn_out = self.out_proj(attn_out)
        x = residual + self.drop(attn_out)

        # --- Feed-forward (pre-norm) ---
        residual = x
        x = self.norm2(x)
        x = residual + self.drop(self.ffn(x))

        return x


class RallyTransformer(nn.Module):
    """Multi-task Transformer for table tennis rally prediction.

    Shared encoder predicts:
      1. actionId  (next-strike ball type)   — per-timestep classification
      2. pointId   (next-strike landing zone) — per-timestep classification
      3. serverGetPoint (rally outcome)       — rally-level binary classification
    """

    def __init__(
        self,
        num_tokens_per_feature: list[int],
        n_act: int,
        n_pt: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        emb_dim: int = 32,
        rally_pool: str = "last_mean_mlp",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.rally_pool = rally_pool
        num_features = len(num_tokens_per_feature)

        # One embedding table per feature; padding_idx=0 maps to zero vector
        self.embs = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=0)
            for n in num_tokens_per_feature
        ])

        # Project concatenated embeddings → d_model
        self.input_proj = nn.Linear(num_features * emb_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        # Shared encoder stack
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_encoder_layers)
        ])

        # Final layer norm
        self.encoder_norm = nn.LayerNorm(d_model)

        # Task heads
        self.action_head = nn.Linear(d_model, n_act)
        self.point_head  = nn.Linear(d_model, n_pt)
        if rally_pool == "mean_linear":
            self.rally_head = nn.Linear(d_model, 1)
        elif rally_pool == "last_mean_mlp":
            self.rally_head = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            raise ValueError(f"Unsupported rally_pool={rally_pool!r}")

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def _build_padding_mask(self, X: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Build boolean padding mask: True = padding position.

        Args:
            X:       [batch, seq_len, num_features]
            lengths: [batch] — actual sequence length per sample

        Returns:
            [batch, seq_len] bool
        """
        B, T, _ = X.shape
        positions = torch.arange(T, device=X.device).unsqueeze(0)  # [1, T]
        mask = positions >= lengths.unsqueeze(1)                     # [B, T]
        return mask

    def forward(
        self,
        X: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            X:       [batch, seq_len, num_features]  int64
            lengths: [batch]                          int64

        Returns:
            logits_action: [batch, seq_len, n_act]
            logits_point:  [batch, seq_len, n_pt]
            logit_rally:   [batch]
        """
        B, T, F = X.shape

        # Feature embeddings
        emb_parts = [emb(X[:, :, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(emb_parts, dim=-1)          # [B, T, F*emb_dim]

        x = self.input_proj(x)                    # [B, T, d_model]
        x = self.input_norm(x)
        x = self.input_drop(x)

        # Padding mask
        pad_mask = self._build_padding_mask(X, lengths)  # [B, T]

        # Shared encoder
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        x = self.encoder_norm(x)                  # [B, T, d_model]

        # Task heads
        logits_action = self.action_head(x)       # [B, T, n_act]
        logits_point  = self.point_head(x)        # [B, T, n_pt]

        # Rally head: combine global context with the final observed strike.
        non_pad = (~pad_mask).float().unsqueeze(-1)          # [B, T, 1]
        denom   = non_pad.sum(dim=1).clamp(min=1.0)         # [B, 1]
        mean_h  = (x * non_pad).sum(dim=1) / denom          # [B, d_model]
        if self.rally_pool == "mean_linear":
            rally_h = mean_h
        else:
            last_idx = (lengths - 1).clamp_min(0).view(B, 1, 1).expand(B, 1, self.d_model)
            last_h = x.gather(1, last_idx).squeeze(1)
            rally_h = torch.cat([last_h, mean_h], dim=-1)
        logit_rally = self.rally_head(rally_h).squeeze(-1)    # [B]

        return logits_action, logits_point, logit_rally

    def get_layer_wise_lr_params(
        self,
        base_lr: float,
        lr_decay: float = 0.9,
    ) -> list[dict]:
        """Return optimizer parameter groups with layer-wise learning rate decay.

        Strategy:
          - Embedding + input projection: base_lr * decay^(num_layers)
          - Encoder layer i (0-indexed from bottom): base_lr * decay^(num_layers - i - 1)
          - Task heads + final norm: base_lr  (no decay)

        Args:
            base_lr:   Learning rate for the top-most layers (task heads).
            lr_decay:  Multiplicative decay factor per layer down.

        Returns:
            List of dicts compatible with torch.optim parameter groups.
        """
        n = self.num_encoder_layers
        param_groups: list[dict] = []

        # Embedding + input projection group
        emb_params = (
            list(self.embs.parameters())
            + list(self.input_proj.parameters())
            + list(self.input_norm.parameters())
        )
        param_groups.append({
            "params": emb_params,
            "lr": base_lr * (lr_decay ** n),
            "name": "embedding_and_projection",
        })

        # Encoder layer groups (bottom = slowest)
        for i, layer in enumerate(self.encoder_layers):
            lr = base_lr * (lr_decay ** (n - i - 1))
            param_groups.append({
                "params": list(layer.parameters()),
                "lr": lr,
                "name": f"encoder_layer_{i}",
            })

        # Task heads + final encoder norm (top = fastest)
        head_params = (
            list(self.encoder_norm.parameters())
            + list(self.action_head.parameters())
            + list(self.point_head.parameters())
            + list(self.rally_head.parameters())
        )
        param_groups.append({
            "params": head_params,
            "lr": base_lr,
            "name": "task_heads",
        })

        return param_groups
