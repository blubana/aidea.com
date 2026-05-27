from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import FEATURES, IGNORE_INDEX, build_datasets, sort_rallies
from model.pcgrad import PCGrad
from model.transformer import RallyTransformer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def banner(console: Console) -> None:
    console.print("[bold cyan]+--------------------------------------------+[/]")
    console.print("[bold cyan]|  AI CUP Rally RoPE Transformer Trainer    |[/]")
    console.print("[bold cyan]+--------------------------------------------+[/]")


def feature_values(df: pd.DataFrame, col: str) -> pd.Series:
    if col == "strikeNumber":
        return df[col].clip(0, 50)
    if col in ("scoreSelf", "scoreOther"):
        return df[col].clip(0, 30)
    return df[col]


def fit_categories(train_df: pd.DataFrame) -> dict[str, list[int]]:
    cats: dict[str, list[int]] = {}
    for col in FEATURES:
        vals = pd.Series(feature_values(train_df, col).dropna().unique()).sort_values()
        cats[col] = vals.astype(int).tolist()
    return cats


def make_class_maps(train_df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    act_classes = sorted(int(v) for v in train_df["actionId"].dropna().unique())
    pt_classes = sorted(int(v) for v in train_df["pointId"].dropna().unique())
    act_id2idx = {v: i for i, v in enumerate(act_classes)}
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}
    return act_classes, pt_classes, act_id2idx, pt_id2idx


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    valid = y[y != IGNORE_INDEX]
    counts = np.bincount(valid.astype(np.int64), minlength=n_classes).astype(np.float32) + 1.0
    weights = 1.0 / counts
    weights = weights * (n_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)


def last_logits(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = (lengths - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, logits.size(-1))
    return logits.gather(1, idx).squeeze(1)


def unpack_batch(batch, device: torch.device):
    X, yA, yP, yR, L = batch
    return (
        X.to(device=device, dtype=torch.long),
        yA.to(device=device, dtype=torch.long),
        yP.to(device=device, dtype=torch.long),
        yR.to(device=device, dtype=torch.float32),
        L.to(device=device, dtype=torch.long),
    )


def evaluate(model, loader, losses, device: torch.device) -> dict[str, float]:
    ce_action, ce_point, bce_rally = losses
    model.eval()
    total_loss = 0.0
    all_a: list[int] = []
    all_ap: list[int] = []
    all_p: list[int] = []
    all_pp: list[int] = []
    all_r: list[float] = []
    all_rp: list[float] = []
    with torch.no_grad():
        for batch in loader:
            X, yA, yP, yR, L = unpack_batch(batch, device)
            la_seq, lp_seq, lr = model(X, L)
            la = last_logits(la_seq, L)
            lp = last_logits(lp_seq, L)
            loss = 0.4 * ce_action(la, yA) + 0.4 * ce_point(lp, yP) + 0.2 * bce_rally(lr, yR)
            total_loss += float(loss.item()) * X.size(0)

            a_true = yA.detach().cpu().numpy()
            p_true = yP.detach().cpu().numpy()
            a_pred = la.argmax(-1).detach().cpu().numpy()
            p_pred = lp.argmax(-1).detach().cpu().numpy()
            ma = a_true != IGNORE_INDEX
            mp = p_true != IGNORE_INDEX
            all_a.extend(a_true[ma].tolist())
            all_ap.extend(a_pred[ma].tolist())
            all_p.extend(p_true[mp].tolist())
            all_pp.extend(p_pred[mp].tolist())
            all_r.extend(yR.detach().cpu().tolist())
            all_rp.extend(torch.sigmoid(lr).detach().cpu().tolist())

    f1_a = f1_score(all_a, all_ap, average="macro", zero_division=0) if all_a else 0.0
    f1_p = f1_score(all_p, all_pp, average="macro", zero_division=0) if all_p else 0.0
    auc = roc_auc_score(all_r, all_rp) if len(set(all_r)) > 1 else 0.5
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "f1_action": float(f1_a),
        "f1_point": float(f1_p),
        "auc": float(auc),
        "score": float(0.4 * f1_a + 0.4 * f1_p + 0.2 * auc),
    }


def save_checkpoint(path: Path, model, optimizer, metadata: dict, epoch: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def main(args: argparse.Namespace) -> None:
    console = Console()
    banner(console)
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    train_csv = Path(args.train) if args.train else data_dir / "train.csv"
    train_df = sort_rallies(pd.read_csv(train_csv))
    if args.limit_rallies > 0:
        keep = train_df["rally_uid"].drop_duplicates().head(args.limit_rallies)
        train_df = train_df[train_df["rally_uid"].isin(keep)].copy()

    rally_labels = train_df.groupby("rally_uid", sort=False)["serverGetPoint"].first()
    rally_ids = rally_labels.index.to_numpy()
    stratify = rally_labels.to_numpy() if len(np.unique(rally_labels)) > 1 else None
    tr_ids, va_ids = train_test_split(
        rally_ids,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=stratify,
    )
    tr_df = train_df[train_df["rally_uid"].isin(tr_ids)].copy()
    va_df = train_df[train_df["rally_uid"].isin(va_ids)].copy()

    cats = fit_categories(tr_df)
    act_classes, pt_classes, act_id2idx, pt_id2idx = make_class_maps(tr_df)
    train_ds, val_ds, maxlen, _ = build_datasets(tr_df, va_df, cats, act_id2idx, pt_id2idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=args.device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=max(args.batch * 2, 128), shuffle=False, num_workers=args.workers, pin_memory=args.device == "cuda")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    num_tokens_per_feature = [len(cats[c]) + 1 for c in FEATURES]  # +1 for UNK token, PAD handled by model
    model = RallyTransformer(
        num_tokens_per_feature=num_tokens_per_feature,
        n_act=len(act_classes),
        n_pt=len(pt_classes),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
        max_seq_len=maxlen,
        emb_dim=args.emb_dim,
    ).to(device)

    act_w = class_weights(train_ds.yA, len(act_classes)).to(device)
    pt_w = class_weights(train_ds.yP, len(pt_classes)).to(device)
    ce_action = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=act_w)
    ce_point = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=pt_w)
    bce_rally = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.get_layer_wise_lr_params(args.lr, args.lr_decay), weight_decay=args.weight_decay)
    pcgrad = PCGrad(optimizer) if args.pcgrad else None

    metadata = {
        "features": FEATURES,
        "cats": cats,
        "action_classes": act_classes,
        "point_classes": pt_classes,
        "num_tokens_per_feature": num_tokens_per_feature,
        "maxlen": maxlen,
        "model_kwargs": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.layers,
            "dim_feedforward": args.ffn,
            "dropout": args.dropout,
            "emb_dim": args.emb_dim,
        },
    }

    table = Table(title="Run config")
    table.add_column("item")
    table.add_column("value")
    for k, v in {
        "device": str(device),
        "train samples": len(train_ds),
        "val samples": len(val_ds),
        "maxlen": maxlen,
        "classes": f"action={len(act_classes)}, point={len(pt_classes)}",
        "pcgrad": args.pcgrad,
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)

    best_score = -1.0
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    latest_path = out_dir / f"latest_transformer_{run_stamp}.pt"
    best_path = out_dir / f"best_transformer_{run_stamp}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, ascii=True)
        for batch in pbar:
            X, yA, yP, yR, L = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            la_seq, lp_seq, lr = model(X, L)
            la = last_logits(la_seq, L)
            lp = last_logits(lp_seq, L)
            loss_a = 0.4 * ce_action(la, yA)
            loss_p = 0.4 * ce_point(lp, yP)
            loss_r = 0.2 * bce_rally(lr, yR)
            loss = loss_a + loss_p + loss_r
            if pcgrad is not None:
                pcgrad.pc_backward([loss_a, loss_p, loss_r])
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item()) * X.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader.dataset))
        metrics = evaluate(model, val_loader, (ce_action, ce_point, bce_rally), device)
        metrics["train_loss"] = train_loss
        console.print(
            f"[bold]Epoch {epoch:02d}[/] train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"F1_action={metrics['f1_action']:.4f} F1_point={metrics['f1_point']:.4f} "
            f"AUC={metrics['auc']:.4f} score={metrics['score']:.4f}"
        )
        save_checkpoint(latest_path, model, optimizer, metadata, epoch, metrics)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            save_checkpoint(best_path, model, optimizer, metadata, epoch, metrics)
            console.print(f"[green]Saved best checkpoint:[/] {best_path}")

    (out_dir / f"metadata_{run_stamp}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[cyan]Latest checkpoint:[/] {latest_path}")
    console.print(f"[cyan]Best checkpoint:[/] {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--train", default="")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--limit-rallies", type=int, default=0, help="Debug/smoke-test limit; 0 uses all rallies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True)
    main(parser.parse_args())
