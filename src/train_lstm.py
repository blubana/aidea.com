"""Train LSTM models from the prepared NPZ dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from models_lstm import RallyLSTMClassifier, RallyLSTMServerBinary


TARGET_MAP = {
    "actionId": {"key": "y_action", "binary": False, "output_dim": 19},
    "pointId": {"key": "y_point", "binary": False, "output_dim": 10},
    "serverGetPoint": {"key": "y_server", "binary": True, "output_dim": 2},
}

PRESETS = {
    "default": {"epochs": 12, "batch_size": 256, "hidden_size": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "patience": 5},
    "action": {"epochs": 20, "batch_size": 256, "hidden_size": 160, "dropout": 0.25, "lr": 8e-4, "weight_decay": 1e-4, "patience": 5},
    "point": {"epochs": 20, "batch_size": 256, "hidden_size": 128, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-4, "patience": 5},
    "server": {"epochs": 16, "batch_size": 256, "hidden_size": 96, "dropout": 0.15, "lr": 1e-3, "weight_decay": 5e-5, "patience": 5},
}

PRESET_FLAGS = {
    "epochs": "--epochs",
    "batch_size": "--batch-size",
    "hidden_size": "--hidden-size",
    "dropout": "--dropout",
    "lr": "--lr",
    "weight_decay": "--weight-decay",
    "patience": "--patience",
}


class RallyDataset(Dataset):
    def __init__(self, x_cat, x_num, lengths, x_manual, y=None, w=None):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.x_manual = torch.as_tensor(x_manual, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y)
        self.w = None if w is None else torch.as_tensor(w, dtype=torch.float32)

    def __len__(self):
        return len(self.x_cat)

    def __getitem__(self, idx):
        items = [self.x_cat[idx], self.x_num[idx], self.lengths[idx], self.x_manual[idx]]
        if self.y is not None:
            items.append(self.y[idx])
        if self.w is not None:
            items.append(self.w[idx])
        return tuple(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/lstm_dataset.npz")
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--target", required=True, choices=list(TARGET_MAP))
    parser.add_argument("--preset", default="default", choices=list(PRESETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-metric", default="auto", choices=["auto", "valid_loss", "macro_f1", "roc_auc"])
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--model-dir", default="models/lstm")
    parser.add_argument("--report-dir", default="reports/lstm")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-sample-weight", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    apply_preset_overrides(args, sys.argv[1:])
    return args


def apply_preset_overrides(args: argparse.Namespace, argv: list[str]) -> None:
    preset_name = args.preset if args.preset != "default" else ("server" if args.target == "serverGetPoint" else args.target.replace("Id", ""))
    values = PRESETS["default"].copy()
    values.update(PRESETS.get(preset_name, {}))
    for attr, flag in PRESET_FLAGS.items():
        if flag not in argv:
            setattr(args, attr, values[attr])


def standardize_manual(train_x: np.ndarray, valid_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((train_x - mean) / std).astype(np.float32), ((valid_x - mean) / std).astype(np.float32), mean, std


def standardize_seq_num(train_x: np.ndarray, valid_x: np.ndarray, train_lengths: np.ndarray, valid_lengths: np.ndarray):
    train_mask = np.arange(train_x.shape[1])[None, :] < train_lengths[:, None]
    valid_mask = np.arange(valid_x.shape[1])[None, :] < valid_lengths[:, None]
    flat = train_x[train_mask]
    mean = flat.mean(axis=0, keepdims=True) if len(flat) else np.zeros((1, train_x.shape[-1]), dtype=np.float32)
    std = flat.std(axis=0, keepdims=True) if len(flat) else np.ones((1, train_x.shape[-1]), dtype=np.float32)
    std[std < 1e-6] = 1.0
    train_out = train_x.copy().astype(np.float32)
    valid_out = valid_x.copy().astype(np.float32)
    train_out[train_mask] = (train_out[train_mask] - mean) / std
    valid_out[valid_mask] = (valid_out[valid_mask] - mean) / std
    return train_out, valid_out, mean.astype(np.float32), std.astype(np.float32)


def make_model(x_cat: np.ndarray, x_num: np.ndarray, x_manual: np.ndarray, args, output_dim: int, binary: bool):
    cardinals = [int(x_cat[:, :, i].max()) + 1 for i in range(x_cat.shape[-1])]
    cls = RallyLSTMServerBinary if binary else RallyLSTMClassifier
    return cls(
        cat_cardinalities=cardinals,
        num_numeric_features=x_num.shape[-1],
        manual_dim=x_manual.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=output_dim,
    )


def to_prob_2d(probs: np.ndarray, binary: bool) -> np.ndarray:
    if not binary:
        return probs
    probs = probs.reshape(-1)
    return np.column_stack([1.0 - probs, probs])


def evaluate_probs(y_true, probs_2d, binary: bool):
    pred = probs_2d.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probs_2d, labels=list(range(probs_2d.shape[1]))),
    }
    if binary:
        metrics["roc_auc"] = roc_auc_score(y_true, probs_2d[:, 1]) if len(np.unique(y_true)) > 1 else None
    return metrics


def early_stop_metric_name(args: argparse.Namespace, binary: bool) -> str:
    if args.early_stop_metric != "auto":
        return args.early_stop_metric
    return "roc_auc" if binary else "macro_f1"


def is_better(metric_name: str, value: float, best: float, min_delta: float) -> bool:
    if metric_name == "valid_loss":
        return value < (best - min_delta)
    return value > (best + min_delta)


def make_loss_fn(binary: bool, class_weight=None):
    if binary:
        return torch.nn.BCEWithLogitsLoss(reduction="none")
    return torch.nn.CrossEntropyLoss(weight=class_weight, reduction="none")


def batch_loss(model, batch, criterion, device, binary: bool, use_sample_weight: bool):
    x_cat, x_num, lengths, x_manual, y, w = [b.to(device) for b in batch]
    logits = model(x_cat, x_num, lengths, x_manual)
    losses = criterion(logits.squeeze(1), y.float()) if binary else criterion(logits, y.long())
    if use_sample_weight:
        losses = losses * w
    return losses.mean(), logits, y


def predict_probs(model, loader, device, binary: bool):
    model.eval()
    probs_list, y_list, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            x_cat, x_num, lengths, x_manual, y, _ = [b.to(device) for b in batch]
            logits = model(x_cat, x_num, lengths, x_manual)
            probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy() if binary else torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            y_list.append(y.cpu().numpy())
    return np.concatenate(y_list), np.concatenate(probs_list)


def train_one_fold(model, train_loader, valid_loader, args, device, binary: bool, class_weight=None):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = make_loss_fn(binary, class_weight)
    metric_name = early_stop_metric_name(args, binary)
    best_score = float("inf") if metric_name == "valid_loss" else -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _, _ = batch_loss(model, batch, criterion, device, binary, args.use_sample_weight)
            train_losses.append(float(loss.item()))
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for batch in valid_loader:
                loss, _, _ = batch_loss(model, batch, criterion, device, binary, args.use_sample_weight)
                valid_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("nan")
        y_epoch, probs_epoch_raw = predict_probs(model, valid_loader, device, binary)
        epoch_metrics = evaluate_probs(y_epoch, to_prob_2d(probs_epoch_raw, binary), binary)
        current_score = valid_loss if metric_name == "valid_loss" else float(epoch_metrics[metric_name])
        improved = is_better(metric_name, current_score, best_score, args.min_delta)
        if improved:
            best_loss = valid_loss
            best_score = current_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"fold epoch {epoch:02d}: train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
            f"macro_f1={epoch_metrics['macro_f1']:.5f}"
            + (f" roc_auc={epoch_metrics['roc_auc']:.5f}" if binary and epoch_metrics.get("roc_auc") is not None else "")
            + f" best_{metric_name}={best_score:.5f}"
        )
        if bad_epochs >= args.patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    y_true, probs_raw = predict_probs(model, valid_loader, device, binary)
    return best_state, best_epoch, best_loss, best_score, metric_name, y_true, to_prob_2d(probs_raw, binary)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = TARGET_MAP[args.target]
    data = np.load(args.data, allow_pickle=True)
    meta_df = pd.read_csv(args.train_features, usecols=["sample_id"])

    x_cat = data["X_cat_seq"]
    x_num = data["X_num_seq"]
    lengths = data["lengths"]
    x_manual = data["X_manual"]
    y = data[spec["key"]].astype(np.int64 if not spec["binary"] else np.int64)
    weights = data["sample_weight"].astype(np.float32) if "sample_weight" in data else np.ones(len(y), dtype=np.float32)
    groups = data["groups_match"]
    sample_ids = meta_df["sample_id"].astype(str).to_numpy()

    if args.sample and args.sample < len(y):
        idx = np.random.choice(len(y), size=args.sample, replace=False)
        x_cat, x_num, lengths, x_manual, y, weights, groups, sample_ids = [arr[idx] for arr in (x_cat, x_num, lengths, x_manual, y, weights, groups, sample_ids)]

    n_splits = min(args.folds, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(x_cat, y, groups))
    if args.max_folds > 0:
        splits = splits[: args.max_folds]

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output_dim = spec["output_dim"]
    oof_proba = np.zeros((len(y), output_dim), dtype=np.float32)
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    metrics = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold}/{len(splits)} target={args.target} ===")
        x_train_manual, x_valid_manual, manual_mean, manual_std = standardize_manual(x_manual[train_idx], x_manual[valid_idx])
        x_train_num, x_valid_num, num_mean, num_std = standardize_seq_num(x_num[train_idx], x_num[valid_idx], lengths[train_idx], lengths[valid_idx])
        train_ds = RallyDataset(x_cat[train_idx], x_train_num, lengths[train_idx], x_train_manual, y[train_idx], weights[train_idx])
        valid_ds = RallyDataset(x_cat[valid_idx], x_valid_num, lengths[valid_idx], x_valid_manual, y[valid_idx], weights[valid_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
        model = make_model(x_cat, x_num, x_manual, args, 1 if spec["binary"] else output_dim, spec["binary"])

        class_weight = None
        if not spec["binary"]:
            counts = np.bincount(y[train_idx].astype(np.int64), minlength=output_dim)
            class_weight = torch.as_tensor(len(train_idx) / np.maximum(counts, 1) / output_dim, dtype=torch.float32, device=device)

        best_state, best_epoch, best_valid_loss, best_score, metric_name, y_true, probs = train_one_fold(
            model, train_loader, valid_loader, args, device, spec["binary"], class_weight
        )
        fold_metrics = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "early_stop_metric": metric_name,
            "best_score": best_score,
            "valid_rows": int(len(valid_idx)),
            **evaluate_probs(y_true, probs, spec["binary"]),
        }
        metrics.append(fold_metrics)
        oof_proba[valid_idx] = probs.astype(np.float32)
        oof_pred[valid_idx] = probs.argmax(axis=1)

        torch.save(
            {
                "model_state_dict": best_state,
                "target": args.target,
                "manual_mean": manual_mean,
                "manual_std": manual_std,
                "num_mean": num_mean,
                "num_std": num_std,
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "early_stop_metric": metric_name,
                "best_score": best_score,
            },
            model_dir / f"{args.target}_fold{fold}.pt",
        )

    valid_mask = oof_pred >= 0
    summary = {
        "target": args.target,
        "device": str(device),
        "use_sample_weight": args.use_sample_weight,
        "preset": args.preset,
        "folds": metrics,
        "fold_average": {
            key: float(np.mean([m[key] for m in metrics if m.get(key) is not None]))
            for key in metrics[0]
            if key not in {"fold", "valid_rows", "best_epoch", "early_stop_metric"}
            and any(m.get(key) is not None for m in metrics)
        },
        "oof": evaluate_probs(y[valid_mask], oof_proba[valid_mask], spec["binary"]),
    }
    (report_dir / f"{args.target}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(report_dir / f"{args.target}_oof_proba.npy", oof_proba)
    pd.DataFrame({"sample_id": sample_ids, "y_true": y, "oof_pred": oof_pred}).to_csv(
        report_dir / f"{args.target}_oof_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
