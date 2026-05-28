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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.dataset import (
    IGNORE_INDEX,
    build_datasets,
    build_submission_like_dataset,
    feature_values,
    resolve_features,
    sort_rallies,
)
from model.pcgrad import PCGrad
from model.rnn import RallyRNN
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


def fit_categories(train_df: pd.DataFrame, features: list[str]) -> dict[str, list[int]]:
    cats: dict[str, list[int]] = {}
    for col in features:
        vals = pd.Series(feature_values(train_df, col).dropna().unique()).sort_values()
        cats[col] = vals.astype(int).tolist()
    return cats


def make_class_maps(train_df: pd.DataFrame) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    act_classes = sorted(int(v) for v in train_df["actionId"].dropna().unique())
    pt_classes = sorted(int(v) for v in train_df["pointId"].dropna().unique())
    act_id2idx = {v: i for i, v in enumerate(act_classes)}
    pt_id2idx = {v: i for i, v in enumerate(pt_classes)}
    return act_classes, pt_classes, act_id2idx, pt_id2idx


def class_weights(
    y: np.ndarray,
    n_classes: int,
    mode: str = "sqrt",
    max_weight: float = 3.0,
    effective_beta: float = 0.999,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    valid = y[y != IGNORE_INDEX]
    counts = np.bincount(valid.astype(np.int64), minlength=n_classes).astype(np.float32) + 1.0
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "effective":
        effective_num = 1.0 - np.power(effective_beta, counts)
        weights = (1.0 - effective_beta) / np.maximum(effective_num, 1e-12)
    else:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    weights = weights / weights.mean()
    if max_weight > 0:
        weights = np.clip(weights, 1.0 / max_weight, max_weight)
        weights = weights / weights.mean()
        weights = np.clip(weights, 1.0 / max_weight, max_weight)
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


def evaluate(
    model: RallyTransformer,
    loader: DataLoader,
    losses,
    device: torch.device,
    loss_weights: tuple[float, float, float],
) -> dict[str, float]:
    ce_action, ce_point, bce_rally = losses
    action_w, point_w, rally_w = loss_weights
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
            loss = action_w * ce_action(la, yA) + point_w * ce_point(lp, yP) + rally_w * bce_rally(lr, yR)
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
    action_point_score = 0.5 * f1_a + 0.5 * f1_p
    point_pred_unique = len(set(all_pp))
    point_true_unique = len(set(all_p))
    action_pred_unique = len(set(all_ap))
    action_true_unique = len(set(all_a))
    if all_pp:
        _, point_pred_counts = np.unique(np.asarray(all_pp), return_counts=True)
        point_top_pred_frac = float(point_pred_counts.max() / max(1, len(all_pp)))
    else:
        point_top_pred_frac = 0.0
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "f1_action": float(f1_a),
        "f1_point": float(f1_p),
        "auc": float(auc),
        "action_point_score": float(action_point_score),
        "action_pred_unique": float(action_pred_unique),
        "action_true_unique": float(action_true_unique),
        "point_pred_unique": float(point_pred_unique),
        "point_true_unique": float(point_true_unique),
        "point_top_pred_frac": point_top_pred_frac,
        "score": float(0.4 * f1_a + 0.4 * f1_p + 0.2 * auc),
    }


def checkpoint_selection_value(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "score":
        return metrics["score"]
    if metric_name == "action_point_score":
        return metrics["action_point_score"]
    if metric_name == "min_f1":
        return min(metrics["f1_action"], metrics["f1_point"])
    raise ValueError(f"Unsupported selection metric: {metric_name}")


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


def save_history_outputs(history: list[dict[str, float]], out_dir: Path, run_stamp: str, save_plots: bool) -> None:
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    csv_path = out_dir / f"history_{run_stamp}.csv"
    history_df.to_csv(csv_path, index=False)

    if not save_plots:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    epochs = history_df["epoch"]

    axes[0, 0].plot(epochs, history_df["train_loss"], marker="o", label="train_loss")
    axes[0, 0].plot(epochs, history_df["loss"], marker="o", label="submit_val_loss")
    if "dense_loss" in history_df:
        axes[0, 0].plot(epochs, history_df["dense_loss"], marker=".", label="dense_val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history_df["f1_action"], marker="o", label="Action Macro F1")
    axes[0, 1].plot(epochs, history_df["f1_point"], marker="o", label="Point Macro F1")
    axes[0, 1].set_title("Submission-like Macro F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history_df["auc"], marker="o", color="tab:green", label="ServerGetPoint AUC")
    axes[1, 0].set_title("Submission-like AUC")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history_df["score"], marker="o", color="tab:red", label="Submission-like score")
    if "dense_score" in history_df:
        axes[1, 1].plot(epochs, history_df["dense_score"], marker=".", color="tab:gray", label="Dense score")
    axes[1, 1].set_title("Overall Score = 0.4*F1_action + 0.4*F1_point + 0.2*AUC")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"training_curves_{run_stamp}.png", dpi=160)
    plt.close(fig)


def make_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer, train_loader: DataLoader):
    if args.scheduler == "none" or len(train_loader) == 0:
        return None
    if args.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
        )
    raise ValueError(f"Unsupported scheduler={args.scheduler!r}")


def build_sequence_model(
    args: argparse.Namespace,
    num_tokens_per_feature: list[int],
    n_act: int,
    n_pt: int,
    maxlen: int,
) -> nn.Module:
    if args.model_type == "transformer":
        return RallyTransformer(
            num_tokens_per_feature=num_tokens_per_feature,
            n_act=n_act,
            n_pt=n_pt,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.layers,
            dim_feedforward=args.ffn,
            dropout=args.dropout,
            max_seq_len=maxlen,
            emb_dim=args.emb_dim,
            rally_pool=args.rally_pool,
        )
    if args.model_type in {"gru", "lstm"}:
        return RallyRNN(
            num_tokens_per_feature=num_tokens_per_feature,
            n_act=n_act,
            n_pt=n_pt,
            d_model=args.d_model,
            num_encoder_layers=args.layers,
            dropout=args.dropout,
            emb_dim=args.emb_dim,
            rnn_type=args.model_type,
            bidirectional=args.bidirectional_rnn,
        )
    raise ValueError(f"Unsupported model_type={args.model_type!r}")


def train_one_split(
    args: argparse.Namespace,
    console: Console,
    train_df: pd.DataFrame,
    tr_ids: np.ndarray,
    va_ids: np.ndarray,
    run_label: str = "",
) -> dict[str, float]:
    tr_df = train_df[train_df["rally_uid"].isin(tr_ids)].copy()
    va_df = train_df[train_df["rally_uid"].isin(va_ids)].copy()
    features = resolve_features(args.feature_set)

    cats = fit_categories(tr_df, features)
    act_classes, pt_classes, act_id2idx, pt_id2idx = make_class_maps(tr_df)
    train_ds, dense_val_ds, maxlen, _ = build_datasets(tr_df, va_df, cats, act_id2idx, pt_id2idx, features)
    submit_val_ds = build_submission_like_dataset(
        va_df,
        cats,
        act_id2idx,
        pt_id2idx,
        maxlen,
        features,
        target_from_end=args.val_target_from_end,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin_memory)
    dense_val_loader = DataLoader(
        dense_val_ds,
        batch_size=max(args.batch * 2, 128),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    submit_val_loader = DataLoader(
        submit_val_ds,
        batch_size=max(args.batch * 2, 128),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    num_tokens_per_feature = [len(cats[c]) + 1 for c in features]
    model = build_sequence_model(args, num_tokens_per_feature, len(act_classes), len(pt_classes), maxlen).to(device)

    act_w = class_weights(
        train_ds.yA,
        len(act_classes),
        mode=args.class_weight_mode,
        max_weight=args.class_weight_max,
        effective_beta=args.class_weight_effective_beta,
    )
    pt_w = class_weights(
        train_ds.yP,
        len(pt_classes),
        mode=args.class_weight_mode,
        max_weight=args.class_weight_max,
        effective_beta=args.class_weight_effective_beta,
    )
    act_w = act_w.to(device) if act_w is not None else None
    pt_w = pt_w.to(device) if pt_w is not None else None
    ce_action = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=act_w, label_smoothing=args.label_smoothing)
    ce_point = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=pt_w, label_smoothing=args.label_smoothing)
    bce_rally = nn.BCEWithLogitsLoss()
    loss_weights = (args.action_loss_weight, args.point_loss_weight, args.rally_loss_weight)

    optimizer = torch.optim.AdamW(model.get_layer_wise_lr_params(args.lr, args.lr_decay), weight_decay=args.weight_decay)
    scheduler = make_scheduler(args, optimizer, train_loader)
    pcgrad = PCGrad(optimizer) if args.pcgrad else None

    metadata = {
        "features": features,
        "feature_set": args.feature_set,
        "cats": cats,
        "action_classes": act_classes,
        "point_classes": pt_classes,
        "num_tokens_per_feature": num_tokens_per_feature,
        "maxlen": maxlen,
        "model_type": args.model_type,
        "model_kwargs": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.layers,
            "dim_feedforward": args.ffn,
            "dropout": args.dropout,
            "emb_dim": args.emb_dim,
            "rally_pool": args.rally_pool,
            "rnn_type": args.model_type if args.model_type in {"gru", "lstm"} else "gru",
            "bidirectional": args.bidirectional_rnn,
        },
        "loss_weights": {
            "action": args.action_loss_weight,
            "point": args.point_loss_weight,
            "rally": args.rally_loss_weight,
        },
        "label_smoothing": args.label_smoothing,
        "class_weight_mode": args.class_weight_mode,
        "class_weight_max": args.class_weight_max,
        "selection_metric": args.selection_metric,
        "validation_mode": "submission_like_prefix",
        "val_target_from_end": args.val_target_from_end,
        "run_label": run_label,
    }

    table = Table(title=f"Run config{f' ({run_label})' if run_label else ''}")
    table.add_column("item")
    table.add_column("value")
    for k, v in {
        "device": str(device),
        "train samples": len(train_ds),
        "dense val samples": len(dense_val_ds),
        "submission-like val samples": len(submit_val_ds),
        "val target from end": args.val_target_from_end,
        "maxlen": maxlen,
        "classes": f"action={len(act_classes)}, point={len(pt_classes)}",
        "model type": args.model_type,
        "feature set": args.feature_set,
        "pcgrad": f"{args.pcgrad} ({args.pcgrad_rally_mode})",
        "scheduler": args.scheduler,
        "label smoothing": args.label_smoothing,
        "class weights": f"{args.class_weight_mode}, cap={args.class_weight_max}",
        "loss weights": loss_weights,
        "selection metric": args.selection_metric,
        "rally pool": args.rally_pool,
        "features": ", ".join(features),
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)

    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    if run_label:
        out_dir = out_dir / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / f"latest_transformer_{run_stamp}.pt"
    best_path = out_dir / f"best_transformer_{run_stamp}.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_a = 0.0
        running_p = 0.0
        running_r = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False, ascii=True)
        for batch in pbar:
            X, yA, yP, yR, L = unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            la_seq, lp_seq, lr = model(X, L)
            la = last_logits(la_seq, L)
            lp = last_logits(lp_seq, L)
            loss_a = args.action_loss_weight * ce_action(la, yA)
            loss_p = args.point_loss_weight * ce_point(lp, yP)
            loss_r = args.rally_loss_weight * bce_rally(lr, yR)
            loss = loss_a + loss_p + loss_r

            if pcgrad is not None:
                if args.pcgrad_rally_mode == "separate":
                    pcgrad.pc_backward([loss_a, loss_p], retain_graph=True)
                    loss_r.backward()
                else:
                    pcgrad.pc_backward([loss_a, loss_p, loss_r])
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running += float(loss.item()) * X.size(0)
            running_a += float(loss_a.item()) * X.size(0)
            running_p += float(loss_p.item()) * X.size(0)
            running_r += float(loss_r.item()) * X.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[-1]['lr']:.2e}")

        train_loss = running / max(1, len(train_loader.dataset))
        train_loss_a = running_a / max(1, len(train_loader.dataset))
        train_loss_p = running_p / max(1, len(train_loader.dataset))
        train_loss_r = running_r / max(1, len(train_loader.dataset))
        metrics = evaluate(model, submit_val_loader, (ce_action, ce_point, bce_rally), device, loss_weights)
        dense_metrics = evaluate(model, dense_val_loader, (ce_action, ce_point, bce_rally), device, loss_weights)
        selection_value = checkpoint_selection_value(metrics, args.selection_metric)
        metrics.update(
            {
                "dense_loss": dense_metrics["loss"],
                "dense_f1_action": dense_metrics["f1_action"],
                "dense_f1_point": dense_metrics["f1_point"],
                "dense_auc": dense_metrics["auc"],
                "dense_action_point_score": dense_metrics["action_point_score"],
                "dense_score": dense_metrics["score"],
                "dense_point_pred_unique": dense_metrics["point_pred_unique"],
                "dense_point_top_pred_frac": dense_metrics["point_top_pred_frac"],
                "train_loss": train_loss,
                "train_loss_action": train_loss_a,
                "train_loss_point": train_loss_p,
                "train_loss_rally": train_loss_r,
                "selection_value": selection_value,
                "epoch": float(epoch),
            }
        )
        history.append({"epoch": epoch, **metrics})
        console.print(
            f"[bold]{f'[{run_label}] ' if run_label else ''}Epoch {epoch:02d}[/] "
            f"train_loss={train_loss:.4f} (A={train_loss_a:.4f}, P={train_loss_p:.4f}, R={train_loss_r:.4f}) "
            f"submit_val_loss={metrics['loss']:.4f} "
            f"F1_action={metrics['f1_action']:.4f} F1_point={metrics['f1_point']:.4f} "
            f"AUC={metrics['auc']:.4f} ap_score={metrics['action_point_score']:.4f} "
            f"score={metrics['score']:.4f} select={selection_value:.4f} dense_score={metrics['dense_score']:.4f} "
            f"point_pred={int(metrics['point_pred_unique'])}/{int(metrics['point_true_unique'])} "
            f"top_point={metrics['point_top_pred_frac']:.2f}"
        )

        save_checkpoint(latest_path, model, optimizer, metadata, epoch, metrics)
        if selection_value > best_score + args.early_stopping_min_delta:
            best_score = selection_value
            best_epoch = epoch
            best_metrics = metrics.copy()
            save_checkpoint(best_path, model, optimizer, metadata, epoch, metrics)
            console.print(f"[green]Saved best checkpoint:[/] {best_path}")
        elif args.early_stopping_patience > 0 and epoch - best_epoch >= args.early_stopping_patience:
            console.print(
                f"[yellow]Early stopping:[/] no submission-like score improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    (out_dir / f"metadata_{run_stamp}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    save_history_outputs(history, out_dir, run_stamp, args.plots)
    console.print(f"[cyan]Latest checkpoint:[/] {latest_path}")
    console.print(f"[cyan]Best checkpoint:[/] {best_path}")
    console.print(f"[cyan]History CSV:[/] {out_dir / f'history_{run_stamp}.csv'}")
    if args.plots:
        console.print(f"[cyan]Training curves:[/] {out_dir / f'training_curves_{run_stamp}.png'}")
    return best_metrics


def iter_cv_splits(rally_ids: np.ndarray, labels: np.ndarray, folds: int, repeats: int, seed: int):
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_ids, labels), start=1):
            run_label = f"repeat{repeat + 1}_fold{fold}"
            yield run_label, rally_ids[tr_idx], rally_ids[va_idx]


def print_cv_summary(console: Console, results: list[dict[str, float]]) -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    metric_cols = ["f1_action", "f1_point", "auc", "action_point_score", "score", "selection_value", "dense_score", "loss", "train_loss"]
    table = Table(title="K-fold CV summary")
    table.add_column("metric")
    table.add_column("mean", justify="right")
    table.add_column("std", justify="right")
    for col in metric_cols:
        if col in df:
            table.add_row(col, f"{df[col].mean():.6f}", f"{df[col].std(ddof=0):.6f}")
    console.print(table)
    summary_path = Path(results[0]["out_dir"]).parent / "cv_summary.csv" if "out_dir" in results[0] else None
    if summary_path is not None:
        df.to_csv(summary_path, index=False)
        console.print(f"[cyan]CV summary CSV:[/] {summary_path}")


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

    if "serverGetPoint" in resolve_features(args.feature_set):
        raise ValueError("serverGetPoint must not be included in FEATURES because it leaks the target.")

    rally_labels = train_df.groupby("rally_uid", sort=False)["serverGetPoint"].first()
    rally_ids = rally_labels.index.to_numpy()
    labels = rally_labels.to_numpy()

    if args.cv_folds <= 1:
        stratify = labels if len(np.unique(labels)) > 1 else None
        tr_ids, va_ids = train_test_split(
            rally_ids,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=stratify,
        )
        train_one_split(args, console, train_df, tr_ids, va_ids)
        return

    if len(rally_ids) < args.cv_folds:
        raise ValueError(f"cv-folds={args.cv_folds} is larger than number of rallies={len(rally_ids)}")
    if len(np.unique(labels)) < 2:
        raise ValueError("Stratified k-fold requires at least two serverGetPoint classes.")

    results: list[dict[str, float]] = []
    for run_label, tr_ids, va_ids in iter_cv_splits(rally_ids, labels, args.cv_folds, args.cv_repeats, args.seed):
        seed_everything(args.seed)
        metrics = train_one_split(args, console, train_df, tr_ids, va_ids, run_label=run_label)
        metrics["run_label"] = run_label
        metrics["out_dir"] = str(Path(args.out_dir) / run_label)
        results.append(metrics)
    print_cv_summary(console, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--train", default="")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--feature-set", choices=("base", "score", "enhanced", "semantic"), default="semantic")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--val-target-from-end", type=int, default=2, help="1 predicts the terminal strike; 2 predicts the pre-terminal strike, avoiding degenerate terminal pointId.")
    parser.add_argument("--cv-folds", type=int, default=1, help="Use stratified k-fold CV when > 1; folds are split by rally_uid.")
    parser.add_argument("--cv-repeats", type=int, default=1, help="Number of repeated CV rounds when --cv-folds > 1.")
    parser.add_argument("--limit-rallies", type=int, default=0, help="Debug/smoke-test limit; 0 uses all rallies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-type", choices=("transformer", "gru", "lstm"), default="transformer")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--bidirectional-rnn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pcgrad-rally-mode", choices=("separate", "together"), default="separate")
    parser.add_argument("--action-loss-weight", type=float, default=0.45)
    parser.add_argument("--point-loss-weight", type=float, default=0.45)
    parser.add_argument("--rally-loss-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-mode", choices=("none", "inverse", "sqrt", "effective"), default="sqrt")
    parser.add_argument("--class-weight-max", type=float, default=3.0)
    parser.add_argument("--class-weight-effective-beta", type=float, default=0.999)
    parser.add_argument("--selection-metric", choices=("score", "action_point_score", "min_f1"), default="action_point_score")
    parser.add_argument("--scheduler", choices=("onecycle", "none"), default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=1000.0)
    parser.add_argument("--rally-pool", choices=("last_mean_mlp", "mean_linear"), default="last_mean_mlp")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Stop after this many epochs without submission-like score improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True, help="Save training curves PNG and history CSV.")
    main(parser.parse_args())
