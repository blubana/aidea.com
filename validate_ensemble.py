from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from data.dataset import sort_rallies
from inference import load_member, member_predict_proba, resolve_checkpoints, validate_class_alignment


def banner(console: Console) -> None:
    console.print("[bold blue]+--------------------------------------------+[/]")
    console.print("[bold blue]|  AI CUP Mixed Ensemble Validator          |[/]")
    console.print("[bold blue]+--------------------------------------------+[/]")


def target_prefix(group: pd.DataFrame, target_from_end: int) -> tuple[pd.DataFrame, pd.Series] | None:
    target_idx = len(group) - max(1, target_from_end)
    if target_idx < 1:
        return None
    return group.iloc[:target_idx].copy(), group.iloc[target_idx]


def evaluate_ensemble(args: argparse.Namespace) -> dict[str, float]:
    console = Console()
    banner(console)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    paths = resolve_checkpoints(args)
    members = [load_member(path, device) for path in paths]
    action_classes, point_classes = validate_class_alignment(members)
    action_to_idx = {v: i for i, v in enumerate(action_classes)}
    point_to_idx = {v: i for i, v in enumerate(point_classes)}

    data_dir = Path(args.data_dir)
    train_csv = Path(args.train) if args.train else data_dir / "train.csv"
    df = sort_rallies(pd.read_csv(train_csv))
    if args.limit_rallies > 0:
        keep = df["rally_uid"].drop_duplicates().head(args.limit_rallies)
        df = df[df["rally_uid"].isin(keep)].copy()

    rally_labels = df.groupby("rally_uid", sort=False)["serverGetPoint"].first()
    rally_ids = rally_labels.index.to_numpy()
    labels = rally_labels.to_numpy()
    _, val_ids = train_test_split(
        rally_ids,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    val_df = df[df["rally_uid"].isin(val_ids)].copy()

    y_a: list[int] = []
    p_a: list[int] = []
    y_p: list[int] = []
    p_p: list[int] = []
    y_r: list[float] = []
    p_r: list[float] = []

    groups = val_df.groupby("rally_uid", sort=False)
    for _, group in tqdm(groups, total=groups.ngroups, desc="validate rallies", ascii=True):
        pair = target_prefix(group, args.val_target_from_end)
        if pair is None:
            continue
        prefix, target = pair
        action_prob_sum: np.ndarray | None = None
        point_prob_sum: np.ndarray | None = None
        rally_prob_sum = 0.0
        for member in members:
            pa, pp, pr = member_predict_proba(prefix, member, device, action_classes, point_classes)
            action_prob_sum = pa if action_prob_sum is None else action_prob_sum + pa
            point_prob_sum = pp if point_prob_sum is None else point_prob_sum + pp
            rally_prob_sum += pr
        if action_prob_sum is None or point_prob_sum is None:
            continue
        true_action = int(target["actionId"])
        true_point = int(target["pointId"])
        if true_action not in action_to_idx or true_point not in point_to_idx:
            continue
        y_a.append(action_to_idx[true_action])
        p_a.append(int(action_prob_sum.argmax()))
        y_p.append(point_to_idx[true_point])
        p_p.append(int(point_prob_sum.argmax()))
        y_r.append(float(target["serverGetPoint"]))
        p_r.append(rally_prob_sum / len(members))

    f1_action = f1_score(y_a, p_a, average="macro", zero_division=0) if y_a else 0.0
    f1_point = f1_score(y_p, p_p, average="macro", zero_division=0) if y_p else 0.0
    auc = roc_auc_score(y_r, p_r) if len(set(y_r)) > 1 else 0.5
    score = 0.4 * f1_action + 0.4 * f1_point + 0.2 * auc
    point_pred_unique = len(set(p_p))
    point_true_unique = len(set(y_p))
    top_point = 0.0
    if p_p:
        _, counts = np.unique(np.asarray(p_p), return_counts=True)
        top_point = float(counts.max() / len(p_p))

    table = Table(title="Mixed ensemble validation")
    table.add_column("item")
    table.add_column("value", justify="right")
    for k, v in {
        "members": len(members),
        "val rallies": len(y_a),
        "target from end": args.val_target_from_end,
        "F1_action": f"{f1_action:.6f}",
        "F1_point": f"{f1_point:.6f}",
        "AUC": f"{auc:.6f}",
        "score": f"{score:.6f}",
        "point_pred": f"{point_pred_unique}/{point_true_unique}",
        "top_point": f"{top_point:.4f}",
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)

    return {
        "f1_action": float(f1_action),
        "f1_point": float(f1_point),
        "auc": float(auc),
        "score": float(score),
        "point_pred_unique": float(point_pred_unique),
        "point_true_unique": float(point_true_unique),
        "point_top_pred_frac": float(top_point),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="*", default=[], help="One or more .pt/.pkl ensemble members.")
    parser.add_argument("--checkpoint-dir", default="", help="Directory containing checkpoint files.")
    parser.add_argument("--checkpoint-glob", default="best_transformer_*.pt")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--train", default="")
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--val-target-from-end", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-rallies", type=int, default=0)
    evaluate_ensemble(parser.parse_args())
