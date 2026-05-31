"""Run bounded feature-group ablations for tabular ExtraTrees baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from train_tabular_baseline import TARGETS, aligned_proba, make_model, split_columns


SAFE_DROP = {
    "sample_id",
    "rally_uid",
    "source_rally_len",
    "target_strikeNumber",
    "sample_weight",
    "label_actionId",
    "label_pointId",
    "label_serverGetPoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--report-dir", default="reports/feature_ablation")
    parser.add_argument("--targets", nargs="+", default=["actionId", "pointId", "serverGetPoint"], choices=list(TARGETS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--drop-raw-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-player-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--drop-transition-features", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def ablation_drop_columns(df: pd.DataFrame, args: argparse.Namespace, target_label: str) -> list[str]:
    drop = set(SAFE_DROP)
    drop.add(target_label)
    if args.drop_raw_ids:
        drop.update({"match", "rally_id", "numberGame", "sex"})
    if args.drop_player_ids:
        drop.update({"gamePlayerId", "gamePlayerOtherId", "last_hitter_id", "last_opponent_id"})
        drop.update({c for c in df.columns if "gamePlayerId" in c or "gamePlayerOtherId" in c})
    if args.drop_transition_features:
        drop.update({c for c in df.columns if "transition" in c})
    return [c for c in df.columns if c not in drop]


def metric_bundle(target_name: str, y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0, sample_weight=weights)),
        "log_loss": float(log_loss(y_true, probs, labels=TARGETS[target_name]["classes"])),
    }
    if target_name == "serverGetPoint" and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    return metrics


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    df = pd.read_csv(args.train_features)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.random_state).reset_index(drop=True)
    groups = df["match"].to_numpy()
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    n_splits = min(args.folds, len(pd.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    rows = []

    variant = {
        "drop_raw_ids": args.drop_raw_ids,
        "drop_player_ids": args.drop_player_ids,
        "drop_transition_features": args.drop_transition_features,
    }
    for target_name in args.targets:
        label_col = TARGETS[target_name]["label"]
        classes = TARGETS[target_name]["classes"]
        cols = ablation_drop_columns(df, args, label_col)
        x = df[cols].copy()
        y = df[label_col].to_numpy(dtype=int)
        numeric_cols, categorical_cols = split_columns(x)
        oof = np.zeros((len(df), len(classes)), dtype=float)
        for train_idx, valid_idx in splitter.split(x, y, groups):
            model = make_model(numeric_cols, categorical_cols, argparse.Namespace(n_estimators=400, max_depth=18, min_samples_leaf=2, random_state=args.random_state))
            model.fit(x.iloc[train_idx], y[train_idx], model__sample_weight=weights[train_idx])
            oof[valid_idx] = aligned_proba(model, x.iloc[valid_idx], classes)
        rows.append({"target": target_name, "features": len(cols), **variant, **metric_bundle(target_name, y, oof, weights)})

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir / "summary.csv", index=False)
    (report_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
