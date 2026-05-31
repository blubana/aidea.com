"""Blend CatBoost OOF predictions with current best OOF blends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from train_server_stacking import normalize_probs


LABEL_COLS = {"actionId": "label_actionId", "pointId": "label_pointId", "serverGetPoint": "label_serverGetPoint"}
TARGET_DIMS = {"actionId": 19, "pointId": 10, "serverGetPoint": 2}


def load_probs(path: Path, dim: int) -> np.ndarray | None:
    if not path.exists():
        return None
    probs = np.load(path)
    return normalize_probs(probs, dim)


def score_action(df: pd.DataFrame, probs: np.ndarray) -> dict[str, float]:
    y = df[LABEL_COLS["actionId"]].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probs, labels=list(range(19)))),
    }


def score_point(df: pd.DataFrame, probs: np.ndarray) -> dict[str, float]:
    y = df[LABEL_COLS["pointId"]].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    w = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, sample_weight=w)),
        "log_loss": float(log_loss(y, probs, labels=list(range(10)))),
    }


def score_server(df: pd.DataFrame, probs: np.ndarray) -> dict[str, float]:
    y = df[LABEL_COLS["serverGetPoint"]].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    w = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs[:, 1])) if len(np.unique(y)) > 1 else float("nan"),
        "weighted_roc_auc": float(roc_auc_score(y, probs[:, 1], sample_weight=w)) if len(np.unique(y)) > 1 else float("nan"),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
    }


def main() -> None:
    df = pd.read_csv("data/processed/prefix_train_features.csv")
    out_dir = Path("reports/catboost_blend")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []

    action_base = load_probs(Path("reports/tabular_baseline/actionId_oof_proba.npy"), 19)
    action_lstm = load_probs(Path("reports/lstm/actionId_oof_proba.npy"), 19)
    action_cat = load_probs(Path("reports/catboost/actionId_oof_proba.npy"), 19)
    if action_base is not None and action_lstm is not None and action_cat is not None:
        base = normalize_probs(0.70 * action_base + 0.30 * action_lstm, 19)
        best = None
        for w in np.arange(0.0, 1.0001, 0.05):
            probs = normalize_probs((1.0 - w) * base + w * action_cat, 19)
            metrics = score_action(df, probs)
            row = {"target": "actionId", "weight_catboost": round(float(w), 2), **metrics}
            rows.append(row)
            if best is None or row["macro_f1"] > best["macro_f1"]:
                best = row
        summary.append(best)

    point_tab = load_probs(Path("reports/tabular_baseline/pointId_oof_proba.npy"), 10)
    point_lstm = load_probs(Path("reports/lstm/pointId_oof_proba.npy"), 10)
    point_phase = load_probs(Path("reports/point_phase/pointId_oof_proba.npy"), 10)
    point_cat = load_probs(Path("reports/catboost/pointId_oof_proba.npy"), 10)
    if point_tab is not None and point_lstm is not None and point_phase is not None and point_cat is not None:
        point_base = normalize_probs(0.35 * point_tab + 0.65 * point_lstm, 10)
        base = normalize_probs(0.60 * point_base + 0.40 * point_phase, 10)
        best_weighted = None
        best_macro = None
        for w in np.arange(0.0, 1.0001, 0.05):
            probs = normalize_probs((1.0 - w) * base + w * point_cat, 10)
            metrics = score_point(df, probs)
            row = {"target": "pointId", "weight_catboost": round(float(w), 2), **metrics}
            rows.append(row)
            if best_weighted is None or row["weighted_macro_f1"] > best_weighted["weighted_macro_f1"]:
                best_weighted = row
            if best_macro is None or row["macro_f1"] > best_macro["macro_f1"]:
                best_macro = row
        summary.extend([{"target": "pointId_best_weighted", **best_weighted}, {"target": "pointId_best_macro", **best_macro}])

    server_tab = load_probs(Path("reports/tabular_baseline/serverGetPoint_oof_proba.npy"), 2)
    server_lstm = load_probs(Path("reports/lstm/serverGetPoint_oof_proba.npy"), 2)
    server_stack = load_probs(Path("reports/server_stacking/serverGetPoint_oof_proba.npy"), 2)
    server_cat = load_probs(Path("reports/catboost/serverGetPoint_oof_proba.npy"), 2)
    if server_tab is not None and server_lstm is not None and server_stack is not None and server_cat is not None:
        server_base = normalize_probs(0.80 * server_tab + 0.20 * server_lstm, 2)
        base = normalize_probs(0.35 * server_base + 0.65 * server_stack, 2)
        best_weighted = None
        best_auc = None
        for w in np.arange(0.0, 1.0001, 0.05):
            probs = normalize_probs((1.0 - w) * base + w * server_cat, 2)
            metrics = score_server(df, probs)
            row = {"target": "serverGetPoint", "weight_catboost": round(float(w), 2), **metrics}
            rows.append(row)
            if best_weighted is None or row["weighted_roc_auc"] > best_weighted["weighted_roc_auc"]:
                best_weighted = row
            if best_auc is None or row["roc_auc"] > best_auc["roc_auc"]:
                best_auc = row
        summary.extend([{"target": "serverGetPoint_best_weighted", **best_weighted}, {"target": "serverGetPoint_best_auc", **best_auc}])

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
