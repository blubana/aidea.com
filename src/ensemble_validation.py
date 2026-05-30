"""Search simple tabular/LSTM blend weights on OOF predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


TARGETS = ["actionId", "pointId", "serverGetPoint"]
TARGET_DIMS = {"actionId": 19, "pointId": 10, "serverGetPoint": 2}
LABEL_COLS = {
    "actionId": "label_actionId",
    "pointId": "label_pointId",
    "serverGetPoint": "label_serverGetPoint",
}


def load_y_true(target: str) -> np.ndarray:
    pred_path = Path(f"reports/lstm/{target}_oof_predictions.csv")
    if pred_path.exists():
        return pd.read_csv(pred_path)["y_true"].to_numpy(dtype=np.int64)
    df = pd.read_csv("data/processed/prefix_train_features.csv", usecols=[LABEL_COLS[target]])
    return df[LABEL_COLS[target]].to_numpy(dtype=np.int64)


def load_probs(path: Path, target: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF probabilities: {path}")
    probs = np.load(path)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    expected_dim = TARGET_DIMS[target]
    if probs.shape[1] < expected_dim:
        padded = np.zeros((probs.shape[0], expected_dim), dtype=probs.dtype)
        padded[:, : probs.shape[1]] = probs
        probs = padded
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / probs.shape[1]), where=row_sum > 0)
    return probs.astype(np.float64)


def score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    row_sum = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / probs.shape[1]), where=row_sum > 0)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probs, labels=list(range(probs.shape[1]))),
        "roc_auc": roc_auc_score(y_true, probs[:, 1]) if probs.shape[1] == 2 and len(np.unique(y_true)) > 1 else np.nan,
    }


def objective_metric(target: str) -> str:
    return "roc_auc" if target == "serverGetPoint" else "macro_f1"


def main() -> None:
    rows = []
    for target in TARGETS:
        tab_path = Path(f"reports/tabular_baseline/{target}_oof_proba.npy")
        lstm_path = Path(f"reports/lstm/{target}_oof_proba.npy")
        if not lstm_path.exists():
            raise FileNotFoundError(f"Missing LSTM OOF for {target}: {lstm_path}. Run src/train_lstm.py first.")
        y_true = load_y_true(target)
        tab = load_probs(tab_path, target)
        lstm = load_probs(lstm_path, target)
        if tab.shape != lstm.shape:
            raise ValueError(f"Shape mismatch for {target}: tabular {tab.shape} vs lstm {lstm.shape}")

        metric = objective_metric(target)
        best = None
        for w in np.arange(0.0, 1.0001, 0.05):
            probs = w * lstm + (1.0 - w) * tab
            metrics = score(y_true, probs)
            row = {"target": target, "objective": metric, "weight_lstm": round(float(w), 2), **metrics}
            if best is None or metrics[metric] > best[metric]:
                best = row
        rows.append(best)

    out_dir = Path("reports/ensemble")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
