"""Tune simple class-probability multipliers on OOF blends for Macro F1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from train_server_stacking import normalize_probs


LABEL_COLS = {"actionId": "label_actionId", "pointId": "label_pointId"}
TARGET_DIMS = {"actionId": 19, "pointId": 10}
FINAL_WEIGHTS = {
    "action": {"tabular": 0.70, "lstm": 0.30, "catboost": 0.15},
    "point_base": {"tabular": 0.35, "lstm": 0.65},
    "point_final": {"base": 0.60, "phase": 0.40, "catboost": 0.30},
}


def load_probs(path: Path, dim: int, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing probability artifact: {path}")
    probs = normalize_probs(np.load(path), dim)
    if len(probs) != expected_rows:
        raise ValueError(f"Row mismatch for {path}: expected {expected_rows}, got {len(probs)}")
    return probs


def metric_bundle(df: pd.DataFrame, target: str, probs: np.ndarray) -> dict[str, float]:
    y = df[LABEL_COLS[target]].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    mask3 = df["prefix_len"].to_numpy() <= 3
    mask4 = df["prefix_len"].to_numpy() <= 4
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, sample_weight=weights)),
        "prefix_le_3_macro_f1": float(f1_score(y[mask3], pred[mask3], average="macro", zero_division=0)) if mask3.any() else float("nan"),
        "prefix_le_4_macro_f1": float(f1_score(y[mask4], pred[mask4], average="macro", zero_division=0)) if mask4.any() else float("nan"),
    }


def alpha_to_multipliers(y: np.ndarray, dim: int, alpha: float) -> np.ndarray:
    counts = np.bincount(y, minlength=dim).astype(float)
    freq = np.maximum(counts / max(counts.sum(), 1.0), 1e-12)
    multipliers = np.power(freq, -alpha)
    return multipliers / np.mean(multipliers)


def apply_multipliers(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    return normalize_probs(probs * multipliers.reshape(1, -1), probs.shape[1])


def greedy_refine(df: pd.DataFrame, target: str, probs: np.ndarray, multipliers: np.ndarray) -> tuple[np.ndarray, dict[str, float], list[dict[str, float]]]:
    best_mult = multipliers.copy()
    best_probs = apply_multipliers(probs, best_mult)
    best_metrics = metric_bundle(df, target, best_probs)
    history: list[dict[str, float]] = []
    for cls in range(len(best_mult)):
        current_best = best_metrics
        current_mult = best_mult.copy()
        for factor in [0.90, 0.95, 1.05, 1.10]:
            trial = best_mult.copy()
            trial[cls] *= factor
            trial /= np.mean(trial)
            trial_probs = apply_multipliers(probs, trial)
            trial_metrics = metric_bundle(df, target, trial_probs)
            if trial_metrics["macro_f1"] > current_best["macro_f1"]:
                current_best = trial_metrics
                current_mult = trial
        if current_best["macro_f1"] > best_metrics["macro_f1"]:
            best_mult = current_mult
            best_metrics = current_best
            history.append({"class_index": cls, "macro_f1": current_best["macro_f1"]})
    return best_mult, best_metrics, history


def main() -> None:
    df = pd.read_csv("data/processed/prefix_train_features.csv")
    out_dir = Path("reports/class_multipliers")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}

    action_tab = load_probs(Path("reports/tabular_baseline/actionId_oof_proba.npy"), 19, len(df))
    action_lstm = load_probs(Path("reports/lstm/actionId_oof_proba.npy"), 19, len(df))
    action_cat = load_probs(Path("reports/catboost/actionId_oof_proba.npy"), 19, len(df))
    action_base = normalize_probs(FINAL_WEIGHTS["action"]["tabular"] * action_tab + FINAL_WEIGHTS["action"]["lstm"] * action_lstm, 19)
    action_probs = normalize_probs((1.0 - FINAL_WEIGHTS["action"]["catboost"]) * action_base + FINAL_WEIGHTS["action"]["catboost"] * action_cat, 19)
    action_y = df[LABEL_COLS["actionId"]].to_numpy(dtype=int)

    best_action = None
    best_action_mult = np.ones(19, dtype=float)
    for alpha in np.arange(0.0, 2.0001, 0.05):
        multipliers = alpha_to_multipliers(action_y, 19, float(alpha))
        probs = apply_multipliers(action_probs, multipliers)
        metrics = metric_bundle(df, "actionId", probs)
        row = {"target": "actionId", "stage": "alpha_grid", "alpha": round(float(alpha), 2), **metrics}
        rows.append(row)
        if best_action is None or metrics["macro_f1"] > best_action["macro_f1"]:
            best_action = row
            best_action_mult = multipliers
    best_action_mult, best_action_metrics, action_history = greedy_refine(df, "actionId", action_probs, best_action_mult)
    np.save(out_dir / "actionId_multipliers.npy", best_action_mult)
    summary["actionId"] = {
        "best_alpha": best_action["alpha"],
        "best_alpha_metrics": best_action,
        "greedy_refined_metrics": best_action_metrics,
        "greedy_refinements": action_history,
    }

    point_tab = load_probs(Path("reports/tabular_baseline/pointId_oof_proba.npy"), 10, len(df))
    point_lstm = load_probs(Path("reports/lstm/pointId_oof_proba.npy"), 10, len(df))
    point_phase = load_probs(Path("reports/point_phase/pointId_oof_proba.npy"), 10, len(df))
    point_cat = load_probs(Path("reports/catboost/pointId_oof_proba.npy"), 10, len(df))
    point_base = normalize_probs(FINAL_WEIGHTS["point_base"]["tabular"] * point_tab + FINAL_WEIGHTS["point_base"]["lstm"] * point_lstm, 10)
    point_no_cat = normalize_probs(FINAL_WEIGHTS["point_final"]["base"] * point_base + FINAL_WEIGHTS["point_final"]["phase"] * point_phase, 10)
    point_probs = normalize_probs((1.0 - FINAL_WEIGHTS["point_final"]["catboost"]) * point_no_cat + FINAL_WEIGHTS["point_final"]["catboost"] * point_cat, 10)
    point_y = df[LABEL_COLS["pointId"]].to_numpy(dtype=int)

    best_point = None
    best_point_mult = np.ones(10, dtype=float)
    for alpha in np.arange(0.0, 2.0001, 0.05):
        multipliers = alpha_to_multipliers(point_y, 10, float(alpha))
        probs = apply_multipliers(point_probs, multipliers)
        metrics = metric_bundle(df, "pointId", probs)
        row = {"target": "pointId", "stage": "alpha_grid", "alpha": round(float(alpha), 2), **metrics}
        rows.append(row)
        if best_point is None or metrics["macro_f1"] > best_point["macro_f1"]:
            best_point = row
            best_point_mult = multipliers
    best_point_mult, best_point_metrics, point_history = greedy_refine(df, "pointId", point_probs, best_point_mult)
    np.save(out_dir / "pointId_multipliers.npy", best_point_mult)
    summary["pointId"] = {
        "best_alpha": best_point["alpha"],
        "best_alpha_metrics": best_point,
        "greedy_refined_metrics": best_point_metrics,
        "greedy_refinements": point_history,
    }

    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
