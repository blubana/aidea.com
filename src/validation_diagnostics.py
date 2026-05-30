"""Validation diagnostics for OOF predictions and leakage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


TARGETS = ["actionId", "pointId", "serverGetPoint"]
TARGET_DIMS = {"actionId": 19, "pointId": 10, "serverGetPoint": 2}
LABEL_COLS = {
    "actionId": "label_actionId",
    "pointId": "label_pointId",
    "serverGetPoint": "label_serverGetPoint",
}
OBJECTIVES = {
    "actionId": "macro_f1",
    "pointId": "macro_f1",
    "serverGetPoint": "roc_auc",
}
DEFAULT_ABLATION_TARGETS = ["pointId", "serverGetPoint"]
FORBIDDEN_FEATURES = {"source_rally_len", "label_actionId", "label_pointId", "label_serverGetPoint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--lstm-dataset", default="data/processed/lstm_dataset.npz")
    parser.add_argument("--run-prefix-ablation", action="store_true")
    parser.add_argument("--ablation-targets", nargs="+", default=DEFAULT_ABLATION_TARGETS, choices=TARGETS)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / probs.shape[1]), where=row_sum > 0)


def align_probs(probs: np.ndarray, target: str) -> np.ndarray:
    expected = TARGET_DIMS[target]
    if probs.shape[1] < expected:
        padded = np.zeros((probs.shape[0], expected), dtype=probs.dtype)
        padded[:, : probs.shape[1]] = probs
        probs = padded
    return normalize_probs(probs.astype(np.float64))


def load_oof_probs(path: Path, target: str) -> np.ndarray | None:
    if not path.exists():
        return None
    probs = np.load(path)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    return align_probs(probs, target)


def load_ensemble_weights() -> dict[str, float]:
    path = Path("reports/ensemble/summary.csv")
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "target" not in df or "weight_lstm" not in df:
        return {}
    return {str(row.target): float(row.weight_lstm) for row in df.itertuples(index=False)}


def weighted_mean(values: np.ndarray, weights: np.ndarray | None) -> float:
    if weights is None:
        return float(np.mean(values))
    denom = float(np.sum(weights))
    return float(np.sum(values * weights) / denom) if denom > 0 else float(np.mean(values))


def multiclass_log_loss(y_true: np.ndarray, probs: np.ndarray, sample_weight: np.ndarray | None) -> float:
    return float(log_loss(y_true, probs, labels=list(range(probs.shape[1])), sample_weight=sample_weight))


def binary_roc_auc(y_true: np.ndarray, probs: np.ndarray, sample_weight: np.ndarray | None) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, probs[:, 1], sample_weight=sample_weight))
    except ValueError:
        return None


def compute_metrics(target: str, y_true: np.ndarray, probs: np.ndarray, sample_weight: np.ndarray | None) -> dict[str, float | None]:
    pred = probs.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred, sample_weight=sample_weight)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0, sample_weight=sample_weight)),
        "log_loss": multiclass_log_loss(y_true, probs, sample_weight),
    }
    if target == "serverGetPoint":
        metrics["roc_auc"] = binary_roc_auc(y_true, probs, sample_weight)
    else:
        metrics["roc_auc"] = None
    return metrics


def build_slice_rows(df: pd.DataFrame, target: str, source: str, probs: np.ndarray) -> list[dict[str, object]]:
    y_true = df[LABEL_COLS[target]].to_numpy(dtype=np.int64)
    weights = df["sample_weight"].to_numpy(dtype=np.float64) if "sample_weight" in df else np.ones(len(df), dtype=np.float64)
    masks = {
        "all_unweighted": np.ones(len(df), dtype=bool),
        "all_weighted": np.ones(len(df), dtype=bool),
        "prefix_len_le_3": df["prefix_len"].to_numpy() <= 3,
        "prefix_len_le_4": df["prefix_len"].to_numpy() <= 4,
    }
    rows = []
    for slice_name, mask in masks.items():
        cur_w = None if slice_name == "all_unweighted" else weights[mask]
        cur = compute_metrics(target, y_true[mask], probs[mask], cur_w)
        rows.append(
            {
                "target": target,
                "source": source,
                "slice": slice_name,
                "rows": int(mask.sum()),
                **cur,
            }
        )
    return rows


def parity_oracle_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    score = score.astype(np.float64)
    pred = (score >= 0.5).astype(np.int64) if np.isin(np.unique(score), [0.0, 1.0]).all() else (score >= np.median(score)).astype(np.int64)
    auc = None
    if len(np.unique(y_true)) > 1 and len(np.unique(score)) > 1:
        try:
            auc = float(roc_auc_score(y_true, score))
        except ValueError:
            auc = None
    return {"accuracy": float(accuracy_score(y_true, pred)), "roc_auc": auc}


def binary_score_metrics(y_true: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    auc = None
    if len(np.unique(y_true)) > 1 and len(np.unique(score)) > 1:
        auc = float(roc_auc_score(y_true, score))
    pred = (score >= 0.5).astype(np.int64)
    return {"accuracy": float(accuracy_score(y_true, pred)), "roc_auc": auc}


def run_leakage_checks(df: pd.DataFrame, lstm_dataset_path: Path) -> dict[str, object]:
    y_server = df["label_serverGetPoint"].to_numpy(dtype=np.int64)
    source_rally_len = df["source_rally_len"].to_numpy(dtype=np.float64)
    prefix_len = df["prefix_len"].to_numpy(dtype=np.float64)
    tabular_checks = {}
    for target in TARGETS:
        bundle = joblib.load(Path("models/tabular_baseline") / f"{target}_extratrees.joblib")
        features = set(bundle.get("feature_columns", []))
        tabular_checks[target] = {
            "forbidden_present": sorted(features.intersection(FORBIDDEN_FEATURES)),
            "passed": not bool(features.intersection(FORBIDDEN_FEATURES)),
        }

    npz = np.load(lstm_dataset_path, allow_pickle=True)
    manual_names = {str(x) for x in npz["manual_feature_names"].tolist()}
    lstm_forbidden = sorted(manual_names.intersection(FORBIDDEN_FEATURES))
    return {
        "server_oracle_source_rally_len_even_score": binary_score_metrics(y_server, 1.0 - (source_rally_len % 2).astype(np.float64)),
        "server_oracle_prefix_len_even_score": binary_score_metrics(y_server, 1.0 - (prefix_len % 2).astype(np.float64)),
        "server_oracle_prefix_len_raw": parity_oracle_metrics(y_server, prefix_len),
        "tabular_feature_column_checks": tabular_checks,
        "lstm_manual_feature_check": {
            "forbidden_present": lstm_forbidden,
            "passed": not bool(lstm_forbidden),
        },
    }


def feature_frame(df: pd.DataFrame, target: str, drop_prefix_len: bool) -> pd.DataFrame:
    drop = {
        "sample_id",
        "rally_uid",
        "source_rally_len",
        "label_actionId",
        "label_pointId",
        "label_serverGetPoint",
        "target_strikeNumber",
    }
    if drop_prefix_len:
        drop.update({"prefix_len", "log_prefix_len", "is_short_prefix", "is_medium_prefix", "is_long_prefix"})
    cols = [c for c in df.columns if c not in drop]
    return pd.get_dummies(df[cols], dummy_na=True)


def ablation_metric(target: str, y_true: np.ndarray, probs: np.ndarray) -> float:
    if target == "serverGetPoint":
        return float(roc_auc_score(y_true, probs[:, 1])) if len(np.unique(y_true)) > 1 else float("nan")
    pred = probs.argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def run_prefix_ablation(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    groups = df["match"].to_numpy()
    rows = []
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for target in args.ablation_targets:
        y = df[LABEL_COLS[target]].to_numpy(dtype=np.int64)
        for drop_prefix_len in [False, True]:
            x = feature_frame(df, target, drop_prefix_len)
            oof = None
            for train_idx, valid_idx in splitter.split(x, y, groups):
                model = ExtraTreesClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=18,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=args.random_state,
                )
                model.fit(x.iloc[train_idx], y[train_idx])
                proba = model.predict_proba(x.iloc[valid_idx])
                if oof is None:
                    n_classes = max(int(np.max(y)) + 1, proba.shape[1])
                    oof = np.zeros((len(y), n_classes), dtype=np.float64)
                model_classes = model.classes_.astype(int)
                aligned = np.zeros((len(valid_idx), oof.shape[1]), dtype=np.float64)
                aligned[:, model_classes] = proba
                oof[valid_idx] = aligned
            rows.append(
                {
                    "target": target,
                    "variant": "drop_prefix_len" if drop_prefix_len else "with_prefix_len",
                    "objective": OBJECTIVES[target],
                    "score": ablation_metric(target, y, oof),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    df = pd.read_csv(args.train_features)
    ensemble_weights = load_ensemble_weights()
    all_rows = []
    summary: dict[str, dict[str, object]] = {}

    for target in TARGETS:
        target_rows = {}
        tab = load_oof_probs(Path("reports/tabular_baseline") / f"{target}_oof_proba.npy", target)
        lstm = load_oof_probs(Path("reports/lstm") / f"{target}_oof_proba.npy", target)
        sources = {"tabular": tab, "lstm": lstm}
        if tab is not None and lstm is not None and target in ensemble_weights:
            w = ensemble_weights[target]
            sources["ensemble"] = normalize_probs(w * lstm + (1.0 - w) * tab)
        for source, probs in sources.items():
            if probs is None:
                continue
            rows = build_slice_rows(df, target, source, probs)
            all_rows.extend(rows)
            target_rows[source] = rows
        summary[target] = target_rows

    out_dir = Path("reports/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_dir / "oof_slices.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    leakage = run_leakage_checks(df, Path(args.lstm_dataset))
    (out_dir / "leakage_checks.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")
    if args.run_prefix_ablation:
        run_prefix_ablation(df, args).to_csv(out_dir / "prefix_ablation.csv", index=False)


if __name__ == "__main__":
    main()
