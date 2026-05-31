"""Train phase-specific actionId tabular models with grouped OOF validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from train_point_phase_models import DEFAULT_BUCKETS, assign_bucket, can_train_bucket, fit_model, make_feature_frame, normalize_proba


ACTION_CLASSES = list(range(19))
LABEL_COL = "label_actionId"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--test-features", default="data/processed/prefix_test_features.csv")
    parser.add_argument("--output-dir", default="models/action_phase")
    parser.add_argument("--report-dir", default="reports/action_phase")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--min-bucket-train-rows", type=int, default=100)
    return parser.parse_args()


def aligned_proba(model, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        if int(cls) in ACTION_CLASSES:
            out[:, ACTION_CLASSES.index(int(cls))] = raw[:, src_idx]
    return normalize_proba(out)


def metric_bundle(df: pd.DataFrame, probs: np.ndarray) -> dict[str, float]:
    probs = normalize_proba(probs)
    y = df[LABEL_COL].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    mask3 = df["prefix_len"].to_numpy() <= 3
    mask4 = df["prefix_len"].to_numpy() <= 4
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, sample_weight=weights)),
        "log_loss": float(log_loss(y, probs, labels=ACTION_CLASSES)),
        "prefix_le_3_macro_f1": float(f1_score(y[mask3], pred[mask3], average="macro", zero_division=0)) if mask3.any() else float("nan"),
        "prefix_le_4_macro_f1": float(f1_score(y[mask4], pred[mask4], average="macro", zero_division=0)) if mask4.any() else float("nan"),
    }


def load_optional_probs(path: Path, dim: int, expected_rows: int) -> np.ndarray | None:
    if not path.exists():
        return None
    probs = normalize_proba(np.load(path))
    if len(probs) != expected_rows:
        raise ValueError(f"Row mismatch for {path}: expected {expected_rows}, got {len(probs)}")
    if probs.shape[1] < dim:
        padded = np.zeros((len(probs), dim), dtype=float)
        padded[:, : probs.shape[1]] = probs
        probs = normalize_proba(padded)
    return probs


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    train_df = pd.read_csv(args.train_features).copy()
    if args.sample and args.sample < len(train_df):
        train_df = train_df.sample(args.sample, random_state=args.random_state).reset_index(drop=True)

    train_df = train_df.assign(phase_bucket=assign_bucket(train_df["next_strikeNumber"]))
    x_all = make_feature_frame(train_df)
    y_all = train_df[LABEL_COL].astype(int).to_numpy()
    weights_all = train_df["sample_weight"].astype(float).to_numpy() if "sample_weight" in train_df else np.ones(len(train_df), dtype=float)
    groups = train_df["match"].to_numpy()

    n_splits = min(args.folds, len(pd.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros((len(train_df), len(ACTION_CLASSES)), dtype=float)
    per_bucket_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_all, y_all, groups), start=1):
        x_train = x_all.iloc[train_idx]
        y_train = y_all[train_idx]
        w_train = weights_all[train_idx]
        train_bucket = train_df.iloc[train_idx]["phase_bucket"].to_numpy()
        valid_bucket = train_df.iloc[valid_idx]["phase_bucket"].to_numpy()
        global_model = fit_model(x_train, y_train, w_train, args)
        fold_proba = aligned_proba(global_model, x_all.iloc[valid_idx])

        for bucket_name in list(DEFAULT_BUCKETS) + ["other"]:
            train_mask = train_bucket == bucket_name
            valid_mask = valid_bucket == bucket_name
            if not valid_mask.any():
                continue
            used_global = True
            if can_train_bucket(y_train[train_mask], args.min_bucket_train_rows):
                bucket_model = fit_model(x_train.iloc[train_mask], y_train[train_mask], w_train[train_mask], args)
                fold_proba[valid_mask] = aligned_proba(bucket_model, x_all.iloc[valid_idx].iloc[valid_mask])
                used_global = False
            bucket_true = y_all[valid_idx][valid_mask]
            bucket_pred = fold_proba[valid_mask].argmax(axis=1)
            per_bucket_rows.append(
                {
                    "fold": fold,
                    "bucket": bucket_name,
                    "train_rows": int(train_mask.sum()),
                    "valid_rows": int(valid_mask.sum()),
                    "used_global_fallback": used_global,
                    "accuracy": float(accuracy_score(bucket_true, bucket_pred)),
                    "macro_f1": float(f1_score(bucket_true, bucket_pred, average="macro", zero_division=0)),
                }
            )
        oof_proba[valid_idx] = fold_proba

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    final_global = fit_model(x_all, y_all, weights_all, args)
    joblib.dump({"model": final_global, "feature_columns": list(x_all.columns), "classes": ACTION_CLASSES, "bucket": "global"}, output_dir / "global.joblib")
    for bucket_name in list(DEFAULT_BUCKETS) + ["other"]:
        bucket_mask = train_df["phase_bucket"].to_numpy() == bucket_name
        if can_train_bucket(y_all[bucket_mask], args.min_bucket_train_rows):
            model = fit_model(x_all.iloc[bucket_mask], y_all[bucket_mask], weights_all[bucket_mask], args)
            bundle = {"model": model, "feature_columns": list(x_all.columns), "classes": ACTION_CLASSES, "bucket": bucket_name}
        else:
            bundle = {"model": final_global, "feature_columns": list(x_all.columns), "classes": ACTION_CLASSES, "bucket": bucket_name, "fallback": "global"}
        joblib.dump(bundle, output_dir / f"{bucket_name}.joblib")

    oof_pred = oof_proba.argmax(axis=1)
    np.save(report_dir / "actionId_oof_proba.npy", oof_proba)
    pd.DataFrame({"sample_id": train_df["sample_id"], "y_true": y_all, "oof_pred": oof_pred, "phase_bucket": train_df["phase_bucket"]}).to_csv(report_dir / "actionId_oof_predictions.csv", index=False)
    pd.DataFrame(per_bucket_rows).to_csv(report_dir / "per_bucket_metrics.csv", index=False)

    summary = {"target": "actionId", "folds": n_splits, "oof": metric_bundle(train_df, oof_proba)}
    n_rows = len(train_df)
    tab = load_optional_probs(Path("reports/tabular_baseline/actionId_oof_proba.npy"), 19, n_rows)
    lstm = load_optional_probs(Path("reports/lstm/actionId_oof_proba.npy"), 19, n_rows)
    cat = load_optional_probs(Path("reports/catboost/actionId_oof_proba.npy"), 19, n_rows)
    if tab is not None and lstm is not None and cat is not None:
        base = normalize_proba(0.85 * normalize_proba(0.70 * tab + 0.30 * lstm) + 0.15 * cat)
        blend_rows = []
        y = y_all
        mask3 = train_df["prefix_len"].to_numpy() <= 3
        mask4 = train_df["prefix_len"].to_numpy() <= 4
        for w in np.arange(0.0, 1.0001, 0.05):
            blend = normalize_proba(w * oof_proba + (1.0 - w) * base)
            pred = blend.argmax(axis=1)
            blend_rows.append(
                {
                    "weight_action_phase": round(float(w), 2),
                    "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                    "weighted_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0, sample_weight=weights_all)),
                    "prefix_le_3_macro_f1": float(f1_score(y[mask3], pred[mask3], average="macro", zero_division=0)),
                    "prefix_le_4_macro_f1": float(f1_score(y[mask4], pred[mask4], average="macro", zero_division=0)),
                    "log_loss": float(log_loss(y, blend, labels=ACTION_CLASSES)),
                }
            )
        blend_df = pd.DataFrame(blend_rows)
        blend_df.to_csv(report_dir / "actionId_blend_summary.csv", index=False)
        summary["blend_with_existing"] = {
            "base_description": "0.85*(0.70*tabular+0.30*lstm)+0.15*catboost",
            "best_weighted_macro_f1": blend_df.sort_values(["weighted_macro_f1", "macro_f1"], ascending=False).iloc[0].to_dict(),
            "best_macro_f1": blend_df.sort_values(["macro_f1", "weighted_macro_f1"], ascending=False).iloc[0].to_dict(),
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
