"""Train a stacked pointId model from safe base features and cross-target OOF probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold

from train_point_phase_models import POINT_CLASSES
from train_server_stacking import (
    FORBIDDEN_FEATURES,
    add_probability_group,
    make_base_feature_frame,
    make_model,
    normalize_probs,
    require_probs,
    split_columns,
)


LABEL_COL = "label_pointId"


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--output-dir", default="models/point_stacking")
    parser.add_argument("--report-dir", default="reports/point_stacking")
    parser.add_argument("--group-col", default="match", choices=["match", "rally_uid"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default="extratrees", choices=["extratrees", "logistic"])
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def aligned_multiclass_proba(model, x: pd.DataFrame, classes: list[int]) -> np.ndarray:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(classes)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        if int(cls) in classes:
            out[:, classes.index(int(cls))] = raw[:, src_idx]
    return normalize_probs(out, len(classes))


def metric_bundle(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    mask3 = df["prefix_len"].to_numpy() <= 3 if "prefix_len" in df else np.zeros(len(df), dtype=bool)
    mask4 = df["prefix_len"].to_numpy() <= 4 if "prefix_len" in df else np.zeros(len(df), dtype=bool)
    return {
        "oof_accuracy": float(accuracy_score(y_true, pred)),
        "oof_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "oof_weighted_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0, sample_weight=weights)),
        "oof_log_loss": float(log_loss(y_true, probs, labels=POINT_CLASSES)),
        "prefix_le_3_macro_f1": float(f1_score(y_true[mask3], pred[mask3], average="macro", zero_division=0)) if mask3.any() else float("nan"),
        "prefix_le_4_macro_f1": float(f1_score(y_true[mask4], pred[mask4], average="macro", zero_division=0)) if mask4.any() else float("nan"),
    }


def build_stacking_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    features = make_base_feature_frame(df)
    n_rows = len(df)
    added_groups: list[str] = []

    def maybe_add(name: str, path: str, dim: int) -> np.ndarray | None:
        p = Path(path)
        if not p.exists():
            return None
        probs = require_probs(p, dim, n_rows)
        nonlocal features
        features = add_probability_group(features, name, probs)
        added_groups.append(name)
        return probs

    action_tab = maybe_add("action_tabular", "reports/tabular_baseline/actionId_oof_proba.npy", 19)
    action_lstm = maybe_add("action_lstm", "reports/lstm/actionId_oof_proba.npy", 19)
    action_cat = maybe_add("action_catboost", "reports/catboost/actionId_oof_proba.npy", 19)
    action_phase = maybe_add("action_phase", "reports/action_phase/actionId_oof_proba.npy", 19)
    point_tab = maybe_add("point_tabular", "reports/tabular_baseline/pointId_oof_proba.npy", 10)
    point_lstm = maybe_add("point_lstm", "reports/lstm/pointId_oof_proba.npy", 10)
    point_cat = maybe_add("point_catboost", "reports/catboost/pointId_oof_proba.npy", 10)
    point_phase = maybe_add("point_phase", "reports/point_phase/pointId_oof_proba.npy", 10)

    action_ensemble = None
    if action_tab is not None and action_lstm is not None:
        action_ensemble = normalize_probs(0.70 * action_tab + 0.30 * action_lstm, 19)
        if action_cat is not None:
            action_ensemble = normalize_probs(0.85 * action_ensemble + 0.15 * action_cat, 19)
        features = add_probability_group(features, "action_ensemble", action_ensemble)
        added_groups.append("action_ensemble")

    if action_phase is not None and action_ensemble is not None:
        action_phase_weight = 0.0
        action_phase_summary = load_optional_json(Path("reports/action_phase/summary.json"))
        if action_phase_summary:
            action_phase_weight = float(
                action_phase_summary.get("blend_with_existing", {})
                .get("best_weighted_macro_f1", {})
                .get("weight_action_phase", action_phase_weight)
            )
        action_adjusted = normalize_probs((1.0 - action_phase_weight) * action_ensemble + action_phase_weight * action_phase, 19)
        features = add_probability_group(features, "action_adjusted", action_adjusted)
        added_groups.append("action_adjusted")

    if point_tab is not None and point_lstm is not None:
        point_base = normalize_probs(0.35 * point_tab + 0.65 * point_lstm, 10)
        features = add_probability_group(features, "point_base", point_base)
        added_groups.append("point_base")
        point_current = point_base
        if point_phase is not None:
            point_current = normalize_probs(0.60 * point_base + 0.40 * point_phase, 10)
        if point_cat is not None:
            point_current = normalize_probs(0.70 * point_current + 0.30 * point_cat, 10)
        features = add_probability_group(features, "point_current", point_current)
        added_groups.append("point_current")

    summary = {
        "probability_groups": added_groups,
        "forbidden_feature_check": {
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
            "present_in_training_features": sorted([c for c in features.columns if c in FORBIDDEN_FEATURES]),
        },
    }
    return features, added_groups, summary


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    df = pd.read_csv(args.train_features)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.random_state).reset_index(drop=True)

    x, added_groups, feature_summary = build_stacking_frame(df)
    y = df[LABEL_COL].to_numpy(dtype=int)
    groups = df[args.group_col].to_numpy()
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    numeric_cols, categorical_cols = split_columns(x)
    n_splits = min(args.folds, len(pd.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two groups for GroupKFold")
    splitter = GroupKFold(n_splits=n_splits)

    oof_proba = np.zeros((len(df), len(POINT_CLASSES)), dtype=float)
    fold_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_model(numeric_cols, categorical_cols, args)
        fit_kwargs = {}
        if args.model == "extratrees":
            fit_kwargs["model__sample_weight"] = weights[train_idx]
        model.fit(x.iloc[train_idx], y[train_idx], **fit_kwargs)
        proba = aligned_multiclass_proba(model, x.iloc[valid_idx], POINT_CLASSES)
        oof_proba[valid_idx] = proba
        pred = proba.argmax(axis=1)
        fold_rows.append(
            {
                "fold": fold,
                "valid_rows": int(len(valid_idx)),
                "accuracy": float(accuracy_score(y[valid_idx], pred)),
                "macro_f1": float(f1_score(y[valid_idx], pred, average="macro", zero_division=0)),
                "weighted_macro_f1": float(f1_score(y[valid_idx], pred, average="macro", zero_division=0, sample_weight=weights[valid_idx])),
                "log_loss": float(log_loss(y[valid_idx], proba, labels=POINT_CLASSES)),
            }
        )

    metrics = metric_bundle(df, y, oof_proba)
    oof_pred = oof_proba.argmax(axis=1)
    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(report_dir / "pointId_oof_proba.npy", oof_proba)
    pd.DataFrame({"sample_id": df["sample_id"], "y_true": y, "oof_pred": oof_pred}).to_csv(report_dir / "pointId_oof_predictions.csv", index=False)

    blend_rows = []
    current_paths = {
        "tab": Path("reports/tabular_baseline/pointId_oof_proba.npy"),
        "lstm": Path("reports/lstm/pointId_oof_proba.npy"),
        "phase": Path("reports/point_phase/pointId_oof_proba.npy"),
        "cat": Path("reports/catboost/pointId_oof_proba.npy"),
    }
    if all(p.exists() for p in current_paths.values()):
        point_tab = require_probs(current_paths["tab"], 10, len(df))
        point_lstm = require_probs(current_paths["lstm"], 10, len(df))
        point_phase = require_probs(current_paths["phase"], 10, len(df))
        point_cat = require_probs(current_paths["cat"], 10, len(df))
        base = normalize_probs(0.35 * point_tab + 0.65 * point_lstm, 10)
        current = normalize_probs(0.60 * base + 0.40 * point_phase, 10)
        current = normalize_probs(0.70 * current + 0.30 * point_cat, 10)
        for weight in np.arange(0.0, 1.0001, 0.05):
            probs = normalize_probs(weight * oof_proba + (1.0 - weight) * current, 10)
            blend_rows.append({"weight_point_stacking": round(float(weight), 2), **metric_bundle(df, y, probs)})
        pd.DataFrame(blend_rows).to_csv(report_dir / "blend_summary.csv", index=False)

    final_model = make_model(numeric_cols, categorical_cols, args)
    fit_kwargs = {}
    if args.model == "extratrees":
        fit_kwargs["model__sample_weight"] = weights
    final_model.fit(x, y, **fit_kwargs)
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(x.columns),
            "probability_groups": added_groups,
            "classes": POINT_CLASSES,
            "target": "pointId",
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
        },
        output_dir / f"pointId_{args.model}.joblib",
    )

    summary = {
        "target": "pointId",
        "objective": "macro_f1",
        "model": args.model,
        "group_col": args.group_col,
        "folds": n_splits,
        "rows": int(len(df)),
        "features": int(x.shape[1]),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        **metrics,
        "folds_detail": fold_rows,
        **feature_summary,
    }
    if blend_rows:
        blend_df = pd.DataFrame(blend_rows)
        summary["blend_with_current_point_ensemble"] = {
            "best_weighted_macro_f1": blend_df.sort_values(["oof_weighted_macro_f1", "oof_macro_f1"], ascending=False).iloc[0].to_dict(),
            "best_macro_f1": blend_df.sort_values(["oof_macro_f1", "oof_weighted_macro_f1"], ascending=False).iloc[0].to_dict(),
        }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
