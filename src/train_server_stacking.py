"""Train a stacked serverGetPoint model from safe tabular and OOF meta features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


LABEL_COL = "label_serverGetPoint"
CLASSES = [0, 1]
FORBIDDEN_FEATURES = {
    "sample_id",
    "rally_uid",
    "source_rally_len",
    "sample_weight",
    "target_strikeNumber",
    "label_actionId",
    "label_pointId",
    "label_serverGetPoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--output-dir", default="models/server_stacking")
    parser.add_argument("--report-dir", default="reports/server_stacking")
    parser.add_argument("--group-col", default="match", choices=["match", "rally_uid"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default="extratrees", choices=["extratrees", "logistic"])
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def normalize_probs(probs: np.ndarray, expected_dim: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[1] < expected_dim:
        padded = np.zeros((probs.shape[0], expected_dim), dtype=float)
        padded[:, : probs.shape[1]] = probs
        probs = padded
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / probs.shape[1]), where=row_sum > 0)


def entropy_from_probs(probs: np.ndarray) -> np.ndarray:
    safe = np.clip(probs, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1)


def add_probability_group(features: pd.DataFrame, group_name: str, probs: np.ndarray) -> pd.DataFrame:
    """Return a defragmented frame with probability-derived features appended."""

    sorted_probs = np.sort(probs, axis=1)
    columns = {f"{group_name}_proba_{idx}": probs[:, idx] for idx in range(probs.shape[1])}
    columns.update(
        {
            f"{group_name}_argmax": probs.argmax(axis=1),
            f"{group_name}_max_prob": probs.max(axis=1),
            f"{group_name}_entropy": entropy_from_probs(probs),
            f"{group_name}_top2_margin": sorted_probs[:, -1] - sorted_probs[:, -2],
        }
    )
    return pd.concat([features, pd.DataFrame(columns, index=features.index)], axis=1).copy()


def make_base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in FORBIDDEN_FEATURES]].copy()


def split_columns(x: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_cols = [c for c in x.columns if is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_model(numeric_cols: List[str], categorical_cols: List[str], args: argparse.Namespace) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if args.model == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols), ("cat", categorical_pipe, categorical_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if args.model == "extratrees":
        estimator = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        )
    else:
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def aligned_binary_proba(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(CLASSES)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        cls = int(cls)
        if cls in CLASSES:
            out[:, CLASSES.index(cls)] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 0.5), where=row_sum > 0)


def metric_bundle(df: pd.DataFrame, y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    pos = probs[:, 1]
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    mask3 = df["prefix_len"].to_numpy() <= 3 if "prefix_len" in df else np.zeros(len(df), dtype=bool)
    mask4 = df["prefix_len"].to_numpy() <= 4 if "prefix_len" in df else np.zeros(len(df), dtype=bool)
    return {
        "oof_accuracy": float(accuracy_score(y_true, pred)),
        "oof_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "oof_log_loss": float(log_loss(y_true, probs, labels=CLASSES)),
        "oof_roc_auc": float(roc_auc_score(y_true, pos)),
        "oof_weighted_roc_auc": float(roc_auc_score(y_true, pos, sample_weight=weights)),
        "prefix_le_3_roc_auc": float(roc_auc_score(y_true[mask3], pos[mask3])) if mask3.any() and len(np.unique(y_true[mask3])) > 1 else float("nan"),
        "prefix_le_4_roc_auc": float(roc_auc_score(y_true[mask4], pos[mask4])) if mask4.any() and len(np.unique(y_true[mask4])) > 1 else float("nan"),
    }


def load_blend_weight(summary_csv: Path, target: str) -> float | None:
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    row = df[df["target"] == target]
    if row.empty or "weight_lstm" not in row.columns:
        return None
    return float(row.iloc[0]["weight_lstm"])


def require_probs(path: Path, expected_dim: int, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF probabilities: {path}")
    probs = normalize_probs(np.load(path), expected_dim)
    if probs.shape[0] != expected_rows:
        raise ValueError(f"Row mismatch for {path}: expected {expected_rows}, got {probs.shape[0]}")
    return probs


def build_stacking_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    features = make_base_feature_frame(df)
    n_rows = len(df)
    added_groups: list[str] = []

    action_tab = require_probs(Path("reports/tabular_baseline/actionId_oof_proba.npy"), 19, n_rows)
    point_tab = require_probs(Path("reports/tabular_baseline/pointId_oof_proba.npy"), 10, n_rows)
    action_lstm = require_probs(Path("reports/lstm/actionId_oof_proba.npy"), 19, n_rows)
    point_lstm = require_probs(Path("reports/lstm/pointId_oof_proba.npy"), 10, n_rows)

    for name, probs in [
        ("action_tabular", action_tab),
        ("point_tabular", point_tab),
        ("action_lstm", action_lstm),
        ("point_lstm", point_lstm),
    ]:
        features = add_probability_group(features, name, probs)
        added_groups.append(name)

    ensemble_summary = Path("reports/ensemble/summary.csv")
    point_base_blend = None
    if ensemble_summary.exists():
        action_weight = load_blend_weight(ensemble_summary, "actionId")
        point_weight = load_blend_weight(ensemble_summary, "pointId")
        if action_weight is not None:
            action_ensemble = normalize_probs(action_weight * action_lstm + (1.0 - action_weight) * action_tab, 19)
            features = add_probability_group(features, "action_ensemble", action_ensemble)
            added_groups.append("action_ensemble")
        if point_weight is not None:
            point_base_blend = normalize_probs(point_weight * point_lstm + (1.0 - point_weight) * point_tab, 10)
            features = add_probability_group(features, "point_ensemble", point_base_blend)
            added_groups.append("point_ensemble")

    point_phase_weight = 0.40
    point_phase_summary_path = Path("reports/point_phase/summary.json")
    point_phase_path = Path("reports/point_phase/pointId_oof_proba.npy")
    if point_phase_path.exists():
        point_phase = require_probs(point_phase_path, 10, n_rows)
        features = add_probability_group(features, "point_phase", point_phase)
        added_groups.append("point_phase")
        if point_phase_summary_path.exists():
            point_phase_summary = json.loads(point_phase_summary_path.read_text(encoding="utf-8"))
            point_phase_weight = float(
                point_phase_summary.get("blend_with_existing", {})
                .get("best_weighted_macro_f1", {})
                .get("weight_point_phase", point_phase_weight)
            )
        if point_base_blend is not None:
            point_phase_blend = normalize_probs(point_phase_weight * point_phase + (1.0 - point_phase_weight) * point_base_blend, 10)
            features = add_probability_group(features, "point_phase_blend", point_phase_blend)
            added_groups.append("point_phase_blend")

    summary = {
        "probability_groups": added_groups,
        "point_phase_blend_weight": point_phase_weight,
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
    oof_proba = np.zeros((len(df), 2), dtype=float)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = make_model(numeric_cols, categorical_cols, args)
        fit_kwargs = {}
        if args.model == "extratrees":
            fit_kwargs["model__sample_weight"] = weights[train_idx]
        model.fit(x.iloc[train_idx], y[train_idx], **fit_kwargs)
        proba = aligned_binary_proba(model, x.iloc[valid_idx])
        oof_proba[valid_idx] = proba
        pred = proba.argmax(axis=1)
        row = {
            "fold": fold,
            "valid_rows": int(len(valid_idx)),
            "accuracy": float(accuracy_score(y[valid_idx], pred)),
            "macro_f1": float(f1_score(y[valid_idx], pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y[valid_idx], proba, labels=CLASSES)),
        }
        if len(np.unique(y[valid_idx])) > 1:
            row["roc_auc"] = float(roc_auc_score(y[valid_idx], proba[:, 1]))
            row["weighted_roc_auc"] = float(roc_auc_score(y[valid_idx], proba[:, 1], sample_weight=weights[valid_idx]))
        fold_rows.append(row)

    metrics = metric_bundle(df, y, oof_proba)
    oof_pred = oof_proba.argmax(axis=1)

    report_dir = Path(args.report_dir)
    output_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(report_dir / "serverGetPoint_oof_proba.npy", oof_proba)
    pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "y_true": y,
            "oof_pred": oof_pred,
            "oof_proba_1": oof_proba[:, 1],
        }
    ).to_csv(report_dir / "serverGetPoint_oof_predictions.csv", index=False)

    blend_rows = []
    ensemble_summary = Path("reports/ensemble/summary.csv")
    server_weight = load_blend_weight(ensemble_summary, "serverGetPoint") if ensemble_summary.exists() else None
    if server_weight is not None:
        tab_server = require_probs(Path("reports/tabular_baseline/serverGetPoint_oof_proba.npy"), 2, len(df))
        lstm_server = require_probs(Path("reports/lstm/serverGetPoint_oof_proba.npy"), 2, len(df))
        server_base = normalize_probs(server_weight * lstm_server + (1.0 - server_weight) * tab_server, 2)
        for weight in np.arange(0.0, 1.0001, 0.05):
            probs = normalize_probs(weight * oof_proba + (1.0 - weight) * server_base, 2)
            blend_rows.append({"weight_server_stacking": round(float(weight), 2), **metric_bundle(df, y, probs)})
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
            "classes": CLASSES,
            "target": "serverGetPoint",
            "forbidden_features": sorted(FORBIDDEN_FEATURES),
        },
        output_dir / f"serverGetPoint_{args.model}.joblib",
    )

    summary = {
        "target": "serverGetPoint",
        "objective": "roc_auc",
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
        summary["blend_with_current_server_ensemble"] = {
            "base_weight_lstm": server_weight,
            "best_weighted_roc_auc": blend_df.sort_values(["oof_weighted_roc_auc", "oof_roc_auc"], ascending=False).iloc[0].to_dict(),
            "best_roc_auc": blend_df.sort_values(["oof_roc_auc", "oof_weighted_roc_auc"], ascending=False).iloc[0].to_dict(),
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
