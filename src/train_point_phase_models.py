"""Train phase-specific pointId tabular models with grouped OOF validation."""

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
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


POINT_CLASSES = list(range(10))
LABEL_COL = "label_pointId"
DROP_COLUMNS = {
    "sample_id",
    "rally_uid",
    "source_rally_len",
    "target_strikeNumber",
    "label_actionId",
    "label_pointId",
    "label_serverGetPoint",
}
DEFAULT_BUCKETS = {
    "receive": (2, 2),
    "third_ball": (3, 3),
    "early_rally": (4, 5),
    "rally": (6, None),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default="data/processed/prefix_train_features.csv")
    parser.add_argument("--test-features", default="data/processed/prefix_test_features.csv")
    parser.add_argument("--output-dir", default="models/point_phase")
    parser.add_argument("--report-dir", default="reports/point_phase")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--min-bucket-train-rows", type=int, default=100)
    parser.add_argument("--blend-with-existing", default="reports/ensemble/summary.csv")
    return parser.parse_args()


def assign_bucket(series: pd.Series) -> pd.Series:
    values = series.astype(int)
    bucket = pd.Series("other", index=series.index, dtype=object)
    for name, (lo, hi) in DEFAULT_BUCKETS.items():
        mask = values >= lo if hi is None else ((values >= lo) & (values <= hi))
        bucket.loc[mask] = name
    return bucket


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if c not in DROP_COLUMNS]].copy()


def split_columns(x: pd.DataFrame) -> tuple[List[str], List[str]]:
    numeric_cols = [c for c in x.columns if is_numeric_dtype(x[c])]
    categorical_cols = [c for c in x.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def make_model(numeric_cols: List[str], categorical_cols: List[str], args: argparse.Namespace) -> Pipeline:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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
    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def aligned_proba(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    model_classes = list(model.named_steps["model"].classes_)
    out = np.zeros((len(x), len(POINT_CLASSES)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        if cls in POINT_CLASSES:
            out[:, POINT_CLASSES.index(int(cls))] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / len(POINT_CLASSES)), where=row_sum > 0)


def normalize_proba(probs: np.ndarray) -> np.ndarray:
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / probs.shape[1]), where=row_sum > 0)


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, weights: np.ndarray, args: argparse.Namespace) -> Pipeline:
    numeric_cols, categorical_cols = split_columns(x_train)
    model = make_model(numeric_cols, categorical_cols, args)
    model.fit(x_train, y_train, model__sample_weight=weights)
    return model


def can_train_bucket(y: np.ndarray, min_rows: int) -> bool:
    return len(y) >= min_rows and len(np.unique(y)) >= 2


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
        "log_loss": float(log_loss(y, probs, labels=POINT_CLASSES)),
        "prefix_le_3_macro_f1": float(f1_score(y[mask3], pred[mask3], average="macro", zero_division=0)) if mask3.any() else float("nan"),
        "prefix_le_4_macro_f1": float(f1_score(y[mask4], pred[mask4], average="macro", zero_division=0)) if mask4.any() else float("nan"),
    }


def load_best_point_ensemble_weight(path: Path) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["target"] == "pointId"]
    if row.empty or "weight_lstm" not in row.columns:
        return None
    return float(row.iloc[0]["weight_lstm"])


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)

    train_df = pd.read_csv(args.train_features).copy()
    test_df = pd.read_csv(args.test_features).copy()
    if args.sample and args.sample < len(train_df):
        train_df = train_df.sample(args.sample, random_state=args.random_state).reset_index(drop=True)

    train_df = train_df.assign(phase_bucket=assign_bucket(train_df["next_strikeNumber"]))
    test_df = test_df.assign(phase_bucket=assign_bucket(test_df["next_strikeNumber"]))
    x_all = make_feature_frame(train_df)
    y_all = train_df[LABEL_COL].astype(int).to_numpy()
    weights_all = train_df["sample_weight"].astype(float).to_numpy() if "sample_weight" in train_df else np.ones(len(train_df), dtype=float)
    groups = train_df["match"].to_numpy()

    n_splits = min(args.folds, len(pd.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros((len(train_df), len(POINT_CLASSES)), dtype=float)
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

    oof_pred = oof_proba.argmax(axis=1)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    final_global = fit_model(x_all, y_all, weights_all, args)
    joblib.dump({"model": final_global, "feature_columns": list(x_all.columns), "classes": POINT_CLASSES, "bucket": "global"}, output_dir / "global.joblib")

    for bucket_name in list(DEFAULT_BUCKETS) + ["other"]:
        bucket_mask = train_df["phase_bucket"].to_numpy() == bucket_name
        if can_train_bucket(y_all[bucket_mask], args.min_bucket_train_rows):
            model = fit_model(x_all.iloc[bucket_mask], y_all[bucket_mask], weights_all[bucket_mask], args)
            bundle = {"model": model, "feature_columns": list(x_all.columns), "classes": POINT_CLASSES, "bucket": bucket_name}
        else:
            bundle = {"model": final_global, "feature_columns": list(x_all.columns), "classes": POINT_CLASSES, "bucket": bucket_name, "fallback": "global"}
        joblib.dump(bundle, output_dir / f"{bucket_name}.joblib")

    np.save(report_dir / "pointId_oof_proba.npy", oof_proba)
    pd.DataFrame(
        {"sample_id": train_df["sample_id"], "y_true": y_all, "oof_pred": oof_pred, "phase_bucket": train_df["phase_bucket"]}
    ).to_csv(report_dir / "pointId_oof_predictions.csv", index=False)
    pd.DataFrame(per_bucket_rows).to_csv(report_dir / "per_bucket_metrics.csv", index=False)

    summary = {"target": "pointId", "folds": n_splits, "oof": metric_bundle(train_df, oof_proba)}

    blend_path = Path(args.blend_with_existing)
    best_weight_lstm = load_best_point_ensemble_weight(blend_path)
    tab_path = Path("reports/tabular_baseline/pointId_oof_proba.npy")
    lstm_path = Path("reports/lstm/pointId_oof_proba.npy")
    if best_weight_lstm is not None and tab_path.exists() and lstm_path.exists():
        tab = np.load(tab_path)
        lstm = np.load(lstm_path)
        base = normalize_proba(best_weight_lstm * lstm + (1.0 - best_weight_lstm) * tab)
        blend_rows = []
        for w in np.arange(0.0, 1.0001, 0.05):
            blend = normalize_proba(w * oof_proba + (1.0 - w) * base)
            pred = blend.argmax(axis=1)
            mask3 = train_df["prefix_len"].to_numpy() <= 3
            mask4 = train_df["prefix_len"].to_numpy() <= 4
            blend_rows.append(
                {
                    "weight_point_phase": round(float(w), 2),
                    "macro_f1": float(f1_score(y_all, pred, average="macro", zero_division=0)),
                    "weighted_macro_f1": float(f1_score(y_all, pred, average="macro", zero_division=0, sample_weight=weights_all)),
                    "prefix_le_3_macro_f1": float(f1_score(y_all[mask3], pred[mask3], average="macro", zero_division=0)),
                    "prefix_le_4_macro_f1": float(f1_score(y_all[mask4], pred[mask4], average="macro", zero_division=0)),
                    "log_loss": float(log_loss(y_all, blend, labels=POINT_CLASSES)),
                }
            )
        blend_df = pd.DataFrame(blend_rows)
        blend_df.to_csv(report_dir / "pointId_blend_summary.csv", index=False)
        summary["blend_with_existing"] = {
            "base_weight_lstm": best_weight_lstm,
            "best_weighted_macro_f1": blend_df.sort_values(["weighted_macro_f1", "macro_f1"], ascending=False).iloc[0].to_dict(),
            "best_macro_f1": blend_df.sort_values(["macro_f1", "weighted_macro_f1"], ascending=False).iloc[0].to_dict(),
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
