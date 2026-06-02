"""Train CatBoost tabular baselines with grouped OOF validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


TARGETS = {
    "actionId": {"label": "label_actionId", "classes": list(range(19)), "binary": False},
    "pointId": {"label": "label_pointId", "classes": list(range(10)), "binary": False},
    "serverGetPoint": {"label": "label_serverGetPoint", "classes": [0, 1], "binary": True},
}
DROP_COLUMNS = {
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
    parser.add_argument("--test-features", default="data/processed/prefix_test_features.csv")
    parser.add_argument("--output-dir", default="models/catboost")
    parser.add_argument("--report-dir", default="reports/catboost")
    parser.add_argument("--group-col", default="match", choices=["match", "rally_uid"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--eval-metric", default="auto", help="CatBoost eval metric; auto uses GPU-safe Logloss for binary GPU runs.")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-sample-weight", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_probs(probs: np.ndarray, dim: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.shape[1] < dim:
        padded = np.zeros((probs.shape[0], dim), dtype=float)
        padded[:, : probs.shape[1]] = probs
        probs = padded
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, row_sum, out=np.full_like(probs, 1.0 / dim), where=row_sum > 0)


def align_model_probs(model: CatBoostClassifier, probs: np.ndarray, classes: list[int]) -> np.ndarray:
    """Align CatBoost probability columns to the full task class list."""

    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    aligned = np.zeros((probs.shape[0], len(classes)), dtype=float)
    class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
    for model_idx, cls in enumerate(model.classes_):
        target_idx = class_to_idx.get(int(cls))
        if target_idx is not None and model_idx < probs.shape[1]:
            aligned[:, target_idx] = probs[:, model_idx]
    return normalize_probs(aligned, len(classes))


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df[[c for c in df.columns if c not in DROP_COLUMNS]].copy()
    cat_cols = [c for c in x.columns if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c])]
    for col in cat_cols:
        x[col] = x[col].fillna("__nan__").astype(str)
    for col in x.columns:
        if col not in cat_cols:
            x[col] = x[col].fillna(-999.0)
    return x


def cat_feature_indices(x: pd.DataFrame) -> list[int]:
    cols = [c for c in x.columns if pd.api.types.is_object_dtype(x[c]) or pd.api.types.is_string_dtype(x[c])]
    return [x.columns.get_loc(c) for c in cols]


def make_model(args: argparse.Namespace, target: str) -> CatBoostClassifier:
    spec = TARGETS[target]
    if args.eval_metric != "auto":
        eval_metric = args.eval_metric
    elif spec["binary"] and args.task_type == "GPU":
        # CatBoost cannot compute AUC as a GPU eval metric; compute AUC later on CPU from predictions.
        eval_metric = "Logloss"
    else:
        eval_metric = "AUC" if spec["binary"] else "MultiClass"
    params = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "random_seed": args.random_state,
        "loss_function": "Logloss" if spec["binary"] else "MultiClass",
        "eval_metric": eval_metric,
        "task_type": args.task_type,
        "verbose": False,
        "allow_writing_files": False,
    }
    return CatBoostClassifier(**params)


def metric_bundle(df: pd.DataFrame, target: str, y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    weights = df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in df else np.ones(len(df), dtype=float)
    mask3 = df["prefix_len"].to_numpy() <= 3
    mask4 = df["prefix_len"].to_numpy() <= 4
    metrics = {
        "oof_accuracy": float(accuracy_score(y_true, pred)),
        "oof_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "oof_weighted_macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0, sample_weight=weights)),
        "oof_log_loss": float(log_loss(y_true, probs, labels=TARGETS[target]["classes"])),
    }
    if target == "serverGetPoint":
        metrics.update(
            {
                "oof_roc_auc": float(roc_auc_score(y_true, probs[:, 1])),
                "oof_weighted_roc_auc": float(roc_auc_score(y_true, probs[:, 1], sample_weight=weights)),
                "prefix_le_3_roc_auc": float(roc_auc_score(y_true[mask3], probs[mask3, 1])) if mask3.any() and len(np.unique(y_true[mask3])) > 1 else float("nan"),
                "prefix_le_4_roc_auc": float(roc_auc_score(y_true[mask4], probs[mask4, 1])) if mask4.any() and len(np.unique(y_true[mask4])) > 1 else float("nan"),
            }
        )
    else:
        metrics.update(
            {
                "prefix_le_3_macro_f1": float(f1_score(y_true[mask3], pred[mask3], average="macro", zero_division=0)) if mask3.any() else float("nan"),
                "prefix_le_4_macro_f1": float(f1_score(y_true[mask4], pred[mask4], average="macro", zero_division=0)) if mask4.any() else float("nan"),
            }
        )
    return metrics


def main() -> None:
    args = parse_args()
    np.random.seed(args.random_state)
    train_df = pd.read_csv(args.train_features)
    test_df = pd.read_csv(args.test_features)
    if args.sample and args.sample < len(train_df):
        train_df = train_df.sample(args.sample, random_state=args.random_state).reset_index(drop=True)

    x_train = make_feature_frame(train_df)
    x_test = make_feature_frame(test_df)
    cat_idx = cat_feature_indices(x_train)
    groups = train_df[args.group_col].to_numpy()
    n_splits = min(args.folds, len(pd.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for target in args.targets:
        spec = TARGETS[target]
        y = train_df[spec["label"]].to_numpy(dtype=int)
        train_weights = train_df["sample_weight"].to_numpy(dtype=float) if args.use_sample_weight and "sample_weight" in train_df else None
        oof_proba = np.zeros((len(train_df), len(spec["classes"])), dtype=float)
        test_fold_probs = []
        fold_rows = []

        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y, groups), start=1):
            model = make_model(args, target)
            train_pool = Pool(x_train.iloc[train_idx], y[train_idx], cat_features=cat_idx, weight=None if train_weights is None else train_weights[train_idx])
            valid_pool = Pool(x_train.iloc[valid_idx], y[valid_idx], cat_features=cat_idx, weight=None if train_weights is None else train_weights[valid_idx])
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=args.early_stopping_rounds)
            proba = align_model_probs(model, model.predict_proba(valid_pool), spec["classes"])
            oof_proba[valid_idx] = proba
            pred = proba.argmax(axis=1)
            row = {
                "fold": fold,
                "valid_rows": int(len(valid_idx)),
                "accuracy": float(accuracy_score(y[valid_idx], pred)),
                "macro_f1": float(f1_score(y[valid_idx], pred, average="macro", zero_division=0)),
                "log_loss": float(log_loss(y[valid_idx], proba, labels=spec["classes"])),
            }
            if spec["binary"] and len(np.unique(y[valid_idx])) > 1:
                row["roc_auc"] = float(roc_auc_score(y[valid_idx], proba[:, 1]))
            fold_rows.append(row)
            model.save_model(output_dir / f"{target}_fold{fold}.cbm")
            test_fold_probs.append(align_model_probs(model, model.predict_proba(Pool(x_test, cat_features=cat_idx)), spec["classes"]))

        test_proba = normalize_probs(np.mean(test_fold_probs, axis=0), len(spec["classes"]))
        final_model = make_model(args, target)
        final_pool = Pool(x_train, y, cat_features=cat_idx, weight=train_weights)
        final_model.fit(final_pool, verbose=False)
        final_model.save_model(output_dir / f"{target}_final.cbm")

        np.save(report_dir / f"{target}_oof_proba.npy", oof_proba)
        np.save(report_dir / f"{target}_test_proba.npy", test_proba)
        pd.DataFrame({"sample_id": train_df["sample_id"], "y_true": y, "oof_pred": oof_proba.argmax(axis=1)}).to_csv(
            report_dir / f"{target}_oof_predictions.csv", index=False
        )
        summary = {
            "target": target,
            "folds": n_splits,
            "rows": int(len(train_df)),
            "features": int(x_train.shape[1]),
            "categorical_features": int(len(cat_idx)),
            "use_sample_weight": bool(train_weights is not None),
            "task_type": args.task_type,
            "eval_metric": make_model(args, target).get_param("eval_metric"),
            "forbidden_feature_check": sorted([c for c in x_train.columns if c in DROP_COLUMNS]),
            **metric_bundle(train_df, target, y, oof_proba),
            "folds_detail": fold_rows,
        }
        summaries.append(summary)

    (report_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
