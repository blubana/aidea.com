from __future__ import annotations

import argparse
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.dataset import encode_features, feature_values, resolve_features, sort_rallies
from train import fit_categories, make_class_maps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def banner(console: Console) -> None:
    console.print("[bold green]+--------------------------------------------+[/]")
    console.print("[bold green]|  AI CUP Rally Tabular Trainer             |[/]")
    console.print("[bold green]+--------------------------------------------+[/]")


def flatten_prefix(feat: np.ndarray, t: int, window: int) -> np.ndarray:
    prefix = feat[:t]
    out = np.zeros((window, feat.shape[1]), dtype=np.float32)
    tail = prefix[-window:]
    out[-len(tail) :] = tail
    return out.reshape(-1)


def build_tabular_samples(
    df: pd.DataFrame,
    cats: dict[str, list[int]],
    features: list[str],
    act_id2idx: dict[int, int],
    pt_id2idx: dict[int, int],
    window: int,
    mode: str,
    target_from_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = sort_rallies(df)
    feat = encode_features(df, cats, features)
    action_raw = df["actionId"].to_numpy()
    point_raw = df["pointId"].to_numpy()
    rally_label = df["serverGetPoint"].to_numpy(dtype=np.float32)

    X: list[np.ndarray] = []
    y_a: list[int] = []
    y_p: list[int] = []
    y_r: list[float] = []
    for _, idx in df.groupby("rally_uid", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        n = int(idx_arr.shape[0])
        if n < 2:
            continue
        if mode == "target_from_end":
            t = n - max(1, target_from_end)
            steps = [t] if t >= 1 else []
        else:
            steps = range(1, n)
        rally_feat = feat[idx_arr]
        for t in steps:
            X.append(flatten_prefix(rally_feat, t, window))
            y_a.append(act_id2idx.get(int(action_raw[idx_arr[t]]), -1))
            y_p.append(pt_id2idx.get(int(point_raw[idx_arr[t]]), -1))
            y_r.append(float(rally_label[idx_arr[0]]))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y_a, dtype=np.int64),
        np.asarray(y_p, dtype=np.int64),
        np.asarray(y_r, dtype=np.int64),
    )


def make_model(backend: str, task: str, seed: int, max_iter: int):
    if backend in {"auto", "lightgbm"}:
        try:
            import lightgbm as lgb

            objective = "binary" if task == "rally" else "multiclass"
            return "lightgbm", lgb.LGBMClassifier(
                objective=objective,
                n_estimators=max_iter,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            if backend == "lightgbm":
                raise

    if backend in {"auto", "catboost"}:
        try:
            from catboost import CatBoostClassifier

            return "catboost", CatBoostClassifier(
                iterations=max_iter,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss" if task == "rally" else "MultiClass",
                random_seed=seed,
                verbose=False,
            )
        except Exception:
            if backend == "catboost":
                raise

    return "sklearn", HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        early_stopping=False,
        random_state=seed,
    )


def positive_class_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba
    classes = getattr(model, "classes_", np.asarray([0, 1]))
    if 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])]
    return proba[:, -1]


def evaluate(models: dict, X: np.ndarray, y_a: np.ndarray, y_p: np.ndarray, y_r: np.ndarray) -> dict[str, float]:
    pa = models["action"].predict(X)
    pp = models["point"].predict(X)
    pr = positive_class_proba(models["rally"], X)
    f1_a = f1_score(y_a, pa, average="macro", zero_division=0) if len(y_a) else 0.0
    f1_p = f1_score(y_p, pp, average="macro", zero_division=0) if len(y_p) else 0.0
    auc = roc_auc_score(y_r, pr) if len(set(y_r.tolist())) > 1 else 0.5
    return {
        "f1_action": float(f1_a),
        "f1_point": float(f1_p),
        "auc": float(auc),
        "score": float(0.4 * f1_a + 0.4 * f1_p + 0.2 * auc),
    }


def train_one_split(args, console, train_df, tr_ids, va_ids, run_label: str = "") -> dict[str, float]:
    features = resolve_features(args.feature_set)
    tr_df = train_df[train_df["rally_uid"].isin(tr_ids)].copy()
    va_df = train_df[train_df["rally_uid"].isin(va_ids)].copy()
    cats = fit_categories(tr_df, features)
    action_classes, point_classes, act_id2idx, pt_id2idx = make_class_maps(tr_df)

    X_tr, yA_tr, yP_tr, yR_tr = build_tabular_samples(
        tr_df, cats, features, act_id2idx, pt_id2idx, args.window, "dense", args.val_target_from_end
    )
    X_va, yA_va, yP_va, yR_va = build_tabular_samples(
        va_df, cats, features, act_id2idx, pt_id2idx, args.window, "target_from_end", args.val_target_from_end
    )

    backend_used = ""
    models = {}
    for task, y in {"action": yA_tr, "point": yP_tr, "rally": yR_tr}.items():
        backend_used, model = make_model(args.backend, task, args.seed, args.max_iter)
        model.fit(X_tr, y)
        models[task] = model

    metrics = evaluate(models, X_va, yA_va, yP_va, yR_va)
    out_dir = Path(args.out_dir) / run_label if run_label else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"tabular_{backend_used}_{stamp}.pkl"
    artifact = {
        "artifact_type": "tabular_ensemble",
        "backend": backend_used,
        "models": models,
        "metadata": {
            "features": features,
            "feature_set": args.feature_set,
            "cats": cats,
            "action_classes": action_classes,
            "point_classes": point_classes,
            "window": args.window,
            "val_target_from_end": args.val_target_from_end,
            "metrics": metrics,
        },
    }
    with out_path.open("wb") as f:
        pickle.dump(artifact, f)

    table = Table(title=f"Tabular run{f' ({run_label})' if run_label else ''}")
    table.add_column("item")
    table.add_column("value")
    for k, v in {
        "backend": backend_used,
        "train samples": len(X_tr),
        "val samples": len(X_va),
        "feature set": args.feature_set,
        "window": args.window,
        "f1_action": f"{metrics['f1_action']:.4f}",
        "f1_point": f"{metrics['f1_point']:.4f}",
        "auc": f"{metrics['auc']:.4f}",
        "score": f"{metrics['score']:.4f}",
        "artifact": str(out_path),
    }.items():
        table.add_row(str(k), str(v))
    console.print(table)
    metrics["out_path"] = str(out_path)
    return metrics


def iter_cv_splits(rally_ids: np.ndarray, labels: np.ndarray, folds: int, seed: int):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(rally_ids, labels), start=1):
        yield f"fold{fold}", rally_ids[tr_idx], rally_ids[va_idx]


def main(args: argparse.Namespace) -> None:
    console = Console()
    banner(console)
    seed_everything(args.seed)
    data_dir = Path(args.data_dir)
    train_csv = Path(args.train) if args.train else data_dir / "train.csv"
    train_df = sort_rallies(pd.read_csv(train_csv))
    if args.limit_rallies > 0:
        keep = train_df["rally_uid"].drop_duplicates().head(args.limit_rallies)
        train_df = train_df[train_df["rally_uid"].isin(keep)].copy()

    rally_labels = train_df.groupby("rally_uid", sort=False)["serverGetPoint"].first()
    rally_ids = rally_labels.index.to_numpy()
    labels = rally_labels.to_numpy()
    if args.cv_folds <= 1:
        tr_ids, va_ids = train_test_split(
            rally_ids,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
        train_one_split(args, console, train_df, tr_ids, va_ids)
        return

    results = []
    for run_label, tr_ids, va_ids in iter_cv_splits(rally_ids, labels, args.cv_folds, args.seed):
        metrics = train_one_split(args, console, train_df, tr_ids, va_ids, run_label)
        metrics["run_label"] = run_label
        results.append(metrics)
    summary = pd.DataFrame(results)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(args.out_dir) / "tabular_cv_summary.csv", index=False)
    console.print(summary[["run_label", "f1_action", "f1_point", "auc", "score", "out_path"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--train", default="")
    parser.add_argument("--out-dir", default="checkpoints_tabular")
    parser.add_argument("--feature-set", choices=("base", "score", "enhanced", "semantic"), default="semantic")
    parser.add_argument("--backend", choices=("auto", "lightgbm", "catboost", "sklearn"), default="auto")
    parser.add_argument("--window", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--val-target-from-end", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--limit-rallies", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
