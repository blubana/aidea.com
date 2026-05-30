"""Generate tabular-baseline submission from saved model bundles."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TARGETS = ["actionId", "pointId", "serverGetPoint"]


def predict_target(bundle_path: Path, features: pd.DataFrame, target: str) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    classes = np.asarray(bundle["classes"])
    x = features[feature_columns].copy()
    raw_proba = model.predict_proba(x)
    model_classes = np.asarray(model.named_steps["model"].classes_)
    proba = np.zeros((len(features), len(classes)), dtype=float)
    for src_idx, cls in enumerate(model_classes):
        dst = np.where(classes == cls)[0]
        if len(dst):
            proba[:, dst[0]] = raw_proba[:, src_idx]
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / len(classes)), where=row_sum > 0)
    if target == "actionId":
        # Test prefixes always contain at least one observed stroke, so the
        # target is stroke >= 2. Serve classes 15-18 are only valid for stroke 1.
        # Remove impossible serve predictions to directly improve action Macro F1.
        serve_mask = np.isin(classes, [15, 16, 17, 18])
        non_first_target = features["target_strikeNumber"].to_numpy() >= 2
        proba[np.ix_(non_first_target, serve_mask)] = 0.0
        row_sum = proba.sum(axis=1, keepdims=True)
        proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    return classes[np.argmax(proba, axis=1)]


def main() -> None:
    features = pd.read_csv("data/processed/prefix_test_features.csv")
    out = pd.DataFrame({"rally_uid": features["sample_id"].astype(int)})
    model_dir = Path("models/tabular_baseline")
    for target in TARGETS:
        out[target] = predict_target(model_dir / f"{target}_extratrees.joblib", features, target)
    submission_dir = Path("submissions")
    submission_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(submission_dir / "submission_tabular_baseline.csv", index=False)


if __name__ == "__main__":
    main()
