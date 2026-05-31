"""Generate the final blended submission from saved tabular, LSTM, phase, stacking, and CatBoost models."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models_lstm import RallyLSTMClassifier, RallyLSTMServerBinary
from train_point_phase_models import DEFAULT_BUCKETS, assign_bucket
from train_server_stacking import FORBIDDEN_FEATURES, add_probability_group, make_base_feature_frame, normalize_probs


TARGET_DIMS = {"actionId": 19, "pointId": 10, "serverGetPoint": 2}
TARGET_BINARY = {"actionId": False, "pointId": False, "serverGetPoint": True}
FINAL_WEIGHTS = {
    "action": {"tabular": 0.70, "lstm": 0.30},
    "action_catboost": 0.15,
    "action_phase": 0.0,
    "point_base": {"tabular": 0.35, "lstm": 0.65},
    "point_final": {"base": 0.60, "phase": 0.40},
    "point_catboost": 0.30,
    "server_base": {"tabular": 0.80, "lstm": 0.20},
    "server_final": {"base": 0.35, "stacking": 0.65},
    "server_catboost": 0.55,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-output", default="float", choices=["float", "bool"])
    parser.add_argument("--use-cross-target-stacking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--submission-path", default="submissions/submission_final_blend.csv")
    parser.add_argument("--class-multiplier-dir", default="reports/class_multipliers")
    parser.add_argument("--use-class-multipliers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-action-phase", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-phase-model-dir", default="models/action_phase")
    return parser.parse_args()


class TestLSTMDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, lengths: np.ndarray, x_manual: np.ndarray):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.lengths = torch.as_tensor(lengths, dtype=torch.long)
        self.x_manual = torch.as_tensor(x_manual, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x_cat)

    def __getitem__(self, idx: int):
        return self.x_cat[idx], self.x_num[idx], self.lengths[idx], self.x_manual[idx]


def aligned_tabular_proba(bundle_path: Path, features: pd.DataFrame, target: str) -> np.ndarray:
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    classes = np.asarray(bundle["classes"], dtype=int)
    x = features[feature_columns].copy()
    raw_proba = model.predict_proba(x)
    estimator_classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    out = np.zeros((len(features), TARGET_DIMS[target]), dtype=float)
    for src_idx, cls in enumerate(estimator_classes):
        dst = np.where(classes == cls)[0]
        if len(dst):
            out[:, int(classes[dst[0]])] = raw_proba[:, src_idx]
    return normalize_probs(out, TARGET_DIMS[target])


def make_lstm_model(checkpoint: dict, x_cat: np.ndarray, x_num: np.ndarray, x_manual: np.ndarray):
    args = checkpoint["args"]
    target = checkpoint["target"]
    binary = TARGET_BINARY[target]
    output_dim = 1 if binary else TARGET_DIMS[target]
    state_dict = checkpoint["model_state_dict"]
    cardinals = [
        int(state_dict[f"embeddings.{idx}.weight"].shape[0])
        for idx in range(x_cat.shape[-1])
    ]
    cls = RallyLSTMServerBinary if binary else RallyLSTMClassifier
    return cls(
        cat_cardinalities=cardinals,
        num_numeric_features=x_num.shape[-1],
        manual_dim=x_manual.shape[-1],
        hidden_size=int(args.get("hidden_size", 128)),
        num_layers=int(args.get("num_layers", 1)),
        dropout=float(args.get("dropout", 0.1)),
        output_dim=output_dim,
    )


def clamp_categorical_indices(x_cat: np.ndarray, model) -> np.ndarray:
    x_cat = x_cat.copy()
    for idx, emb in enumerate(model.embeddings):
        invalid = x_cat[:, :, idx] >= emb.num_embeddings
        if invalid.any():
            x_cat[:, :, idx][invalid] = 0
    return x_cat


def standardize_test_inputs(checkpoint: dict, x_manual: np.ndarray, x_num: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    manual_mean = np.asarray(checkpoint["manual_mean"], dtype=np.float32)
    manual_std = np.asarray(checkpoint["manual_std"], dtype=np.float32)
    num_mean = np.asarray(checkpoint["num_mean"], dtype=np.float32)
    num_std = np.asarray(checkpoint["num_std"], dtype=np.float32)
    x_manual_std = ((x_manual.astype(np.float32) - manual_mean) / manual_std).astype(np.float32)
    x_num_std = x_num.copy().astype(np.float32)
    mask = np.arange(x_num.shape[1])[None, :] < lengths[:, None]
    x_num_std[mask] = ((x_num_std[mask] - num_mean) / num_std).astype(np.float32)
    return x_manual_std, x_num_std


def predict_lstm_target(target: str, npz_path: Path, model_dir: Path, batch_size: int = 512) -> np.ndarray:
    checkpoints = sorted(model_dir.glob(f"{target}_fold*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Missing LSTM checkpoints for {target}")
    data = np.load(npz_path, allow_pickle=True)
    base_x_cat = data["test_X_cat_seq"]
    base_x_num = data["test_X_num_seq"]
    base_lengths = data["test_lengths"]
    base_x_manual = data["test_X_manual"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_probs = []
    with torch.no_grad():
        for checkpoint_path in checkpoints:
            # Checkpoints are generated locally by src/train_lstm.py and contain
            # NumPy arrays for fold-local normalization stats, so PyTorch's
            # default weights_only=True loader cannot deserialize them.
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = make_lstm_model(checkpoint, base_x_cat, base_x_num, base_x_manual)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval().to(device)
            x_manual, x_num = standardize_test_inputs(checkpoint, base_x_manual, base_x_num, base_lengths)
            x_cat = clamp_categorical_indices(base_x_cat, model)
            loader = DataLoader(TestLSTMDataset(x_cat, x_num, base_lengths, x_manual), batch_size=batch_size, shuffle=False)
            parts = []
            for batch in loader:
                x_cat_b, x_num_b, lengths_b, x_manual_b = [b.to(device) for b in batch]
                logits = model(x_cat_b, x_num_b, lengths_b, x_manual_b)
                if TARGET_BINARY[target]:
                    probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
                    probs = np.column_stack([1.0 - probs, probs])
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                parts.append(probs)
            fold_probs.append(np.concatenate(parts, axis=0))
    return normalize_probs(np.mean(fold_probs, axis=0), TARGET_DIMS[target])


def load_phase_models(model_dir: Path) -> dict[str, dict]:
    bundles = {}
    for bucket in list(DEFAULT_BUCKETS) + ["other", "global"]:
        path = model_dir / f"{bucket}.joblib"
        if path.exists():
            bundles[bucket] = joblib.load(path)
    if "global" not in bundles:
        raise FileNotFoundError("Missing point phase global model")
    return bundles


def aligned_bundle_predict(bundle: dict, features: pd.DataFrame, dim: int) -> np.ndarray:
    model = bundle["model"]
    x = features[bundle["feature_columns"]].copy()
    raw = model.predict_proba(x)
    estimator_classes = np.asarray(model.named_steps["model"].classes_, dtype=int)
    out = np.zeros((len(features), dim), dtype=float)
    for src_idx, cls in enumerate(estimator_classes):
        if 0 <= int(cls) < dim:
            out[:, int(cls)] = raw[:, src_idx]
    return normalize_probs(out, dim)


def load_catboost_test_probs(path: Path, dim: int, expected_rows: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing CatBoost test probabilities: {path}")
    probs = normalize_probs(np.load(path), dim)
    if len(probs) != expected_rows:
        raise ValueError(f"CatBoost probability row mismatch for {path}: expected {expected_rows}, got {len(probs)}")
    return probs


def predict_point_phase(features: pd.DataFrame, model_dir: Path) -> np.ndarray:
    bundles = load_phase_models(model_dir)
    buckets = assign_bucket(features["next_strikeNumber"])
    features = features.copy()
    features["phase_bucket"] = buckets
    out = np.zeros((len(features), 10), dtype=float)
    for bucket in list(DEFAULT_BUCKETS) + ["other"]:
        mask = (buckets == bucket).to_numpy()
        if not mask.any():
            continue
        out[mask] = aligned_bundle_predict(bundles.get(bucket, bundles["global"]), features.loc[mask], 10)
    return normalize_probs(out, 10)


def predict_action_phase(features: pd.DataFrame, model_dir: Path) -> np.ndarray:
    bundles = load_phase_models(model_dir)
    buckets = assign_bucket(features["next_strikeNumber"])
    features = features.copy()
    features["phase_bucket"] = buckets
    out = np.zeros((len(features), 19), dtype=float)
    for bucket in list(DEFAULT_BUCKETS) + ["other"]:
        mask = (buckets == bucket).to_numpy()
        if not mask.any():
            continue
        out[mask] = aligned_bundle_predict(bundles.get(bucket, bundles["global"]), features.loc[mask], 19)
    return normalize_probs(out, 19)


def apply_action_serve_mask(probs: np.ndarray, test_features: pd.DataFrame) -> np.ndarray:
    out = probs.copy()
    mask = test_features["target_strikeNumber"].to_numpy() >= 2
    out[np.ix_(mask, np.array([15, 16, 17, 18]))] = 0.0
    return normalize_probs(out, out.shape[1])


def build_server_stacking_test_frame(
    test_features: pd.DataFrame,
    action_tab: np.ndarray,
    point_tab: np.ndarray,
    action_lstm: np.ndarray,
    point_lstm: np.ndarray,
    action_ensemble: np.ndarray,
    point_ensemble: np.ndarray,
    point_phase: np.ndarray,
    point_phase_blend: np.ndarray,
) -> pd.DataFrame:
    x = make_base_feature_frame(test_features)
    for name, probs in [
        ("action_tabular", action_tab),
        ("point_tabular", point_tab),
        ("action_lstm", action_lstm),
        ("point_lstm", point_lstm),
        ("action_ensemble", action_ensemble),
        ("point_ensemble", point_ensemble),
        ("point_phase", point_phase),
        ("point_phase_blend", point_phase_blend),
    ]:
        add_probability_group(x, name, probs)
    forbidden_present = [c for c in x.columns if c in FORBIDDEN_FEATURES]
    if forbidden_present:
        raise ValueError(f"Forbidden features in server stacking test frame: {forbidden_present}")
    return x


def validate_submission(df: pd.DataFrame, expected_rows: int, server_output: str) -> None:
    if len(df) != expected_rows:
        raise ValueError(f"Submission row mismatch: expected {expected_rows}, got {len(df)}")
    if df.isna().any().any():
        raise ValueError("Submission contains NaN values")
    if not df["actionId"].between(0, 18).all():
        raise ValueError("actionId out of range")
    if not df["pointId"].between(0, 9).all():
        raise ValueError("pointId out of range")
    if not df["serverGetPoint"].between(0, 1).all():
        raise ValueError("serverGetPoint out of range")
    if server_output == "bool":
        values = df["serverGetPoint"].to_numpy()
        if not np.isin(values, [0, 1]).all():
            raise ValueError("serverGetPoint bool mode must contain only 0/1")


def save_probabilities(report_dir: Path, arrays: Iterable[tuple[str, np.ndarray]]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays:
        np.save(report_dir / f"{name}.npy", arr)


def load_class_multipliers(path: Path, expected_dim: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing class multipliers: {path}")
    values = np.asarray(np.load(path), dtype=float).reshape(-1)
    if len(values) != expected_dim:
        raise ValueError(f"Class multiplier length mismatch for {path}: expected {expected_dim}, got {len(values)}")
    if np.any(values <= 0) or not np.isfinite(values).all():
        raise ValueError(f"Class multipliers must be finite positive values: {path}")
    return values


def apply_class_multipliers(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    return normalize_probs(probs * multipliers.reshape(1, -1), probs.shape[1])


def main() -> None:
    args = parse_args()
    test_features = pd.read_csv("data/processed/prefix_test_features.csv")
    npz_path = Path("data/processed/lstm_dataset.npz")
    tabular_dir = Path("models/tabular_baseline")
    lstm_dir = Path("models/lstm")
    point_phase_dir = Path("models/point_phase")
    action_phase_dir = Path(args.action_phase_model_dir)
    server_stack_path = Path("models/server_stacking/serverGetPoint_extratrees.joblib")
    catboost_report_dir = Path("reports/catboost")
    class_multiplier_dir = Path(args.class_multiplier_dir)
    report_dir = Path("reports/final_blend")
    submission_path = Path(args.submission_path)

    action_tab = aligned_tabular_proba(tabular_dir / "actionId_extratrees.joblib", test_features, "actionId")
    point_tab = aligned_tabular_proba(tabular_dir / "pointId_extratrees.joblib", test_features, "pointId")
    server_tab = aligned_tabular_proba(tabular_dir / "serverGetPoint_extratrees.joblib", test_features, "serverGetPoint")
    action_lstm = predict_lstm_target("actionId", npz_path, lstm_dir)
    point_lstm = predict_lstm_target("pointId", npz_path, lstm_dir)
    server_lstm = predict_lstm_target("serverGetPoint", npz_path, lstm_dir)
    action_catboost = load_catboost_test_probs(catboost_report_dir / "actionId_test_proba.npy", 19, len(test_features))
    point_catboost = load_catboost_test_probs(catboost_report_dir / "pointId_test_proba.npy", 10, len(test_features))
    server_catboost = load_catboost_test_probs(catboost_report_dir / "serverGetPoint_test_proba.npy", 2, len(test_features))

    action_base = normalize_probs(FINAL_WEIGHTS["action"]["tabular"] * action_tab + FINAL_WEIGHTS["action"]["lstm"] * action_lstm, 19)
    action_final = normalize_probs((1.0 - FINAL_WEIGHTS["action_catboost"]) * action_base + FINAL_WEIGHTS["action_catboost"] * action_catboost, 19)
    action_phase = None
    if args.use_action_phase:
        action_phase = predict_action_phase(test_features, action_phase_dir)
        action_final = normalize_probs((1.0 - FINAL_WEIGHTS["action_phase"]) * action_final + FINAL_WEIGHTS["action_phase"] * action_phase, 19)
    action_final = apply_action_serve_mask(action_final, test_features)

    point_base = normalize_probs(FINAL_WEIGHTS["point_base"]["tabular"] * point_tab + FINAL_WEIGHTS["point_base"]["lstm"] * point_lstm, 10)
    point_phase = predict_point_phase(test_features, point_phase_dir)
    point_without_catboost = normalize_probs(FINAL_WEIGHTS["point_final"]["base"] * point_base + FINAL_WEIGHTS["point_final"]["phase"] * point_phase, 10)
    point_final = normalize_probs((1.0 - FINAL_WEIGHTS["point_catboost"]) * point_without_catboost + FINAL_WEIGHTS["point_catboost"] * point_catboost, 10)

    server_base = normalize_probs(FINAL_WEIGHTS["server_base"]["tabular"] * server_tab + FINAL_WEIGHTS["server_base"]["lstm"] * server_lstm, 2)
    action_ensemble = normalize_probs(0.70 * action_tab + 0.30 * action_lstm, 19)
    point_ensemble = normalize_probs(0.35 * point_tab + 0.65 * point_lstm, 10)
    point_phase_blend = normalize_probs(0.40 * point_phase + 0.60 * point_base, 10)
    server_stack_features = build_server_stacking_test_frame(
        test_features,
        action_tab,
        point_tab,
        action_lstm,
        point_lstm,
        action_ensemble,
        point_ensemble,
        point_phase,
        point_phase_blend,
    )
    server_bundle = joblib.load(server_stack_path)
    missing = [c for c in server_bundle["feature_columns"] if c not in server_stack_features.columns]
    if missing:
        raise ValueError(f"Missing server stacking features: {missing[:10]}")
    server_stack = aligned_bundle_predict(server_bundle, server_stack_features, 2)
    server_without_catboost = normalize_probs(FINAL_WEIGHTS["server_final"]["base"] * server_base + FINAL_WEIGHTS["server_final"]["stacking"] * server_stack, 2)
    server_final = normalize_probs((1.0 - FINAL_WEIGHTS["server_catboost"]) * server_without_catboost + FINAL_WEIGHTS["server_catboost"] * server_catboost, 2)

    action_final_adjusted = action_final.copy()
    point_final_adjusted = point_final.copy()
    class_multiplier_summary = {"enabled": args.use_class_multipliers}
    if args.use_class_multipliers:
        action_multipliers = load_class_multipliers(class_multiplier_dir / "actionId_multipliers.npy", 19)
        point_multipliers = load_class_multipliers(class_multiplier_dir / "pointId_multipliers.npy", 10)
        action_final_adjusted = apply_class_multipliers(action_final_adjusted, action_multipliers)
        point_final_adjusted = apply_class_multipliers(point_final_adjusted, point_multipliers)
        class_multiplier_summary.update(
            {
                "directory": str(class_multiplier_dir),
                "action_multipliers_path": str(class_multiplier_dir / "actionId_multipliers.npy"),
                "point_multipliers_path": str(class_multiplier_dir / "pointId_multipliers.npy"),
                "action_multiplier_mean": float(np.mean(action_multipliers)),
                "point_multiplier_mean": float(np.mean(point_multipliers)),
            }
        )

    server_output_values = server_final[:, 1] if args.server_output == "float" else server_final.argmax(axis=1).astype(int)

    submission = pd.DataFrame(
        {
            "rally_uid": test_features["sample_id"].astype(int),
            "actionId": action_final_adjusted.argmax(axis=1).astype(int),
            "pointId": point_final_adjusted.argmax(axis=1).astype(int),
            "serverGetPoint": server_output_values,
        }
    )
    validate_submission(submission, len(test_features), args.server_output)
    if submission["rally_uid"].nunique() != len(test_features):
        raise ValueError("Row count does not equal unique test rally count")

    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)

    save_probabilities(
        report_dir,
        [
            ("action_tabular_proba", action_tab),
            ("action_lstm_proba", action_lstm),
            ("action_catboost_proba", action_catboost),
            ("action_base_proba", action_base),
            ("action_phase_proba", action_phase if action_phase is not None else np.zeros_like(action_final)),
            ("action_final_proba", action_final),
            ("action_final_adjusted_proba", action_final_adjusted),
            ("point_tabular_proba", point_tab),
            ("point_lstm_proba", point_lstm),
            ("point_phase_proba", point_phase),
            ("point_catboost_proba", point_catboost),
            ("point_without_catboost_proba", point_without_catboost),
            ("point_final_proba", point_final),
            ("point_final_adjusted_proba", point_final_adjusted),
            ("server_tabular_proba", server_tab),
            ("server_lstm_proba", server_lstm),
            ("server_stacking_proba", server_stack),
            ("server_catboost_proba", server_catboost),
            ("server_without_catboost_proba", server_without_catboost),
            ("server_final_proba", server_final),
        ],
    )
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "submission_path": str(submission_path),
                "rows": int(len(submission)),
                "server_output_mode": args.server_output,
                "use_cross_target_stacking": args.use_cross_target_stacking,
                "use_action_phase": args.use_action_phase,
                "action_phase_model_dir": str(action_phase_dir),
                "class_multipliers": class_multiplier_summary,
                "weights": FINAL_WEIGHTS,
                "server_stack_probability_groups": [
                    "action_tabular",
                    "point_tabular",
                    "action_lstm",
                    "point_lstm",
                    "action_ensemble",
                    "point_ensemble",
                    "point_phase",
                    "point_phase_blend",
                ],
                "forbidden_server_features": sorted(FORBIDDEN_FEATURES),
                "sanity_checks": {
                    "row_count_matches_test": True,
                    "no_nans": True,
                    "valid_class_ranges": True,
                    "server_output_valid": True,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
