"""Build sequence + manual-feature dataset for first LSTM models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CAT_STEP_COLS = [
    "sex",
    "gamePlayerId",
    "gamePlayerOtherId",
    "strikeId",
    "handId",
    "strengthId",
    "spinId",
    "pointId",
    "actionId",
    "positionId",
]

NUM_STEP_COLS = [
    "strikeNumber",
    "scoreSelf",
    "scoreOther",
    "score_diff",
    "score_sum",
    "is_odd_stroke",
    "is_serve_stroke",
    "is_receive_stroke",
    "is_third_ball",
    "is_rally_phase",
]

META_EXCLUDE = {
    "sample_id",
    "rally_uid",
    "match",
    "source_rally_len",
    "target_strikeNumber",
    "sample_weight",
    "label_actionId",
    "label_pointId",
    "label_serverGetPoint",
}


def _prepare_raw(df: pd.DataFrame, is_test: bool) -> pd.DataFrame:
    out = df.copy()
    for col in CAT_STEP_COLS + ["match", "rally_uid"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1).astype(np.int64)
    for col in ["strikeNumber", "scoreSelf", "scoreOther"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(np.int64)
    out["score_diff"] = out["scoreSelf"] - out["scoreOther"]
    out["score_sum"] = out["scoreSelf"] + out["scoreOther"]
    out["is_odd_stroke"] = (out["strikeNumber"] % 2).astype(np.float32)
    out["is_serve_stroke"] = (out["strikeNumber"] == 1).astype(np.float32)
    out["is_receive_stroke"] = (out["strikeNumber"] == 2).astype(np.float32)
    out["is_third_ball"] = (out["strikeNumber"] == 3).astype(np.float32)
    out["is_rally_phase"] = (out["strikeNumber"] >= 4).astype(np.float32)
    if not is_test and "serverGetPoint" in out.columns:
        out["serverGetPoint"] = pd.to_numeric(out["serverGetPoint"], errors="coerce").fillna(-1).astype(np.int64)
    return out.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True)


def _build_group_lookup(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(k): g.reset_index(drop=True) for k, g in df.groupby("rally_uid", sort=False)}


def _parse_train_sample_id(sample_id: str) -> tuple[int, int]:
    rally_uid, k = sample_id.rsplit("_", 1)
    return int(rally_uid), int(k)


def _manual_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in META_EXCLUDE:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def _encode_cat(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.int64)
    out = np.where(values >= 0, values + 1, 0)
    return out.astype(np.int64)


def _make_sequence(rows: pd.DataFrame, prefix_len: int, max_len: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a padded sequence from an observed prefix.

    If the prefix is longer than max_len, the most recent max_len strokes are
    kept. Valid timesteps are placed at the beginning and padding at the end so
    pack_padded_sequence(lengths=...) reads the real strokes, not padding.
    """

    seq = rows.iloc[:prefix_len]
    actual_len = min(len(seq), max_len)
    seq = seq.iloc[-actual_len:]
    x_cat = np.zeros((max_len, len(CAT_STEP_COLS)), dtype=np.int64)
    x_num = np.zeros((max_len, len(NUM_STEP_COLS)), dtype=np.float32)
    if actual_len:
        x_cat[:actual_len] = _encode_cat(seq[CAT_STEP_COLS].to_numpy())
        x_num[:actual_len] = seq[NUM_STEP_COLS].to_numpy(dtype=np.float32)
    return x_cat, x_num, actual_len


def build_dataset(dataset_dir: Path, processed_dir: Path, output_path: Path, max_len: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_raw = _prepare_raw(pd.read_csv(dataset_dir / "train.csv"), is_test=False)
    test_raw = _prepare_raw(pd.read_csv(dataset_dir / "test_new.csv"), is_test=True)
    train_prefix = pd.read_csv(processed_dir / "prefix_train_features.csv")
    test_prefix = pd.read_csv(processed_dir / "prefix_test_features.csv")

    manual_cols = _manual_feature_columns(train_prefix)
    train_groups = _build_group_lookup(train_raw)
    test_groups = _build_group_lookup(test_raw)

    n_train = len(train_prefix)
    n_test = len(test_prefix)
    x_cat_seq = np.zeros((n_train, max_len, len(CAT_STEP_COLS)), dtype=np.int64)
    x_num_seq = np.zeros((n_train, max_len, len(NUM_STEP_COLS)), dtype=np.float32)
    lengths = np.zeros(n_train, dtype=np.int64)
    test_x_cat_seq = np.zeros((n_test, max_len, len(CAT_STEP_COLS)), dtype=np.int64)
    test_x_num_seq = np.zeros((n_test, max_len, len(NUM_STEP_COLS)), dtype=np.float32)
    test_lengths = np.zeros(n_test, dtype=np.int64)

    for i, sample_id in enumerate(train_prefix["sample_id"].astype(str)):
        rally_uid, prefix_len = _parse_train_sample_id(sample_id)
        x_cat_seq[i], x_num_seq[i], lengths[i] = _make_sequence(train_groups[rally_uid], prefix_len, max_len)

    for i, rally_uid in enumerate(test_prefix["sample_id"].astype(int)):
        test_x_cat_seq[i], test_x_num_seq[i], test_lengths[i] = _make_sequence(
            test_groups[int(rally_uid)], len(test_groups[int(rally_uid)]), max_len
        )

    np.savez_compressed(
        output_path,
        X_cat_seq=x_cat_seq,
        X_num_seq=x_num_seq,
        lengths=lengths,
        X_manual=train_prefix[manual_cols].to_numpy(dtype=np.float32),
        manual_feature_names=np.asarray(manual_cols, dtype=object),
        cat_feature_names=np.asarray(CAT_STEP_COLS, dtype=object),
        num_feature_names=np.asarray(NUM_STEP_COLS, dtype=object),
        y_action=train_prefix["label_actionId"].to_numpy(dtype=np.int64),
        y_point=train_prefix["label_pointId"].to_numpy(dtype=np.int64),
        y_server=train_prefix["label_serverGetPoint"].to_numpy(dtype=np.float32),
        sample_weight=train_prefix.get("sample_weight", pd.Series(np.ones(n_train))).to_numpy(dtype=np.float32),
        groups_match=train_prefix["match"].to_numpy(dtype=np.int64),
        groups_rally_uid=train_prefix["rally_uid"].to_numpy(dtype=np.int64),
        test_X_cat_seq=test_x_cat_seq,
        test_X_num_seq=test_x_num_seq,
        test_lengths=test_lengths,
        test_X_manual=test_prefix[manual_cols].to_numpy(dtype=np.float32),
        test_rally_uid=test_prefix["sample_id"].to_numpy(dtype=np.int64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/lstm_dataset.npz")
    parser.add_argument("--max-len", type=int, default=12)
    args = parser.parse_args()
    build_dataset(Path(args.dataset_dir), Path(args.processed_dir), Path(args.output), args.max_len)


if __name__ == "__main__":
    main()
