from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


FEATURES = [
    "sex",
    "handId",
    "strengthId",
    "spinId",
    "positionId",
    "strikeId",
    "scoreSelf",
    "scoreOther",
    "strikeNumber",
]

PAD_TOKEN = 0
UNK_TOKEN = 1
IGNORE_INDEX = -100


class RallyDataset(Dataset):
    def __init__(self, X, yA, yP, yR, L):
        self.X = np.asarray(X, dtype=np.int64)
        self.yA = np.asarray(yA, dtype=np.int64)
        self.yP = np.asarray(yP, dtype=np.int64)
        self.yR = np.asarray(yR, dtype=np.float32)
        self.L = np.asarray(L, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.yA[idx], self.yP[idx], self.yR[idx], self.L[idx]


def encode_features(df: pd.DataFrame, cats: dict) -> np.ndarray:
    encoded_cols: list[np.ndarray] = []
    for col in FEATURES:
        if col == "strikeNumber":
            vals = df[col].clip(0, 50)
        elif col in ("scoreSelf", "scoreOther"):
            vals = df[col].clip(0, 30)
        else:
            vals = df[col]

        raw_codes = pd.Categorical(vals, categories=cats[col]).codes
        codes = np.where(raw_codes < 0, UNK_TOKEN, raw_codes + 2)
        encoded_cols.append(codes.astype(np.int64, copy=False))

    return np.stack(encoded_cols, axis=1)


def _max_rally_len(train_df: pd.DataFrame, val_df: pd.DataFrame) -> int:
    train_max = int(train_df.groupby("rally_uid", sort=False).size().max()) if len(train_df) else 0
    val_max = int(val_df.groupby("rally_uid", sort=False).size().max()) if len(val_df) else 0
    return max(train_max, val_max)


def sort_rallies(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by rally and shot order.

    Dense sliding-window samples rely on chronological order. Keeping this in
    the dataset builder avoids accidental future/previous shot swaps when a
    caller passes an unsorted frame.
    """
    sort_cols = [c for c in ("rally_uid", "strikeNumber") if c in df.columns]
    if not sort_cols:
        return df.reset_index(drop=True)
    return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _build_split_samples(
    df: pd.DataFrame,
    cats: dict,
    act_id2idx: dict,
    pt_id2idx: dict,
    maxlen: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(df) == 0:
        num_features = len(FEATURES)
        return (
            np.zeros((0, maxlen, num_features), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    df = sort_rallies(df)
    feat = encode_features(df, cats)
    action_raw = df["actionId"].to_numpy()
    point_raw = df["pointId"].to_numpy()
    rally_label = df["serverGetPoint"].to_numpy(dtype=np.float32)

    X_list: list[np.ndarray] = []
    yA_list: list[int] = []
    yP_list: list[int] = []
    yR_list: list[float] = []
    L_list: list[int] = []

    for _, idx in df.groupby("rally_uid", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        n = int(idx_arr.shape[0])
        if n < 2:
            continue

        rally_feat = feat[idx_arr]
        rally_actions = action_raw[idx_arr]
        rally_points = point_raw[idx_arr]
        rally_server_get_point = float(rally_label[idx_arr[0]])

        for t in range(1, n):
            l = t
            seq = rally_feat[:t]
            padded = np.zeros((maxlen, len(FEATURES)), dtype=np.int64)
            padded[:l, :] = seq

            next_action = act_id2idx.get(rally_actions[t], IGNORE_INDEX)
            next_point = pt_id2idx.get(rally_points[t], IGNORE_INDEX)

            X_list.append(padded)
            yA_list.append(int(next_action))
            yP_list.append(int(next_point))
            yR_list.append(rally_server_get_point)
            L_list.append(l)

    return (
        np.asarray(X_list, dtype=np.int64),
        np.asarray(yA_list, dtype=np.int64),
        np.asarray(yP_list, dtype=np.int64),
        np.asarray(yR_list, dtype=np.float32),
        np.asarray(L_list, dtype=np.int64),
    )


def build_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cats: dict,
    act_id2idx: dict,
    pt_id2idx: dict,
) -> tuple[RallyDataset, RallyDataset, int, int]:
    train_df = sort_rallies(train_df)
    val_df = sort_rallies(val_df)
    maxlen = _max_rally_len(train_df, val_df)
    num_features = len(FEATURES)

    train_X, train_yA, train_yP, train_yR, train_L = _build_split_samples(
        train_df, cats, act_id2idx, pt_id2idx, maxlen
    )
    val_X, val_yA, val_yP, val_yR, val_L = _build_split_samples(
        val_df, cats, act_id2idx, pt_id2idx, maxlen
    )

    train_ds = RallyDataset(train_X, train_yA, train_yP, train_yR, train_L)
    val_ds = RallyDataset(val_X, val_yA, val_yP, val_yR, val_L)
    return train_ds, val_ds, maxlen, num_features
