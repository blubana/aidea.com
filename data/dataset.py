from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


BASE_FEATURES = [
    "sex",
    "handId",
    "strengthId",
    "spinId",
    "actionId",
    "pointId",
    "positionId",
    "strikeId",
    "scoreSelf",
    "scoreOther",
    "strikeNumber",
]

SCORE_FEATURES = [
    *BASE_FEATURES[:-1],
    "scoreDiff",
    "scoreDiffBucket",
    "scoreTotal",
    "scorePhase",
    "isTied",
    "isLeading",
    "isDeuceLike",
    "isServeShot",
    "shotParity",
    "rallyLengthBucket",
    "strikeNumber",
]

ENHANCED_FEATURES = [
    *SCORE_FEATURES[:-1],
    "actionPointCombo",
    "spinStrengthCombo",
    "lastActionId",
    "lastPointId",
    "lastStrikeId",
    "lastPositionId",
    "strikeNumber",
]

SEMANTIC_FEATURES = [
    *ENHANCED_FEATURES[:-1],
    "currentHitterIsServer",
    "serverScore",
    "receiverScore",
    "serverScoreDiff",
    "serverScoreDiffBucket",
    "serverIsLeading",
    "serverPosition",
    "receiverPosition",
    "serveReceivePositionCombo",
    "shotPhase",
    "isThirdBall",
    "pointDepth",
    "pointLane",
    "lastPointDepth",
    "lastPointLane",
    "pointDepthTransition",
    "pointLaneTransition",
    "actionType",
    "lastActionType",
    "actionTypeTransition",
    "serveActionId",
    "serveSpinId",
    "servePointId",
    "servePointDepth",
    "servePointLane",
    "receiveActionId",
    "receivePointId",
    "receivePointDepth",
    "receivePointLane",
    "serveReceiveActionCombo",
    "serveReceivePointCombo",
    "serveSpinPointCombo",
    "strikeNumber",
]

FEATURE_SETS = {
    "base": BASE_FEATURES,
    "score": SCORE_FEATURES,
    "enhanced": ENHANCED_FEATURES,
    "semantic": SEMANTIC_FEATURES,
}
FEATURES = SEMANTIC_FEATURES

PAD_TOKEN = 0
UNK_TOKEN = 1
IGNORE_INDEX = -100


def resolve_features(feature_set: str = "enhanced") -> list[str]:
    if feature_set not in FEATURE_SETS:
        valid = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature_set={feature_set!r}; valid options: {valid}")
    return list(FEATURE_SETS[feature_set])


def _rally_order(df: pd.DataFrame) -> pd.Series:
    if "rally_uid" not in df.columns:
        return pd.Series(np.arange(len(df)), index=df.index)
    return df.groupby("rally_uid", sort=False).cumcount()


def _first_value(df: pd.DataFrame, col: str, default: int = 0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    if "rally_uid" not in df.columns:
        return df[col].fillna(default)
    return df.groupby("rally_uid", sort=False)[col].transform("first").fillna(default)


def _nth_value_causal(df: pd.DataFrame, col: str, n: int, default: int = 0) -> pd.Series:
    """Broadcast the nth rally value only to rows where that shot is already known."""
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    if "rally_uid" not in df.columns:
        value = df[col].iloc[n] if len(df) > n else default
        return pd.Series(np.where(np.arange(len(df)) >= n, value, default), index=df.index)
    order = _rally_order(df)
    nth = df.groupby("rally_uid", sort=False)[col].transform(lambda s: s.iloc[n] if len(s) > n else default)
    return pd.Series(np.where(order >= n, nth, default), index=df.index).fillna(default)


def _point_depth(point: pd.Series) -> pd.Series:
    point = point.fillna(0).astype(int)
    return pd.Series(
        np.select(
            [point.between(1, 3), point.between(4, 6), point.between(7, 9)],
            [1, 2, 3],
            default=0,
        ),
        index=point.index,
    )


def _point_lane(point: pd.Series) -> pd.Series:
    point = point.fillna(0).astype(int)
    return pd.Series(
        np.select(
            [point.isin([1, 4, 7]), point.isin([2, 5, 8]), point.isin([3, 6, 9])],
            [1, 2, 3],
            default=0,
        ),
        index=point.index,
    )


def _action_type(action: pd.Series) -> pd.Series:
    action = action.fillna(0).astype(int)
    return pd.Series(
        np.select(
            [
                action.between(1, 7),
                action.between(8, 11),
                action.between(12, 14),
                action.between(15, 18),
            ],
            [1, 2, 3, 4],
            default=0,
        ),
        index=action.index,
    )


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


def feature_values(df: pd.DataFrame, col: str) -> pd.Series:
    score_self = df["scoreSelf"].clip(0, 30)
    score_other = df["scoreOther"].clip(0, 30)
    score_diff = (score_self - score_other).clip(-30, 30)
    score_total = (score_self + score_other).clip(0, 60)
    server_player = _first_value(df, "gamePlayerId")
    current_is_server = (df["gamePlayerId"].fillna(-1).astype(int) == server_player.astype(int)).astype(int)
    server_score = pd.Series(np.where(current_is_server == 1, score_self, score_other), index=df.index).clip(0, 30)
    receiver_score = pd.Series(np.where(current_is_server == 1, score_other, score_self), index=df.index).clip(0, 30)
    server_score_diff = (server_score - receiver_score).clip(-30, 30)

    if col == "strikeNumber":
        return df[col].clip(0, 50)
    if col in ("scoreSelf", "scoreOther"):
        return df[col].clip(0, 30)
    if col == "scoreDiff":
        return score_diff + 30
    if col == "scoreDiffBucket":
        return pd.Series(
            np.select(
                [
                    score_diff <= -5,
                    score_diff.between(-4, -2),
                    score_diff == -1,
                    score_diff == 0,
                    score_diff == 1,
                    score_diff.between(2, 4),
                    score_diff >= 5,
                ],
                [0, 1, 2, 3, 4, 5, 6],
                default=3,
            ),
            index=df.index,
        )
    if col == "scoreTotal":
        return score_total
    if col == "scorePhase":
        return pd.Series(np.select([score_total <= 6, score_total <= 14], [0, 1], default=2), index=df.index)
    if col == "isTied":
        return (score_self == score_other).astype(int)
    if col == "isLeading":
        return (score_self > score_other).astype(int)
    if col == "isDeuceLike":
        return ((score_self >= 10) & (score_other >= 10)).astype(int)
    if col == "isServeShot":
        return (df["strikeNumber"] == 1).astype(int)
    if col == "shotParity":
        return (df["strikeNumber"].clip(0, 50) % 2).astype(int)
    if col == "rallyLengthBucket":
        strike_number = df["strikeNumber"].clip(0, 50)
        return pd.Series(
            np.select(
                [
                    strike_number == 1,
                    strike_number == 2,
                    strike_number == 3,
                    strike_number.between(4, 5),
                    strike_number.between(6, 8),
                    strike_number >= 9,
                ],
                [0, 1, 2, 3, 4, 5],
                default=0,
            ),
            index=df.index,
        )
    if col == "actionPointCombo":
        return df["actionId"].clip(0, 99) * 100 + df["pointId"].clip(0, 99)
    if col == "spinStrengthCombo":
        return df["spinId"].clip(0, 99) * 100 + df["strengthId"].clip(0, 99)
    if col == "currentHitterIsServer":
        return current_is_server
    if col == "serverScore":
        return server_score
    if col == "receiverScore":
        return receiver_score
    if col == "serverScoreDiff":
        return server_score_diff + 30
    if col == "serverScoreDiffBucket":
        return pd.Series(
            np.select(
                [
                    server_score_diff <= -5,
                    server_score_diff.between(-4, -2),
                    server_score_diff == -1,
                    server_score_diff == 0,
                    server_score_diff == 1,
                    server_score_diff.between(2, 4),
                    server_score_diff >= 5,
                ],
                [0, 1, 2, 3, 4, 5, 6],
                default=3,
            ),
            index=df.index,
        )
    if col == "serverIsLeading":
        return (server_score > receiver_score).astype(int)
    if col == "serverPosition":
        return _nth_value_causal(df, "positionId", 0).clip(0, 9)
    if col == "receiverPosition":
        return _nth_value_causal(df, "positionId", 1).clip(0, 9)
    if col == "serveReceivePositionCombo":
        return _nth_value_causal(df, "positionId", 0).clip(0, 9) * 10 + _nth_value_causal(df, "positionId", 1).clip(0, 9)
    if col == "shotPhase":
        strike_number = df["strikeNumber"].clip(0, 50)
        return pd.Series(
            np.select(
                [
                    strike_number == 1,
                    strike_number == 2,
                    strike_number == 3,
                    strike_number.between(4, 5),
                    strike_number >= 6,
                ],
                [1, 2, 3, 4, 5],
                default=0,
            ),
            index=df.index,
        )
    if col == "isThirdBall":
        return (df["strikeNumber"] == 3).astype(int)
    if col == "pointDepth":
        return _point_depth(df["pointId"])
    if col == "pointLane":
        return _point_lane(df["pointId"])
    if col == "lastPointDepth":
        last_point = df.groupby("rally_uid", sort=False)["pointId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["pointId"].shift(1).fillna(0)
        return _point_depth(last_point)
    if col == "lastPointLane":
        last_point = df.groupby("rally_uid", sort=False)["pointId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["pointId"].shift(1).fillna(0)
        return _point_lane(last_point)
    if col == "pointDepthTransition":
        last_point = df.groupby("rally_uid", sort=False)["pointId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["pointId"].shift(1).fillna(0)
        return _point_depth(last_point) * 10 + _point_depth(df["pointId"])
    if col == "pointLaneTransition":
        last_point = df.groupby("rally_uid", sort=False)["pointId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["pointId"].shift(1).fillna(0)
        return _point_lane(last_point) * 10 + _point_lane(df["pointId"])
    if col == "actionType":
        return _action_type(df["actionId"])
    if col == "lastActionType":
        last_action = df.groupby("rally_uid", sort=False)["actionId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["actionId"].shift(1).fillna(0)
        return _action_type(last_action)
    if col == "actionTypeTransition":
        last_action = df.groupby("rally_uid", sort=False)["actionId"].shift(1).fillna(0) if "rally_uid" in df.columns else df["actionId"].shift(1).fillna(0)
        return _action_type(last_action) * 10 + _action_type(df["actionId"])
    if col == "serveActionId":
        return _nth_value_causal(df, "actionId", 0).clip(0, 99)
    if col == "serveSpinId":
        return _nth_value_causal(df, "spinId", 0).clip(0, 99)
    if col == "servePointId":
        return _nth_value_causal(df, "pointId", 0).clip(0, 99)
    if col == "servePointDepth":
        return _point_depth(_nth_value_causal(df, "pointId", 0))
    if col == "servePointLane":
        return _point_lane(_nth_value_causal(df, "pointId", 0))
    if col == "receiveActionId":
        return _nth_value_causal(df, "actionId", 1).clip(0, 99)
    if col == "receivePointId":
        return _nth_value_causal(df, "pointId", 1).clip(0, 99)
    if col == "receivePointDepth":
        return _point_depth(_nth_value_causal(df, "pointId", 1))
    if col == "receivePointLane":
        return _point_lane(_nth_value_causal(df, "pointId", 1))
    if col == "serveReceiveActionCombo":
        return _nth_value_causal(df, "actionId", 0).clip(0, 99) * 100 + _nth_value_causal(df, "actionId", 1).clip(0, 99)
    if col == "serveReceivePointCombo":
        return _nth_value_causal(df, "pointId", 0).clip(0, 99) * 100 + _nth_value_causal(df, "pointId", 1).clip(0, 99)
    if col == "serveSpinPointCombo":
        return _nth_value_causal(df, "spinId", 0).clip(0, 99) * 100 + _nth_value_causal(df, "pointId", 0).clip(0, 99)
    if col.startswith("last"):
        raw_col = col[4].lower() + col[5:]
        if raw_col not in df.columns:
            raise KeyError(f"Cannot derive {col}: missing source column {raw_col!r}")
        if "rally_uid" in df.columns:
            return df.groupby("rally_uid", sort=False)[raw_col].shift(1).fillna(0).astype(int)
        return df[raw_col].shift(1).fillna(0).astype(int)
    return df[col]

def encode_features(df: pd.DataFrame, cats: dict, features: list[str] | None = None) -> np.ndarray:
    features = FEATURES if features is None else features
    encoded_cols: list[np.ndarray] = []
    for col in features:
        vals = feature_values(df, col)
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
    features: list[str] | None = None,
    last_prefix_only: bool = False,
    target_from_end: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = FEATURES if features is None else features
    num_features = len(features)
    if len(df) == 0:
        return (
            np.zeros((0, maxlen, num_features), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    df = sort_rallies(df)
    feat = encode_features(df, cats, features)
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

        if last_prefix_only:
            target_t = n - max(1, target_from_end)
            if target_t < 1:
                continue
            target_steps = [target_t]
        else:
            target_steps = range(1, n)
        for t in target_steps:
            l = t
            seq = rally_feat[:t]
            padded = np.zeros((maxlen, num_features), dtype=np.int64)
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
    features: list[str] | None = None,
) -> tuple[RallyDataset, RallyDataset, int, int]:
    features = FEATURES if features is None else features
    train_df = sort_rallies(train_df)
    val_df = sort_rallies(val_df)
    maxlen = _max_rally_len(train_df, val_df)
    num_features = len(features)

    train_X, train_yA, train_yP, train_yR, train_L = _build_split_samples(
        train_df, cats, act_id2idx, pt_id2idx, maxlen, features
    )
    val_X, val_yA, val_yP, val_yR, val_L = _build_split_samples(
        val_df, cats, act_id2idx, pt_id2idx, maxlen, features
    )

    train_ds = RallyDataset(train_X, train_yA, train_yP, train_yR, train_L)
    val_ds = RallyDataset(val_X, val_yA, val_yP, val_yR, val_L)
    return train_ds, val_ds, maxlen, num_features


def build_submission_like_dataset(
    df: pd.DataFrame,
    cats: dict,
    act_id2idx: dict,
    pt_id2idx: dict,
    maxlen: int,
    features: list[str] | None = None,
    target_from_end: int = 2,
) -> RallyDataset:
    X, yA, yP, yR, L = _build_split_samples(
        df,
        cats,
        act_id2idx,
        pt_id2idx,
        maxlen,
        features,
        last_prefix_only=True,
        target_from_end=target_from_end,
    )
    return RallyDataset(X, yA, yP, yR, L)
