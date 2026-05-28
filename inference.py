from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from tqdm.auto import tqdm

from data.dataset import FEATURES, encode_features, sort_rallies
from model.rnn import RallyRNN
from model.transformer import RallyTransformer
from train_tabular import flatten_prefix
from train import last_logits


@dataclass
class EnsembleMember:
    path: Path
    kind: str
    model: object
    cats: dict[str, list[int]]
    features: list[str]
    action_classes: list[int]
    point_classes: list[int]
    maxlen: int = 0
    window: int = 0
    rally_model: object | None = None
    action_model: object | None = None
    point_model: object | None = None


def banner(console: Console) -> None:
    console.print("[bold magenta]+--------------------------------------------+[/]")
    console.print("[bold magenta]|  AI CUP Rally RoPE Transformer Inference  |[/]")
    console.print("[bold magenta]+--------------------------------------------+[/]")


def pad_sequence(feat: np.ndarray, maxlen: int) -> tuple[np.ndarray, int]:
    if feat.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape={feat.shape}")
    length = min(len(feat), maxlen)
    out = np.zeros((maxlen, feat.shape[1]), dtype=np.int64)
    if length > 0:
        out[:length] = feat[-length:]
    return out, max(1, length)


def default_out_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"submission_transformer_{stamp}.csv"


def resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.checkpoint]
    if args.checkpoint_dir:
        root = Path(args.checkpoint_dir)
        matches = root.rglob(args.checkpoint_glob) if args.recursive else root.glob(args.checkpoint_glob)
        paths.extend(sorted(p for p in matches if p.is_file()))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("Provide --checkpoint or --checkpoint-dir with matching checkpoint files.")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    return paths


def model_kwargs_for_checkpoint(meta: dict, state: dict[str, torch.Tensor]) -> dict:
    kwargs = dict(meta["model_kwargs"])
    model_type = meta.get("model_type", "transformer")
    if model_type == "transformer":
        keep = {"d_model", "nhead", "num_encoder_layers", "dim_feedforward", "dropout", "emb_dim", "rally_pool"}
        kwargs = {k: v for k, v in kwargs.items() if k in keep}
        if "rally_pool" not in kwargs:
            kwargs["rally_pool"] = "mean_linear" if "rally_head.weight" in state else "last_mean_mlp"
    elif model_type in {"gru", "lstm"}:
        keep = {"d_model", "num_encoder_layers", "dropout", "emb_dim", "bidirectional"}
        kwargs = {k: v for k, v in kwargs.items() if k in keep}
    return kwargs


def load_neural_member(path: Path, device: torch.device) -> EnsembleMember:
    ckpt = torch.load(path, map_location=device)
    meta = ckpt["metadata"]
    state = ckpt["model_state"]
    action_classes = [int(v) for v in meta["action_classes"]]
    point_classes = [int(v) for v in meta["point_classes"]]
    model_type = meta.get("model_type", "transformer")
    kwargs = model_kwargs_for_checkpoint(meta, state)
    if model_type == "transformer":
        model = RallyTransformer(
            num_tokens_per_feature=meta["num_tokens_per_feature"],
            n_act=len(action_classes),
            n_pt=len(point_classes),
            max_seq_len=int(meta["maxlen"]),
            **kwargs,
        ).to(device)
    elif model_type in {"gru", "lstm"}:
        model = RallyRNN(
            num_tokens_per_feature=meta["num_tokens_per_feature"],
            n_act=len(action_classes),
            n_pt=len(point_classes),
            rnn_type=model_type,
            **kwargs,
        ).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint model_type={model_type!r} in {path}")
    model.load_state_dict(state)
    model.eval()
    return EnsembleMember(
        path=path,
        kind=model_type,
        model=model,
        cats={k: list(v) for k, v in meta["cats"].items()},
        features=list(meta.get("features", FEATURES)),
        action_classes=action_classes,
        point_classes=point_classes,
        maxlen=int(meta["maxlen"]),
    )


def load_tabular_member(path: Path) -> EnsembleMember:
    with path.open("rb") as f:
        artifact = pickle.load(f)
    if artifact.get("artifact_type") != "tabular_ensemble":
        raise ValueError(f"Not a tabular ensemble artifact: {path}")
    meta = artifact["metadata"]
    models = artifact["models"]
    return EnsembleMember(
        path=path,
        kind=f"tabular:{artifact.get('backend', 'unknown')}",
        model=models,
        cats={k: list(v) for k, v in meta["cats"].items()},
        features=list(meta.get("features", FEATURES)),
        action_classes=[int(v) for v in meta["action_classes"]],
        point_classes=[int(v) for v in meta["point_classes"]],
        window=int(meta["window"]),
        action_model=models["action"],
        point_model=models["point"],
        rally_model=models["rally"],
    )


def load_member(path: Path, device: torch.device) -> EnsembleMember:
    if path.suffix.lower() == ".pkl":
        return load_tabular_member(path)
    return load_neural_member(path, device)


def validate_class_alignment(members: list[EnsembleMember]) -> tuple[list[int], list[int]]:
    action_classes = sorted({v for member in members for v in member.action_classes})
    point_classes = sorted({v for member in members for v in member.point_classes})
    return action_classes, point_classes


def print_ensemble_table(console: Console, members: list[EnsembleMember]) -> None:
    table = Table(title="Inference ensemble")
    table.add_column("#", justify="right")
    table.add_column("kind")
    table.add_column("checkpoint")
    table.add_column("features", justify="right")
    table.add_column("shape", justify="right")
    for i, member in enumerate(members, start=1):
        shape = f"T={member.maxlen}" if member.maxlen else f"W={member.window}"
        table.add_row(str(i), member.kind, str(member.path), str(len(member.features)), shape)
    console.print(table)


def model_classes(model) -> np.ndarray:
    return np.asarray(getattr(model, "classes_", []), dtype=np.int64)


def aligned_predict_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = model_classes(model)
    if proba.ndim == 1:
        proba = np.stack([1.0 - proba, proba], axis=1)
    out = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    if len(classes) == proba.shape[1]:
        for src_idx, cls_idx in enumerate(classes):
            if 0 <= int(cls_idx) < n_classes:
                out[:, int(cls_idx)] = proba[:, src_idx]
    else:
        out[:, : min(n_classes, proba.shape[1])] = proba[:, : min(n_classes, proba.shape[1])]
    return out


def aligned_positive_proba(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba.astype(np.float32)
    classes = model_classes(model)
    if len(classes) == proba.shape[1] and 1 in classes:
        return proba[:, int(np.where(classes == 1)[0][0])].astype(np.float32)
    return proba[:, -1].astype(np.float32)


def align_member_probs(probs: np.ndarray, member_classes: list[int], global_classes: list[int]) -> np.ndarray:
    out = np.zeros((len(global_classes),), dtype=np.float32)
    global_index = {cls: i for i, cls in enumerate(global_classes)}
    for src_idx, cls in enumerate(member_classes):
        if cls in global_index and src_idx < len(probs):
            out[global_index[cls]] = probs[src_idx]
    total = float(out.sum())
    if total > 0:
        out /= total
    return out


def member_predict_proba(
    group: pd.DataFrame,
    member: EnsembleMember,
    device: torch.device,
    action_classes: list[int],
    point_classes: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    feat = encode_features(group, member.cats, member.features)
    if member.kind.startswith("tabular"):
        x = flatten_prefix(feat, len(feat), member.window)[None, :]
        pa_local = aligned_predict_proba(member.action_model, x, len(member.action_classes)).squeeze(0)
        pp_local = aligned_predict_proba(member.point_model, x, len(member.point_classes)).squeeze(0)
        pa = align_member_probs(pa_local, member.action_classes, action_classes)
        pp = align_member_probs(pp_local, member.point_classes, point_classes)
        pr = float(aligned_positive_proba(member.rally_model, x)[0])
        return pa, pp, pr

    padded, length = pad_sequence(feat, member.maxlen)
    X = torch.tensor(padded[None, ...], dtype=torch.long, device=device)
    L = torch.tensor([length], dtype=torch.long, device=device)
    with torch.no_grad():
        la_seq, lp_seq, lr = member.model(X, L)
        pa_local = F.softmax(last_logits(la_seq, L), dim=-1).squeeze(0).detach().cpu().numpy()
        pp_local = F.softmax(last_logits(lp_seq, L), dim=-1).squeeze(0).detach().cpu().numpy()
        pa = align_member_probs(pa_local, member.action_classes, action_classes)
        pp = align_member_probs(pp_local, member.point_classes, point_classes)
        pr = float(torch.sigmoid(lr).item())
    return pa, pp, pr


def predict_group(
    group: pd.DataFrame,
    members: list[EnsembleMember],
    device: torch.device,
    action_classes: list[int],
    point_classes: list[int],
) -> tuple[int, int, float]:
    action_prob_sum: np.ndarray | None = None
    point_prob_sum: np.ndarray | None = None
    rally_prob_sum = 0.0

    for member in members:
        pa, pp, pr = member_predict_proba(group, member, device, action_classes, point_classes)
        action_prob_sum = pa if action_prob_sum is None else action_prob_sum + pa
        point_prob_sum = pp if point_prob_sum is None else point_prob_sum + pp
        rally_prob_sum += pr

    assert action_prob_sum is not None
    assert point_prob_sum is not None
    a_idx = int(action_prob_sum.argmax(-1).item())
    p_idx = int(point_prob_sum.argmax(-1).item())
    return a_idx, p_idx, rally_prob_sum / len(members)


def main(args: argparse.Namespace) -> None:
    console = Console()
    banner(console)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = resolve_checkpoints(args)
    members = [load_member(path, device) for path in checkpoint_paths]
    action_classes, point_classes = validate_class_alignment(members)
    print_ensemble_table(console, members)

    data_dir = Path(args.data_dir)
    test_csv = Path(args.test) if args.test else data_dir / "test_new.csv"
    sample_csv = Path(args.sample) if args.sample else data_dir / "sample_submission.csv"
    test_df = sort_rallies(pd.read_csv(test_csv))
    if args.limit_rallies > 0:
        keep = test_df["rally_uid"].drop_duplicates().head(args.limit_rallies)
        test_df = test_df[test_df["rally_uid"].isin(keep)].copy()

    rows: list[dict] = []
    with torch.no_grad():
        groups = test_df.groupby("rally_uid", sort=False)
        for rid, group in tqdm(groups, total=groups.ngroups, desc="predict rallies", ascii=True):
            a_idx, p_idx, rally_prob = predict_group(group, members, device, action_classes, point_classes)
            rows.append(
                {
                    "rally_uid": int(rid),
                    "actionId": int(action_classes[a_idx]),
                    "pointId": int(point_classes[p_idx]),
                    "serverGetPoint": rally_prob,
                }
            )

    pred_df = pd.DataFrame(rows).sort_values("rally_uid")
    if sample_csv.exists():
        sample = pd.read_csv(sample_csv)
        if "rally_uid" in sample.columns and len(sample) > 0:
            out = sample[["rally_uid"]].merge(pred_df, on="rally_uid", how="left")
        else:
            out = pred_df
    else:
        out = pred_df
    out = out[["rally_uid", "actionId", "pointId", "serverGetPoint"]]

    out_path = Path(args.out) if args.out else default_out_path(Path(args.out_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    console.print(f"[green]Saved submission:[/] {out_path}")
    console.print(out.head().to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="*", default=[], help="One or more checkpoint files to ensemble.")
    parser.add_argument("--checkpoint-dir", default="", help="Directory containing checkpoints when --checkpoint is omitted.")
    parser.add_argument("--checkpoint-glob", default="best_transformer_*.pt")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--test", default="")
    parser.add_argument("--sample", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-rallies", type=int, default=0, help="Debug limit for inference smoke tests; 0 uses all rallies.")
    main(parser.parse_args())
