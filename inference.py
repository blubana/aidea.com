from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from tqdm.auto import tqdm

from data.dataset import FEATURES, encode_features, sort_rallies
from model.transformer import RallyTransformer
from train import last_logits


def banner(console: Console) -> None:
    console.print("[bold magenta]+--------------------------------------------+[/]")
    console.print("[bold magenta]|  AI CUP Rally RoPE Transformer Inference  |[/]")
    console.print("[bold magenta]+--------------------------------------------+[/]")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)


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


def main(args: argparse.Namespace) -> None:
    console = Console()
    banner(console)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    meta = ckpt["metadata"]

    action_classes = [int(v) for v in meta["action_classes"]]
    point_classes = [int(v) for v in meta["point_classes"]]
    model_kwargs = meta["model_kwargs"]
    model = RallyTransformer(
        num_tokens_per_feature=meta["num_tokens_per_feature"],
        n_act=len(action_classes),
        n_pt=len(point_classes),
        max_seq_len=int(meta["maxlen"]),
        **model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_dir = Path(args.data_dir)
    test_csv = Path(args.test) if args.test else data_dir / "test_new.csv"
    sample_csv = Path(args.sample) if args.sample else data_dir / "sample_submission.csv"
    test_df = sort_rallies(pd.read_csv(test_csv))
    cats = {k: list(v) for k, v in meta["cats"].items()}
    maxlen = int(meta["maxlen"])

    rows: list[dict] = []
    with torch.no_grad():
        groups = test_df.groupby("rally_uid", sort=False)
        for rid, g in tqdm(groups, total=groups.ngroups, desc="predict rallies", ascii=True):
            feat = encode_features(g, cats)
            padded, length = pad_sequence(feat, maxlen)
            X = torch.tensor(padded[None, ...], dtype=torch.long, device=device)
            L = torch.tensor([length], dtype=torch.long, device=device)
            la_seq, lp_seq, lr = model(X, L)
            la = last_logits(la_seq, L)
            lp = last_logits(lp_seq, L)
            a_idx = int(la.argmax(-1).item())
            p_idx = int(lp.argmax(-1).item())
            rows.append(
                {
                    "rally_uid": int(rid),
                    "actionId": int(action_classes[a_idx]),
                    "pointId": int(point_classes[p_idx]),
                    "serverGetPoint": float(torch.sigmoid(lr).item()),
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="dataset/AI CUP競賽資料集")
    parser.add_argument("--test", default="")
    parser.add_argument("--sample", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-dir", default="submissions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())
