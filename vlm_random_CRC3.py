"""Uniform-random cold-start baseline for CRC100K.

For a given seed this script uses the same target-class configuration and the
same uniformly sampled candidate pool as the other two baseline scripts.
It then selects the final query uniformly at random without using a visual or
language model.
"""

import argparse
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/root/gpufree-data/OpenPath-main")
DEFAULT_DATA_CSV = PROJECT_ROOT / "al_file" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "al_file"
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_DIR / "experiment_results_random.log"
NUM_CLASSES = 9


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_id_classes(seed: int):
    # RandomState reproduces np.random.seed(seed) + np.random.choice(...),
    # which is the protocol used by the original CRC script.
    rng = np.random.RandomState(seed)
    id_cls = sorted(rng.choice(np.arange(NUM_CLASSES), size=3, replace=False).tolist())
    ood_cls = sorted([c for c in range(NUM_CLASSES) if c not in id_cls])
    return id_cls, ood_cls


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"{csv_path} must contain at least image-path and label columns")
    df = df.iloc[:, :2].copy()
    df.columns = ["img", "cls_label"]
    df["img"] = df["img"].astype(str)
    df["cls_label"] = df["cls_label"].astype(int)
    return df


def sample_candidate_pool(df: pd.DataFrame, pool_size: int, pool_seed: int) -> pd.DataFrame:
    if pool_size <= 0 or pool_size >= len(df):
        return df.sample(frac=1.0, random_state=pool_seed).reset_index(drop=True)
    return df.sample(n=pool_size, replace=False, random_state=pool_seed).reset_index(drop=True)


def count_id(df: pd.DataFrame, id_cls) -> int:
    return int(df["cls_label"].isin(id_cls).sum())


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"


def append_report(log_path: Path, report: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(report)


def get_args():
    parser = argparse.ArgumentParser(description="CRC100K uniform-random cold-start baseline")
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--pool_size", type=int, default=300,
                        help="Shared random candidate-pool size; <=0 uses the full pool")
    parser.add_argument("--pool_seed", type=int, default=2026,
                        help="Fixed pool seed used by vlm_new_random_CRC3.py")
    parser.add_argument("--query_size", type=int, default=50)
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_file", type=Path, default=DEFAULT_LOG_FILE)
    return parser.parse_args()


def main() -> None:
    total_start = time.perf_counter()
    args = get_args()
    seed_everything(args.seed)
    id_cls, ood_cls = choose_id_classes(args.seed)

    data_start = time.perf_counter()
    full_df = load_dataframe(args.data_csv)
    data_seconds = time.perf_counter() - data_start

    pool_start = time.perf_counter()
    pool_df = sample_candidate_pool(full_df, args.pool_size, args.pool_seed)
    pool_seconds = time.perf_counter() - pool_start

    if len(pool_df) < args.query_size:
        raise ValueError(
            f"Candidate pool contains {len(pool_df)} samples, fewer than query_size={args.query_size}"
        )

    selection_start = time.perf_counter()
    # A separate deterministic selection seed prevents accidental dependence
    # between pool construction and final random querying.
    selection_seed = args.seed
    selected_df = pool_df.sample(
        n=args.query_size,
        replace=False,
        random_state=selection_seed,
    ).reset_index(drop=True)
    selection_seconds = time.perf_counter() - selection_start

    pool_id = count_id(pool_df, id_cls)
    selected_id = count_id(selected_df, id_cls)
    pool_qp = 100.0 * pool_id / len(pool_df)
    final_qp = 100.0 * selected_id / len(selected_df)

    save_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"query_random_seed{args.seed}.csv"
    selected_df.rename(columns={"cls_label": "label"}).to_csv(output_csv, index=False)
    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - total_start

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    report = f"""
================================================================================
Method: Random
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
Seed: {args.seed}
ID classes: {id_cls}
OOD classes: {ood_cls}
Final query size: {len(selected_df)}
Pool ID/OOD: {pool_id}/{len(pool_df) - pool_id}
Pool QP: {pool_qp:.2f}%
Final query ID/OOD: {selected_id}/{len(selected_df) - selected_id}
Final query QP: {final_qp:.2f}%
Total runtime: {format_duration(total_seconds)}
================================================================================
"""
    print(report)
    append_report(args.log_file, report)
    print(f"Results appended to: {args.log_file}")


if __name__ == "__main__":
    main()
