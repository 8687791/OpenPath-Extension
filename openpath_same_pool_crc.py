"""OpenPath-same-pool baseline for CRC100K.

The original BiomedCLIP + PLIP zero-shot screening and K-Means++ selection are
run on the same uniformly sampled candidate pool used by the other baselines.
"""

import argparse
import os
import platform
import random
import time
from pathlib import Path

# These variables must be set before Hugging Face/OpenCLIP imports.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import pandas as pd
import torch
from open_clip import create_model_from_pretrained, get_tokenizer
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

from dataset.alb_dataset2 import Tumor_dataset_val_cls, get_loader


PROJECT_ROOT = Path("/root/gpufree-data/OpenPath-main")
DEFAULT_DATA_CSV = PROJECT_ROOT / "al_file" / "train.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "al_file"
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_DIR / "experiment_results_random.log"
NUM_CLASSES = 9

BASE_PROMPTS = [
    "An H&E image of adipose tissue",                 # 0 ADI
    "An H&E image of background",                     # 1 BACK
    "An H&E image of debris",                         # 2 DEB
    "An H&E image of lymphocytes",                    # 3 LYM
    "An H&E image of mucus",                          # 4 MUC
    "An H&E image of smooth muscle",                  # 5 MUS
    "An H&E image of normal mucosa",                  # 6 NORM
    "An H&E image of cancer-associated stroma",       # 7 STR
    "An H&E image of adenocarcinoma epithelium",      # 8 TUM
]

DISTRACTORS = {
    0: ["An H&E image of vessels"],
    1: ["An H&E image of fibrous tissue"],
    2: ["An H&E image of necrotic tissue"],
    3: ["An H&E image of inflammatory infiltrates"],
    4: ["An H&E image of submucosa"],
    5: ["An H&E image of stroma"],
    6: ["An H&E image of glandular tissue", "An H&E image of squamous epithelium"],
    7: ["An H&E image of nerves"],
    8: ["An H&E image of dysplasia", "An H&E image of hyperplasia"],
}


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


def build_prompts(id_cls):
    prompts = [BASE_PROMPTS[index] for index in id_cls]
    distractors = []
    for index in id_cls:
        distractors.extend(DISTRACTORS[index])
    prompts.extend(list(dict.fromkeys(distractors)))
    return prompts


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_models(model_type):
    """Load only the model(s) required by the selected fixed baseline."""
    bmc_model = None
    bmc_tokenizer = None
    plip_model = None
    plip_processor = None

    if model_type in {"BMC", "combine"}:
        bmc_model, _ = create_model_from_pretrained(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        bmc_tokenizer = get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        bmc_model = bmc_model.cuda().eval()

    if model_type in {"plip", "combine"}:
        plip_model = CLIPModel.from_pretrained("vinid/plip")
        plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
        plip_model = plip_model.cuda().eval()

    return bmc_model, bmc_tokenizer, plip_model, plip_processor


@torch.inference_mode()
def zero_shot_screen(loader, prompts, model_type, models):
    bmc_model, bmc_tokenizer, plip_model, plip_processor = models
    prediction_chunks = []
    probability_chunks = []
    embedding_chunks = []
    names = []

    bmc_texts = None
    plip_text_inputs = None
    if model_type in {"BMC", "combine"}:
        bmc_texts = bmc_tokenizer(prompts).cuda()
    if model_type in {"plip", "combine"}:
        plip_text_inputs = plip_processor(text=prompts, return_tensors="pt", padding=True)
        plip_text_inputs = {key: value.cuda() for key, value in plip_text_inputs.items()}

    for sample in loader:
        images = sample["img"].cuda(non_blocking=True)
        batch_names = list(sample["img_name"])

        if model_type in {"plip", "combine"}:
            plip_inputs = dict(plip_text_inputs)
            plip_inputs["pixel_values"] = images
            plip_outputs = plip_model(**plip_inputs)
            plip_probs = plip_outputs.logits_per_image.softmax(dim=1)
            plip_features = plip_outputs.image_embeds

        if model_type in {"BMC", "combine"}:
            bmc_features, bmc_text_features, logit_scale = bmc_model(images, bmc_texts)
            bmc_probs = (logit_scale * bmc_features @ bmc_text_features.t()).softmax(dim=-1)

        if model_type == "plip":
            probs = plip_probs
            features = plip_features
        elif model_type == "BMC":
            probs = bmc_probs
            features = bmc_features
        elif model_type == "combine":
            probs = (plip_probs + bmc_probs) / 2.0
            features = torch.cat([plip_features, bmc_features], dim=1)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        prediction_chunks.append(probs.argmax(dim=1).cpu())
        probability_chunks.append(probs.cpu())
        embedding_chunks.append(features.float().cpu())
        names.extend(batch_names)

    return (
        torch.cat(prediction_chunks).numpy().astype(np.uint8),
        torch.cat(probability_chunks).numpy(),
        np.asarray(names),
        torch.cat(embedding_chunks).numpy(),
    )


def kmeans_representatives(embeddings: np.ndarray, query_size: int, seed: int) -> np.ndarray:
    if len(embeddings) < query_size:
        raise ValueError(
            f"OpenPath screening produced {len(embeddings)} ID candidates, "
            f"fewer than query_size={query_size}"
        )
    kmeans = KMeans(
        n_clusters=query_size,
        init="k-means++",
        n_init=10,
        random_state=seed,
    )
    labels = kmeans.fit_predict(embeddings)
    selected = []
    for cluster_id in range(query_size):
        members = np.flatnonzero(labels == cluster_id)
        distances = ((embeddings[members] - kmeans.cluster_centers_[cluster_id]) ** 2).sum(axis=1)
        selected.append(members[int(distances.argmin())])
    return np.asarray(selected, dtype=int)


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"


def append_report(log_path: Path, report: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(report)


def get_args():
    parser = argparse.ArgumentParser(description="OpenPath same-candidate-pool baseline for CRC100K")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument(
        "--id_cls",
        nargs=3,
        type=int,
        default=None,
        metavar=("ID1", "ID2", "ID3"),
        help="Optional explicit ID classes. If omitted, three classes are generated from --seed.",
    )
    parser.add_argument("--pool_size", type=int, default=300)
    parser.add_argument("--pool_seed", type=int, default=2026,
                        help="Fixed pool seed used by vlm_new_random_CRC3.py")
    parser.add_argument("--query_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model_type", choices=["BMC", "plip", "combine"], default="combine")
    parser.add_argument("--data_csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log_file", type=Path, default=DEFAULT_LOG_FILE)
    args = parser.parse_args()

    if args.id_cls is None:
        args.id_cls, args.ood_cls = choose_id_classes(args.seed)
    else:
        args.id_cls = sorted(args.id_cls)
        if len(set(args.id_cls)) != 3:
            parser.error("--id_cls must contain three different class IDs")
        if any(class_id < 0 or class_id >= NUM_CLASSES for class_id in args.id_cls):
            parser.error(f"--id_cls values must be between 0 and {NUM_CLASSES - 1}")
        args.ood_cls = sorted(
            class_id for class_id in range(NUM_CLASSES) if class_id not in args.id_cls
        )

    args.num_class = len(args.id_cls)
    return args


def main() -> None:
    total_start = time.perf_counter()
    args = get_args()
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("OpenPath-same-pool requires a CUDA GPU")
    torch.cuda.set_device(args.gpu)
    id_cls = args.id_cls
    ood_cls = args.ood_cls

    data_start = time.perf_counter()
    full_df = load_dataframe(args.data_csv)
    data_seconds = time.perf_counter() - data_start

    pool_start = time.perf_counter()
    pool_df = sample_candidate_pool(full_df, args.pool_size, args.pool_seed)
    records = [
        {"img": row.img, "label": int(row.cls_label)}
        for row in pool_df.itertuples(index=False)
    ]
    dataset = Tumor_dataset_val_cls(args, files=records)
    loader = get_loader(args, dataset, shuffle=False, batch_size=args.batch_size)
    pool_seconds = time.perf_counter() - pool_start

    synchronize()
    model_start = time.perf_counter()
    models = load_models(args.model_type)
    synchronize()
    model_seconds = time.perf_counter() - model_start

    prompts = build_prompts(id_cls)
    synchronize()
    screening_start = time.perf_counter()
    predictions, _, names, embeddings = zero_shot_screen(
        loader, prompts, args.model_type, models
    )
    synchronize()
    screening_seconds = time.perf_counter() - screening_start

    passed_mask = predictions < len(id_cls)
    passed_names = names[passed_mask]
    passed_embeddings = embeddings[passed_mask]
    truth = {
        os.path.basename(str(image)): int(label)
        for image, label in zip(pool_df["img"], pool_df["cls_label"])
    }
    raw_id = sum(int(truth[os.path.basename(str(name))] in id_cls) for name in passed_names)
    raw_ood = len(passed_names) - raw_id
    raw_qp = 100.0 * raw_id / len(passed_names) if len(passed_names) else 0.0

    selection_start = time.perf_counter()
    selected_indices = kmeans_representatives(passed_embeddings, args.query_size, args.seed)
    selected_names = passed_names[selected_indices]
    selected_labels = np.asarray(
        [truth[os.path.basename(str(name))] for name in selected_names],
        dtype=int,
    )
    selection_seconds = time.perf_counter() - selection_start

    selected_id = int(np.isin(selected_labels, id_cls).sum())
    selected_ood = len(selected_labels) - selected_id
    final_qp = 100.0 * selected_id / len(selected_labels)

    save_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"query_openpath_same_pool_seed{args.seed}.csv"
    pd.DataFrame({"img": selected_names, "label": selected_labels}).to_csv(output_csv, index=False)
    save_seconds = time.perf_counter() - save_start
    total_seconds = time.perf_counter() - total_start

    report = f"""
================================================================================
Method: OpenPath-same-pool
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
Seed: {args.seed}
ID classes: {id_cls}
OOD classes: {ood_cls}
Final query size: {len(selected_names)}
Screened ID-candidate count: {len(passed_names)}
Screened candidate ID/OOD: {raw_id}/{raw_ood}
Raw screened QP: {raw_qp:.2f}%
Final query ID/OOD: {selected_id}/{selected_ood}
Final query QP: {final_qp:.2f}%
Total runtime: {format_duration(total_seconds)}
================================================================================
"""
    print(report)
    append_report(args.log_file, report)
    print(f"Results appended to: {args.log_file}")


if __name__ == "__main__":
    main()
