from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Set

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.common import (  # noqa: E402
    build_net_args,
    compute_ncentroid_from_features,
    format_duration,
    infer_dataset_name,
    infer_normal_id,
    load_net_weights_from_ckpt,
    load_ncentroid,
    now,
    resolve_labels_file,
    resolve_ncentroid_path,
    summarize_video_prediction,
)

if TYPE_CHECKING:
    from src.models.components.anomaly_clip import AnomalyCLIP


@dataclass
class EncodedVideoResult:
    cache_path: Path
    video_name: str
    summary: str
    total_time_sec: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify cached CLIP video features with a chosen AnomalyCLIP checkpoint."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Cache file paths, directories, or glob patterns (for example encoded/*.pt).",
    )
    parser.add_argument("--ckpt", required=True, help="Path to the AnomalyCLIP checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=("shanghaitech", "ucfcrime", "xdviolence"),
        help="Dataset configuration that matches the checkpoint.",
    )
    parser.add_argument(
        "--labels-file",
        help="CSV with class labels. Defaults to the repository labels file for the chosen dataset.",
    )
    parser.add_argument(
        "--normal-id",
        type=int,
        help="Index of the Normal class. Defaults to the Normal entry from the labels CSV.",
    )
    parser.add_argument(
        "--ncentroid-path",
        help="Optional path to a saved ncentroid.pt. If omitted, the script tries common locations and otherwise computes it from each video's features.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Abnormality threshold above which a sampled frame is considered anomalous.",
    )
    parser.add_argument(
        "--cache-ext",
        default=".pt",
        help="Extension for cache files when scanning folders (default: .pt).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, scan it recursively for cache files.",
    )
    parser.add_argument(
        "--gt",
        help="Optional path to ground-truth JSON in format {video_filename: threat_caption}.",
    )
    return parser


def ensure_extension(cache_ext: str) -> str:
    return cache_ext if cache_ext.startswith(".") else f".{cache_ext}"


def resolve_cache_inputs(inputs: Sequence[str], cache_ext: str, recursive: bool = False) -> List[Path]:
    resolved: List[Path] = []
    seen = set()

    for raw_input in inputs:
        candidate = Path(raw_input)

        if candidate.is_dir():
            pattern = f"*{cache_ext}"
            iterator = candidate.rglob(pattern) if recursive else candidate.glob(pattern)
            matches = sorted(p for p in iterator if p.is_file())
        else:
            matches = sorted(Path(p) for p in glob.glob(raw_input))
            if not matches and candidate.is_file():
                matches = [candidate]

        for match in matches:
            match = match.resolve()
            if match not in seen and match.is_file():
                resolved.append(match)
                seen.add(match)

    return resolved


def load_encoded_payload(cache_path: Path) -> tuple[str, torch.Tensor, List[float], float]:
    payload = torch.load(cache_path, map_location="cpu")

    if isinstance(payload, torch.Tensor):
        features = payload
        video_name = cache_path.stem
        timestamps_sec = list(range(int(features.shape[0])))
        duration_sec = float(features.shape[0])
    elif isinstance(payload, dict):
        features = payload.get("features")
        if not isinstance(features, torch.Tensor):
            raise TypeError(f"Cache file {cache_path} does not contain a tensor under 'features'.")

        video_name = str(payload.get("video_name") or Path(payload.get("video_path", cache_path.stem)).name)

        timestamps_raw = payload.get("timestamps_sec")
        if timestamps_raw is None:
            timestamps_sec = list(range(int(features.shape[0])))
        else:
            timestamps_sec = [float(x) for x in timestamps_raw]

        duration_raw = payload.get("duration_sec")
        if duration_raw is None:
            duration_sec = float(timestamps_sec[-1]) if timestamps_sec else float(features.shape[0])
        else:
            duration_sec = float(duration_raw)
    else:
        raise TypeError(f"Unsupported cache payload type in {cache_path}: {type(payload)!r}")

    if features.ndim != 2:
        raise ValueError(
            f"Expected [num_frames, feature_dim] tensor in {cache_path}, got shape {tuple(features.shape)}."
        )

    if not timestamps_sec:
        timestamps_sec = list(range(int(features.shape[0])))

    if len(timestamps_sec) != int(features.shape[0]):
        raise ValueError(
            f"timestamps_sec length mismatch in {cache_path}: "
            f"{len(timestamps_sec)} vs {features.shape[0]}"
        )

    return video_name, features.to(dtype=torch.float32), timestamps_sec, duration_sec


def pad_features_to_model_length(
    features: torch.Tensor,
    num_segments: int,
    seg_length: int,
) -> tuple[torch.Tensor, int, int, int]:
    original_length = int(features.shape[0])
    base_unit = num_segments * seg_length
    padded_length = int(math.ceil(original_length / base_unit) * base_unit)

    if padded_length == original_length:
        padded = features
    else:
        pad_count = padded_length - original_length
        tail = features[-1:].repeat(pad_count, 1)
        padded = torch.cat([features, tail], dim=0)

    segment_size = padded_length // base_unit
    return padded, original_length, padded_length, segment_size


def load_gt_abnormal_videos(gt_path: str) -> Set[str]:
    gt_file = Path(gt_path).expanduser().resolve()
    if not gt_file.is_file():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_file}")

    with gt_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Ground-truth JSON must be an object mapping video_filename to label: {gt_file}")

    return {Path(str(video_name)).name for video_name in payload.keys()}


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    repo_root = REPO_ROOT
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_name = infer_dataset_name(
        checkpoint_path=checkpoint_path,
        labels_file=Path(args.labels_file).expanduser().resolve() if args.labels_file else None,
        dataset=args.dataset,
    )
    labels_file = resolve_labels_file(repo_root, dataset_name, args.labels_file)
    if not labels_file.is_file():
        raise FileNotFoundError(f"Labels CSV not found: {labels_file}")

    labels_df = pd.read_csv(labels_file)
    normal_id = infer_normal_id(labels_df, args.normal_id)
    class_names = labels_df["name"].astype(str).tolist()

    device = torch.device(args.device)
    from src.models.components.anomaly_clip import AnomalyCLIP

    net = AnomalyCLIP(**build_net_args(dataset_name, labels_file, normal_id))
    load_net_weights_from_ckpt(net=net, checkpoint_path=checkpoint_path, device=device, strict=False)

    # Cache prompt-based text embeddings once: they are constant in eval mode for a loaded checkpoint.
    with torch.inference_mode():
        cached_text_features = net.get_text_features()
    net.get_text_features = lambda: cached_text_features  # type: ignore[method-assign]

    ncentroid_path = resolve_ncentroid_path(
        checkpoint_path=checkpoint_path,
        explicit_path=args.ncentroid_path,
        repo_root=repo_root,
    )
    shared_ncentroid = load_ncentroid(ncentroid_path, device) if ncentroid_path else None

    cache_ext = ensure_extension(args.cache_ext)
    cache_paths = resolve_cache_inputs(args.inputs, cache_ext=cache_ext, recursive=args.recursive)
    if not cache_paths:
        raise FileNotFoundError("No cache files matched the provided inputs.")

    gt_abnormal_videos: Set[str] | None = load_gt_abnormal_videos(args.gt) if args.gt else None
    predicted_abnormal_videos: Set[str] = set()

    results: List[EncodedVideoResult] = []
    total = len(cache_paths)
    normal_videos = 0
    abnormal_videos = 0

    wall_start = time.perf_counter()

    for idx, cache_path in enumerate(cache_paths, start=1):
        item_start = now(device)

        video_name, features, timestamps_sec, duration_sec = load_encoded_payload(cache_path)
        padded, original_length, padded_length, segment_size = pad_features_to_model_length(
            features,
            num_segments=net.num_segments,
            seg_length=net.seg_length,
        )

        with torch.inference_mode():
            ncentroid = shared_ncentroid
            frame_features = padded.to(device=device)
            if ncentroid is None:
                ncentroid = compute_ncentroid_from_features(frame_features)

            feature_input = frame_features.unsqueeze(0).unsqueeze(0)
            dummy_labels = torch.zeros(padded_length, dtype=torch.long, device=device)

            similarity, abnormal_scores = net(
                feature_input,
                dummy_labels,
                ncentroid,
                segment_size=segment_size,
                test_mode=True,
            )

            similarity = similarity[:original_length]
            abnormal_scores = abnormal_scores[:original_length]
            anomaly_class_probs = torch.softmax(similarity, dim=1) * abnormal_scores.unsqueeze(1)

            predicted_labels = []
            for score, class_probs in zip(abnormal_scores, anomaly_class_probs):
                if float(score) < args.threshold:
                    predicted_labels.append(normal_id)
                else:
                    predicted = int(torch.argmax(class_probs).item())
                    if predicted >= normal_id:
                        predicted += 1
                    predicted_labels.append(predicted)

        summary = summarize_video_prediction(
            predicted_labels=predicted_labels,
            timestamps_sec=timestamps_sec,
            class_names=class_names,
            normal_id=normal_id,
            duration_sec=duration_sec,
        )

        elapsed = now(device) - item_start

        resolved_video_name = Path(video_name).name

        if summary == "Normal":
            normal_videos += 1
        else:
            abnormal_videos += 1
            predicted_abnormal_videos.add(resolved_video_name)

        print(f"{video_name} {idx} from {total}, {summary}.")

        results.append(
            EncodedVideoResult(
                cache_path=cache_path,
                video_name=video_name,
                summary=summary,
                total_time_sec=elapsed,
            )
        )

    total_time = time.perf_counter() - wall_start
    avg_time = sum(r.total_time_sec for r in results) / len(results)

    print(f"Average execution time: {format_duration(avg_time)}")
    print(f"Total execution time: {format_duration(total_time)}")
    print(f"Normal videos: {normal_videos}")
    print(f"Not-Normal videos: {abnormal_videos}")

    if gt_abnormal_videos is not None:
        tp = len(predicted_abnormal_videos & gt_abnormal_videos)
        fp = len(predicted_abnormal_videos - gt_abnormal_videos)
        fn = len(gt_abnormal_videos - predicted_abnormal_videos)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        print(f"TPs: {tp}, FPs: {fp}, FNs: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}")


if __name__ == "__main__":
    main()
