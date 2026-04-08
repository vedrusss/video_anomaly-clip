from __future__ import annotations

import argparse
import glob
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

if TYPE_CHECKING:
    from src.models.components.anomaly_clip import AnomalyCLIP

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

DATASET_DEFAULTS = {
    "ucfcrime": {
        "labels_file": "data/ucf_labels.csv",
        "normal_id": 7,
        "concat_features": False,
        "emb_size": 256,
        "depth": 1,
    },
    "shanghaitech": {
        "labels_file": "data/sht_labels.csv",
        "normal_id": 8,
        "concat_features": True,
        "emb_size": 256,
        "depth": 2,
    },
    "xdviolence": {
        "labels_file": "data/xd_labels.csv",
        "normal_id": 4,
        "concat_features": False,
        "emb_size": 128,
        "depth": 1,
    },
}


@dataclass
class VideoBatch:
    frames: torch.Tensor
    original_length: int
    padded_length: int
    frame_indices: List[int]
    timestamps_sec: List[float]
    native_fps: float
    sample_fps: float
    duration_sec: float
    segment_size: int


@dataclass
class SegmentPrediction:
    class_id: int
    class_name: str
    start_time_sec: float
    end_time_sec: float


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device: torch.device) -> float:
    sync_device(device)
    return time.perf_counter()


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def format_duration(seconds: float) -> str:
    return f"{seconds:.3f}s"


def build_clip_preprocess(input_size: int = 224) -> transforms.Compose:
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(clip_mean, clip_std),
        ]
    )


def supported_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def resolve_video_inputs(inputs: Sequence[str], recursive: bool = False) -> List[Path]:
    resolved: List[Path] = []
    seen = set()

    for raw_input in inputs:
        candidate = Path(raw_input)

        if candidate.is_dir():
            iterator = candidate.rglob("*") if recursive else candidate.glob("*")
            matches = sorted(p for p in iterator if p.is_file() and supported_video(p))
        else:
            matches = sorted(Path(p) for p in glob.glob(raw_input))
            if not matches and candidate.is_file():
                matches = [candidate]

        for match in matches:
            match = match.resolve()
            if match not in seen and match.is_file() and supported_video(match):
                resolved.append(match)
                seen.add(match)

    return resolved


def infer_dataset_name(checkpoint_path: Path, labels_file: Path | None, dataset: str | None) -> str:
    if dataset:
        dataset = dataset.lower()
        if dataset not in DATASET_DEFAULTS:
            raise ValueError(f"Unsupported dataset '{dataset}'.")
        return dataset

    if labels_file is not None:
        labels_name = labels_file.name.lower()
        for dataset_name, defaults in DATASET_DEFAULTS.items():
            if Path(defaults["labels_file"]).name.lower() == labels_name:
                return dataset_name

    checkpoint_lower = checkpoint_path.as_posix().lower()
    for dataset_name in DATASET_DEFAULTS:
        if dataset_name in checkpoint_lower:
            return dataset_name

    raise ValueError(
        "Could not infer dataset from checkpoint path. Pass --dataset explicitly "
        "(ucfcrime, shanghaitech, xdviolence)."
    )


def resolve_labels_file(repo_root: Path, dataset_name: str, labels_file: str | None) -> Path:
    if labels_file:
        return Path(labels_file).expanduser().resolve()
    return (repo_root / DATASET_DEFAULTS[dataset_name]["labels_file"]).resolve()


def infer_normal_id(labels_df: pd.DataFrame, normal_id: int | None) -> int:
    if normal_id is not None:
        return normal_id

    names = labels_df["name"].astype(str).str.lower()
    matches = labels_df.loc[names == "normal", "id"].tolist()
    if not matches:
        raise ValueError("Could not infer normal class id from labels CSV. Pass --normal-id.")
    return int(matches[0])


def build_net_args(dataset_name: str, labels_file: Path, normal_id: int) -> dict:
    defaults = DATASET_DEFAULTS[dataset_name]
    return {
        "arch": "ViT-B/16",
        "shared_context": False,
        "ctx_init": "",
        "n_ctx": 8,
        "seg_length": 16,
        "num_segments": 32,
        "select_idx_dropout_topk": 0.7,
        "select_idx_dropout_bottomk": 0.7,
        "heads": 8,
        "dim_heads": None,
        "concat_features": defaults["concat_features"],
        "emb_size": defaults["emb_size"],
        "depth": defaults["depth"],
        "num_topk": 3,
        "num_bottomk": 3,
        "labels_file": str(labels_file),
        "normal_id": normal_id,
        "dropout_prob": 0.0,
        "temporal_module": "axial",
        "direction_module": "learned_encoder_finetune",
        "selector_module": "directions",
        "batch_norm": True,
        "feature_size": 512,
        "use_similarity_as_features": False,
        "load_from_features": True,
        "stride": 1,
        "ncrops": 1,
    }


def normalize_checkpoint_state_dict(state_dict: dict) -> dict:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("net."):
            new_key = new_key[len("net.") :]
        normalized[new_key] = value
    return normalized


def load_net_weights_from_ckpt(
    net: "AnomalyCLIP",
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = normalize_checkpoint_state_dict(state_dict)
    model_state = net.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
    missing, unexpected = net.load_state_dict(filtered_state, strict=strict)

    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys for inference: {unexpected}")

    allowed_missing_prefixes = (
        "image_encoder.",
        "token_embedding.",
        "text_encoder.transformer.",
        "text_encoder.positional_embedding",
        "text_encoder.ln_final.",
    )
    required_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if required_missing:
        raise RuntimeError(
            "Checkpoint is missing required AnomalyCLIP weights: "
            + ", ".join(required_missing[:20])
        )

    net.to(device)
    net.eval()


def resolve_ncentroid_path(
    checkpoint_path: Path,
    explicit_path: str | None,
    repo_root: Path,
) -> Path | None:
    candidates: List[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser().resolve())

    candidates.append(checkpoint_path.parent / "ncentroid.pt")
    candidates.append(repo_root / "logs" / "train" / "runs" / checkpoint_path.parent.name / "ncentroid.pt")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_ncentroid(ncentroid_path: Path, device: torch.device) -> torch.Tensor:
    ncentroid = torch.load(ncentroid_path, map_location="cpu")
    if not isinstance(ncentroid, torch.Tensor):
        raise TypeError(f"Expected tensor in {ncentroid_path}, got {type(ncentroid)!r}.")
    return ncentroid.to(device=device, dtype=torch.float32)


def sample_frame_indices(
    total_frames: int,
    native_fps: float,
    target_fps: float | None,
) -> List[int]:
    if total_frames <= 0:
        return []

    if not target_fps or target_fps <= 0 or not native_fps or native_fps <= 0:
        return list(range(total_frames))

    if target_fps >= native_fps:
        return list(range(total_frames))

    step = native_fps / target_fps
    indices = np.floor(np.arange(0, total_frames, step)).astype(int)
    indices = np.unique(np.clip(indices, 0, total_frames - 1))
    return indices.tolist()


def preprocess_video(
    video_path: Path,
    preprocess: transforms.Compose,
    num_segments: int,
    seg_length: int,
    target_fps: float | None,
) -> VideoBatch:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if native_fps <= 0:
        native_fps = float(target_fps) if target_fps and target_fps > 0 else 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = (total_frames / native_fps) if total_frames > 0 else 0.0

    desired_indices = sample_frame_indices(total_frames, native_fps, target_fps)
    tensors: List[torch.Tensor] = []
    sampled_indices: List[int] = []
    sampled_times_sec: List[float] = []

    if desired_indices:
        desired_ptr = 0
        current_frame = 0

        while desired_ptr < len(desired_indices):
            ok, frame = cap.read()
            if not ok:
                break
            if current_frame == desired_indices[desired_ptr]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb)
                tensors.append(preprocess(pil_image))
                sampled_indices.append(current_frame)
                sampled_times_sec.append(current_frame / native_fps)
                desired_ptr += 1
            current_frame += 1
    else:
        current_frame = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            tensors.append(preprocess(pil_image))
            sampled_indices.append(current_frame)
            sampled_times_sec.append(current_frame / native_fps)
            current_frame += 1

    cap.release()

    if not tensors:
        raise RuntimeError(f"No frames sampled from video: {video_path}")

    original_length = len(tensors)
    base_unit = num_segments * seg_length
    padded_length = int(math.ceil(original_length / base_unit) * base_unit)
    padded_frames = list(tensors)
    while len(padded_frames) < padded_length:
        padded_frames.append(padded_frames[-1].clone())

    frames = torch.stack(padded_frames, dim=0)
    segment_size = padded_length // base_unit
    effective_sample_fps = original_length / duration_sec if duration_sec > 0 else native_fps

    return VideoBatch(
        frames=frames,
        original_length=original_length,
        padded_length=padded_length,
        frame_indices=sampled_indices,
        timestamps_sec=sampled_times_sec,
        native_fps=native_fps,
        sample_fps=effective_sample_fps,
        duration_sec=duration_sec,
        segment_size=segment_size,
    )


def extract_clip_features(
    frames: torch.Tensor,
    net: "AnomalyCLIP",
    device: torch.device,
    batch_size: int,
    normalize: bool,
) -> torch.Tensor:
    feats = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size].to(device)
            encoded = net.image_encoder(batch.float())
            if normalize:
                encoded = F.normalize(encoded, dim=-1)
            feats.append(encoded)
    return torch.cat(feats, dim=0)


def compute_ncentroid_from_features(features: torch.Tensor) -> torch.Tensor:
    return features.mean(dim=0)


def insert_normal_probabilities(
    abnormal_scores: torch.Tensor,
    class_probs: torch.Tensor,
    normal_id: int,
) -> torch.Tensor:
    normal_probs = (1.0 - abnormal_scores).unsqueeze(1)
    return torch.cat(
        (class_probs[:, :normal_id], normal_probs, class_probs[:, normal_id:]),
        dim=1,
    )


def frame_predictions_to_segments(
    predicted_labels: Sequence[int],
    timestamps_sec: Sequence[float],
    class_names: Sequence[str],
    normal_id: int,
    duration_sec: float,
) -> List[SegmentPrediction]:
    segments: List[SegmentPrediction] = []
    if not predicted_labels:
        return segments

    average_step = 0.0
    if len(timestamps_sec) > 1:
        diffs = np.diff(np.asarray(timestamps_sec, dtype=float))
        average_step = float(np.median(diffs))

    start = 0
    while start < len(predicted_labels):
        label = predicted_labels[start]
        if label == normal_id:
            start += 1
            continue

        end = start
        while end + 1 < len(predicted_labels) and predicted_labels[end + 1] == label:
            end += 1

        start_time = float(timestamps_sec[start])
        if end + 1 < len(timestamps_sec):
            end_time = float(timestamps_sec[end + 1])
        elif duration_sec > 0:
            end_time = duration_sec
        else:
            end_time = float(timestamps_sec[end] + average_step)

        segments.append(
            SegmentPrediction(
                class_id=label,
                class_name=class_names[label],
                start_time_sec=start_time,
                end_time_sec=max(start_time, end_time),
            )
        )
        start = end + 1

    return segments


def dominant_abnormal_class(
    predicted_labels: Sequence[int],
    class_names: Sequence[str],
    normal_id: int,
) -> str | None:
    abnormal = [label for label in predicted_labels if label != normal_id]
    if not abnormal:
        return None
    dominant = max(set(abnormal), key=abnormal.count)
    return class_names[dominant]


def summarize_video_prediction(
    predicted_labels: Sequence[int],
    timestamps_sec: Sequence[float],
    class_names: Sequence[str],
    normal_id: int,
    duration_sec: float,
) -> str:
    segments = frame_predictions_to_segments(
        predicted_labels=predicted_labels,
        timestamps_sec=timestamps_sec,
        class_names=class_names,
        normal_id=normal_id,
        duration_sec=duration_sec,
    )

    if not segments:
        return "Normal"

    dominant = dominant_abnormal_class(predicted_labels, class_names, normal_id)
    parts = [
        f"{segment.class_name} [{format_seconds(segment.start_time_sec)}-{format_seconds(segment.end_time_sec)}]"
        for segment in segments
    ]

    if dominant is None:
        return "; ".join(parts)
    return f"{dominant} (dominant) | " + "; ".join(parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AnomalyCLIP VAR inference on one or more raw videos."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video paths, directories, or glob patterns (for example folder/*.mp4).",
    )
    parser.add_argument("--ckpt", required=True, help="Path to the AnomalyCLIP checkpoint.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DEFAULTS.keys()),
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
        help="Optional path to a saved ncentroid.pt. If omitted, the script tries common locations and otherwise computes it from the current video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target sampling FPS for raw video inference. Use <=0 to keep every frame.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Abnormality threshold above which a sampled frame is considered anomalous.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP feature extraction.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="L2-normalize CLIP frame features before they are passed to AnomalyCLIP.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, scan it recursively for video files.",
    )
    return parser
