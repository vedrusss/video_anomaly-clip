from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.common import (  # noqa: E402
    build_clip_preprocess,
    extract_clip_features,
    format_duration,
    resolve_video_inputs,
    sample_frame_indices,
)


@dataclass
class SampledVideo:
    frames: torch.Tensor
    frame_indices: List[int]
    timestamps_sec: List[float]
    native_fps: float
    sample_fps: float
    duration_sec: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode raw videos into CLIP feature caches for later AnomalyCLIP classification."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video paths, directories, or glob patterns (for example folder/*.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where encoded feature caches will be written.",
    )
    parser.add_argument(
        "--cache-ext",
        default=".pt",
        help="Extension for cache files (default: .pt).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Target sampling FPS. Use <=0 to keep every frame.",
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
        help="Encoding device, for example cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="L2-normalize CLIP frame features before saving.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When an input is a directory, scan it recursively for video files.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA autocast during CLIP encoding for better throughput.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Autocast dtype when --amp is enabled.",
    )
    return parser


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def preprocess_video_unpadded(
    video_path: Path,
    preprocess,
    target_fps: float | None,
) -> SampledVideo:
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

    frames = torch.stack(tensors, dim=0)
    sample_fps = (len(sampled_indices) / duration_sec) if duration_sec > 0 else native_fps

    return SampledVideo(
        frames=frames,
        frame_indices=sampled_indices,
        timestamps_sec=sampled_times_sec,
        native_fps=native_fps,
        sample_fps=sample_fps,
        duration_sec=duration_sec,
    )


def ensure_cache_ext(cache_ext: str) -> str:
    return cache_ext if cache_ext.startswith(".") else f".{cache_ext}"


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_ext = ensure_cache_ext(args.cache_ext)
    video_paths = resolve_video_inputs(args.inputs, recursive=args.recursive)
    if not video_paths:
        raise FileNotFoundError("No video files matched the provided inputs.")

    collisions = {}
    for video_path in video_paths:
        out_path = output_dir / f"{video_path.stem}{cache_ext}"
        collisions.setdefault(out_path, []).append(video_path)

    duplicated = {path: srcs for path, srcs in collisions.items() if len(srcs) > 1}
    if duplicated:
        details = "; ".join(
            f"{path.name} <- {[src.name for src in srcs]}" for path, srcs in duplicated.items()
        )
        raise RuntimeError(
            "Output filename collision detected. Video stems must be unique for this output directory: "
            + details
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    preprocess = build_clip_preprocess()

    from src.models.components.clip import clip

    clip_model, _ = clip.load("ViT-B/16", device="cpu")
    clip_model.float()
    image_encoder = clip_model.visual
    image_encoder.to(device)
    image_encoder.eval()

    total = len(video_paths)
    per_video_times: List[float] = []
    wall_start = time.perf_counter()

    for idx, video_path in enumerate(video_paths, start=1):
        item_start = time.perf_counter()

        sampled_video = preprocess_video_unpadded(
            video_path=video_path,
            preprocess=preprocess,
            target_fps=args.fps,
        )

        features = extract_clip_features(
            frames=sampled_video.frames,
            net=None,
            image_encoder=image_encoder,
            device=device,
            batch_size=args.batch_size,
            normalize=args.normalize_features,
            amp=args.amp,
            amp_dtype=amp_dtype,
        )

        payload = {
            "format": "anomalyclip_clip_features_v1",
            "video_name": video_path.name,
            "video_path": str(video_path),
            "features": features.detach().to(device="cpu", dtype=torch.float32),
            "frame_indices": sampled_video.frame_indices,
            "timestamps_sec": sampled_video.timestamps_sec,
            "native_fps": sampled_video.native_fps,
            "sample_fps": sampled_video.sample_fps,
            "duration_sec": sampled_video.duration_sec,
            "normalize_features": bool(args.normalize_features),
        }

        out_path = output_dir / f"{video_path.stem}{cache_ext}"
        torch.save(payload, out_path)

        elapsed = time.perf_counter() - item_start
        per_video_times.append(elapsed)

        print(
            f"{datetime.now().strftime('%H:%M:%S')}, {video_path.name} finished "
            f"({idx} from {total}) in {elapsed:.3f} seconds."
        )

    total_time = time.perf_counter() - wall_start
    avg_time = sum(per_video_times) / len(per_video_times)

    print(f"Saved feature caches to: {output_dir}")
    print(f"Average encode time: {format_duration(avg_time)}")
    print(f"Total encode time: {format_duration(total_time)}")


if __name__ == "__main__":
    main()
