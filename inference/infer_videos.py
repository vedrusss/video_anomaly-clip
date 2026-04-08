from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.common import (  # noqa: E402
    build_arg_parser,
    build_clip_preprocess,
    build_net_args,
    compute_ncentroid_from_features,
    extract_clip_features,
    format_duration,
    infer_dataset_name,
    infer_normal_id,
    load_net_weights_from_ckpt,
    load_ncentroid,
    now,
    preprocess_video,
    resolve_labels_file,
    resolve_ncentroid_path,
    resolve_video_inputs,
    summarize_video_prediction,
)
if TYPE_CHECKING:
    from src.models.components.anomaly_clip import AnomalyCLIP


@dataclass
class VideoInferenceResult:
    video_path: Path
    summary: str
    preprocess_time_sec: float
    classify_time_sec: float

    @property
    def total_time_sec(self) -> float:
        return self.preprocess_time_sec + self.classify_time_sec


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
    preprocess = build_clip_preprocess()
    from src.models.components.anomaly_clip import AnomalyCLIP

    net = AnomalyCLIP(**build_net_args(dataset_name, labels_file, normal_id))
    load_net_weights_from_ckpt(net=net, checkpoint_path=checkpoint_path, device=device, strict=False)

    ncentroid_path = resolve_ncentroid_path(
        checkpoint_path=checkpoint_path,
        explicit_path=args.ncentroid_path,
        repo_root=repo_root,
    )
    shared_ncentroid = load_ncentroid(ncentroid_path, device) if ncentroid_path else None

    video_paths = resolve_video_inputs(args.inputs, recursive=args.recursive)
    if not video_paths:
        raise FileNotFoundError("No video files matched the provided inputs.")

    results: List[VideoInferenceResult] = []

    for video_path in video_paths:
        preprocess_start = now(device)
        video_batch = preprocess_video(
            video_path=video_path,
            preprocess=preprocess,
            num_segments=net.num_segments,
            seg_length=net.seg_length,
            target_fps=args.fps,
        )
        preprocess_end = now(device)

        classify_start = preprocess_end
        with torch.no_grad():
            features = extract_clip_features(
                frames=video_batch.frames,
                net=net,
                device=device,
                batch_size=args.batch_size,
                normalize=args.normalize_features,
            )

            ncentroid = shared_ncentroid
            if ncentroid is None:
                ncentroid = compute_ncentroid_from_features(features)

            feature_input = features.unsqueeze(0).unsqueeze(0)
            dummy_labels = torch.zeros(
                video_batch.padded_length,
                dtype=torch.long,
                device=device,
            )

            similarity, abnormal_scores = net(
                feature_input,
                dummy_labels,
                ncentroid,
                segment_size=video_batch.segment_size,
                test_mode=True,
            )

            similarity = similarity[: video_batch.original_length]
            abnormal_scores = abnormal_scores[: video_batch.original_length]
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

        classify_end = now(device)

        summary = summarize_video_prediction(
            predicted_labels=predicted_labels,
            timestamps_sec=video_batch.timestamps_sec,
            class_names=class_names,
            normal_id=normal_id,
            duration_sec=video_batch.duration_sec,
        )

        results.append(
            VideoInferenceResult(
                video_path=video_path,
                summary=summary,
                preprocess_time_sec=preprocess_end - preprocess_start,
                classify_time_sec=classify_end - classify_start,
            )
        )

    for result in results:
        print(f"{result.video_path.name} -> {result.summary}")

    print()

    for result in results:
        print(
            f"{result.video_path.name}: "
            f"preprocess={format_duration(result.preprocess_time_sec)}, "
            f"classify={format_duration(result.classify_time_sec)}, "
            f"total={format_duration(result.total_time_sec)}"
        )

    avg_preprocess = sum(r.preprocess_time_sec for r in results) / len(results)
    avg_classify = sum(r.classify_time_sec for r in results) / len(results)
    avg_total = sum(r.total_time_sec for r in results) / len(results)

    print(f"Average preprocess: {format_duration(avg_preprocess)}")
    print(f"Average classify: {format_duration(avg_classify)}")
    print(f"Average total: {format_duration(avg_total)}")


if __name__ == "__main__":
    main()
