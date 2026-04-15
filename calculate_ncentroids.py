from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute AnomalyCLIP ncentroid from feature .npy files listed in an annotation file."
    )
    parser.add_argument(
        "--feat-root",
        required=True,
        help="Root directory with feature files referenced by annotation paths.",
    )
    parser.add_argument(
        "--ann-path",
        required=True,
        help="Annotation file path (e.g. Anomaly_Train_Normal.txt).",
    )
    parser.add_argument(
        "--out-path",
        required=True,
        help="Where to save output centroid tensor (.pt).",
    )
    return parser


def build_feature_index(feat_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for p in feat_root.rglob("*.npy"):
        index.setdefault(p.name, []).append(p.resolve())
    return index


def candidate_paths(feat_root: Path, raw_ref: str) -> list[Path]:
    raw_ref = raw_ref.strip()
    cands: list[Path] = []

    def add(rel_or_abs: str) -> None:
        p = Path(rel_or_abs)
        resolved = p.resolve() if p.is_absolute() else (feat_root / p).resolve()
        if resolved not in cands:
            cands.append(resolved)

    add(raw_ref)

    if not raw_ref.lower().endswith(".npy"):
        add(f"{raw_ref}.npy")

    raw_lower = raw_ref.lower()
    for ext in VIDEO_EXTS:
        if raw_lower.endswith(ext):
            add(f"{raw_ref[: -len(ext)]}.npy")
            break

    return cands


def resolve_feature_path(
    *,
    feat_root: Path,
    raw_ref: str,
    file_index: dict[str, list[Path]],
    line_idx: int,
) -> Path:
    cands = candidate_paths(feat_root=feat_root, raw_ref=raw_ref)
    for p in cands:
        if p.is_file():
            return p

    # Fallback for datasets like XD-Violence where annotation names can contain many dots:
    # try by basename among all feature files.
    baseline_name = Path(cands[-1]).name if cands else Path(raw_ref).name
    matches = file_index.get(baseline_name, [])

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        cls_hint = Path(raw_ref).parts[0].lower() if Path(raw_ref).parts else ""
        hinted = [m for m in matches if m.parent.name.lower() == cls_hint]
        if len(hinted) == 1:
            return hinted[0]
        raise FileNotFoundError(
            f"Ambiguous feature match for annotation line {line_idx}: {raw_ref}. "
            f"Candidates: {matches[:5]}"
        )

    raise FileNotFoundError(
        f"Missing feature file referenced in annotation line {line_idx}: {raw_ref}"
    )


def iter_feature_paths(feat_root: Path, ann_path: Path, file_index: dict[str, list[Path]]):
    for line_idx, raw_line in enumerate(ann_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        rel = line.split()[0]
        feat_path = resolve_feature_path(
            feat_root=feat_root,
            raw_ref=rel,
            file_index=file_index,
            line_idx=line_idx,
        )
        yield line_idx, feat_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    feat_root = Path(args.feat_root).expanduser().resolve()
    ann_path = Path(args.ann_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()

    if not feat_root.exists():
        raise FileNotFoundError(f"Feature root not found: {feat_root}")
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    file_index = build_feature_index(feat_root)

    running_sum: np.ndarray | None = None
    total_frames = 0
    feature_dim: int | None = None
    file_count = 0

    for _, feat_path in iter_feature_paths(feat_root=feat_root, ann_path=ann_path, file_index=file_index):
        arr = np.load(feat_path, allow_pickle=True)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim < 2:
            raise ValueError(
                f"Expected at least 2D feature array in {feat_path}, got shape {arr.shape}"
            )

        flat = arr.reshape(-1, arr.shape[-1])
        if feature_dim is None:
            feature_dim = int(flat.shape[1])
            running_sum = np.zeros(feature_dim, dtype=np.float64)
        elif flat.shape[1] != feature_dim:
            raise ValueError(
                f"Inconsistent feature dim in {feat_path}: got {flat.shape[1]}, expected {feature_dim}"
            )

        running_sum += flat.sum(axis=0, dtype=np.float64)
        total_frames += int(flat.shape[0])
        file_count += 1

    if running_sum is None or feature_dim is None or total_frames == 0:
        raise RuntimeError("No feature vectors were loaded. Check annotation content and feature paths.")

    centroid_np = (running_sum / float(total_frames)).astype(np.float32)
    centroid = torch.from_numpy(centroid_np)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(centroid, out_path)

    print(f"Saved ncentroid to: {out_path}")
    print(f"Files used: {file_count}")
    print(f"Frames used: {total_frames}")
    print(f"Feature dim: {feature_dim}")
    print(f"Centroid norm (L2): {float(torch.norm(centroid)):.6f}")


if __name__ == "__main__":
    main()
