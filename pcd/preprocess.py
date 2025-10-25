"""Utilities for preprocessing point clouds prior to dataset generation.

This module implements the pipeline described in the user instructions:

1. Collect point cloud files from an input directory using multiple file
   patterns.
2. Randomly down-sample each cloud to keep 80 percent of the points.
3. Inject uniformly sampled noise worth 10 percent of the original point
   count within the bounding box of the down-sampled cloud.
4. Persist the processed clouds as ASCII `.ply` files using Open3D.
5. Rename all resulting `.ply` files to a sequential numeric scheme
   starting at 1001.

The implementation is careful to avoid name collisions during the renaming
phase and exposes a small CLI for convenience.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import open3d as o3d


DEFAULT_PATTERNS: Sequence[str] = ("*.ply", "*.pcd", "*.xyz", "*.xyzn", "*.xyzrgb")


@dataclass
class ProcessStats:
    """Aggregate statistics about processed point clouds."""

    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0

    def register_processed(self) -> None:
        self.total_files += 1
        self.processed_files += 1

    def register_skipped(self) -> None:
        self.total_files += 1
        self.skipped_files += 1


def collect_point_cloud_files(directory: str | os.PathLike[str], patterns: Sequence[str] = DEFAULT_PATTERNS) -> List[pathlib.Path]:
    """Return a sorted list of point cloud files in *directory* matching *patterns*.

    The patterns mirror those suggested in the instructions and can be
    extended by the caller if required.
    """

    dir_path = pathlib.Path(directory)
    files: List[pathlib.Path] = []
    for pattern in patterns:
        files.extend(dir_path.glob(pattern))
    # Use a stable ordering so that the renaming step is deterministic.
    return sorted(set(files))


def _downsample_point_cloud(pcd: o3d.geometry.PointCloud, sampling_ratio: float = 0.8) -> o3d.geometry.PointCloud:
    """Randomly down-sample *pcd* keeping ``sampling_ratio`` points.

    ``PointCloud.random_down_sample`` returns ``None`` when the sampling
    ratio is invalid, so we guard against that scenario. When the cloud has
    zero points we simply return a copy without modification.
    """

    if np.asarray(pcd.points).size == 0:
        return o3d.geometry.PointCloud(pcd)

    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError("sampling_ratio must be within (0, 1]")

    down_sampled = pcd.random_down_sample(sampling_ratio)
    if down_sampled is None:
        # Fall back to a manual implementation that mirrors the behavior of
        # ``random_down_sample`` when a degenerate ratio is supplied.
        points = np.asarray(pcd.points)
        count = max(1, int(round(sampling_ratio * len(points))))
        indices = np.random.choice(len(points), size=count, replace=False)
        down_sampled = o3d.geometry.PointCloud()
        down_sampled.points = o3d.utility.Vector3dVector(points[indices])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[indices]
            down_sampled.colors = o3d.utility.Vector3dVector(colors)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)[indices]
            down_sampled.normals = o3d.utility.Vector3dVector(normals)
        return down_sampled
    return down_sampled


def _generate_noise_points(base_points: np.ndarray, count: int) -> np.ndarray:
    """Generate ``count`` uniformly sampled noise points within the bounding box.

    ``base_points`` is expected to contain the down-sampled point cloud used
    as reference. When ``count`` is zero an empty array is returned.
    """

    if count <= 0:
        return np.empty((0, 3), dtype=float)

    if base_points.size == 0:
        # When the down-sampled cloud is empty we simply sample from a tiny
        # cube around the origin to avoid producing NaNs.
        return np.zeros((count, 3), dtype=float)

    min_bounds = base_points.min(axis=0)
    max_bounds = base_points.max(axis=0)
    # Handle degenerate dimensions by collapsing min/max to avoid
    # ``np.random.uniform`` raising an error when min == max.
    span = np.maximum(max_bounds - min_bounds, 1e-9)
    noise = np.empty((count, 3), dtype=float)
    for axis in range(3):
        noise[:, axis] = np.random.uniform(min_bounds[axis], min_bounds[axis] + span[axis], size=count)
    return noise


def _merge_points(original: o3d.geometry.PointCloud, noise_points: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a new point cloud containing ``original`` points and ``noise_points``."""

    merged = o3d.geometry.PointCloud(original)
    base_points = np.asarray(merged.points)
    if noise_points.size:
        all_points = np.vstack([base_points, noise_points]) if base_points.size else noise_points
        merged.points = o3d.utility.Vector3dVector(all_points)
    return merged


def _ensure_ascii_ply(pcd: o3d.geometry.PointCloud, destination: pathlib.Path) -> None:
    """Persist *pcd* as an ASCII ``.ply`` file at *destination*."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(destination), pcd, write_ascii=True)


def process_point_cloud(
    file_path: pathlib.Path,
    output_directory: pathlib.Path,
    sampling_ratio: float = 0.8,
    noise_ratio: float = 0.1,
) -> pathlib.Path:
    """Process a single point cloud file and return the resulting path.

    The processed file is saved using the pattern ``"{stem}_processed.ply"``
    inside ``output_directory``.
    """

    pcd = o3d.io.read_point_cloud(str(file_path))
    original_count = len(pcd.points)
    if original_count == 0:
        raise ValueError(f"Point cloud '{file_path}' contains no points")

    down_sampled = _downsample_point_cloud(pcd, sampling_ratio=sampling_ratio)
    retained_points = np.asarray(down_sampled.points)
    noise_count = 0 if noise_ratio <= 0 else max(1, math.ceil(noise_ratio * original_count))
    noise_points = _generate_noise_points(retained_points, noise_count)
    augmented = _merge_points(down_sampled, noise_points)

    processed_name = f"{file_path.stem}_processed.ply"
    destination = output_directory / processed_name
    _ensure_ascii_ply(augmented, destination)

    return destination


def convert_to_ascii_ply(
    source_path: pathlib.Path,
    output_directory: pathlib.Path,
) -> pathlib.Path:
    """Convert *source_path* to an ASCII `.ply` stored in *output_directory*."""

    pcd = o3d.io.read_point_cloud(str(source_path))
    destination = output_directory / f"{source_path.stem}.ply"
    _ensure_ascii_ply(pcd, destination)
    return destination


def rename_ply_files(directory: pathlib.Path, start_index: int = 1001) -> List[pathlib.Path]:
    """Rename every ``.ply`` file in *directory* to a sequential numeric scheme."""

    ply_files = sorted(directory.glob("*.ply"))
    temporary_paths: List[pathlib.Path] = []
    for idx, path in enumerate(ply_files):
        temp_path = path.with_name(f"__tmp__{idx}__{path.name}")
        path.rename(temp_path)
        temporary_paths.append(temp_path)

    final_paths: List[pathlib.Path] = []
    for offset, temp_path in enumerate(temporary_paths):
        new_name = f"{start_index + offset}.ply"
        final_path = temp_path.with_name(new_name)
        temp_path.rename(final_path)
        final_paths.append(final_path)
    return final_paths


def process_directory(
    input_dir: pathlib.Path,
    sampling_ratio: float = 0.8,
    noise_ratio: float = 0.1,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    convert_originals: bool = True,
) -> ProcessStats:
    """Process every point cloud in *input_dir* according to the pipeline.

    ``convert_originals`` controls whether non-``.ply`` files are converted to
    ASCII ``.ply`` files alongside the processed versions before the final
    renaming step.
    """

    stats = ProcessStats()
    files = collect_point_cloud_files(input_dir, patterns)
    output_dir = pathlib.Path(input_dir)

    for file_path in files:
        try:
            process_point_cloud(file_path, output_dir, sampling_ratio, noise_ratio)
            stats.register_processed()
            if convert_originals and file_path.suffix.lower() != ".ply":
                convert_to_ascii_ply(file_path, output_dir)
        except Exception as exc:  # pragma: no cover - defensive programming
            print(f"Skipping '{file_path}': {exc}")
            stats.register_skipped()

    # Rename after collecting to avoid reprocessing already renamed files.
    rename_ply_files(output_dir)

    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process point clouds by down-sampling and adding noise")
    parser.add_argument("input_dir", type=pathlib.Path, help="Directory containing point cloud files")
    parser.add_argument("--sampling-ratio", type=float, default=0.8, help="Fraction of points to retain during down-sampling")
    parser.add_argument("--noise-ratio", type=float, default=0.1, help="Fraction of original point count to add as noise")
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help="Glob patterns describing point cloud files to process",
    )
    parser.add_argument(
        "--no-convert-originals",
        action="store_true",
        help="Do not convert non-PLY originals into ASCII PLY files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> ProcessStats:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    stats = process_directory(
        args.input_dir,
        sampling_ratio=args.sampling_ratio,
        noise_ratio=args.noise_ratio,
        patterns=args.patterns,
        convert_originals=not args.no_convert_originals,
    )
    print(
        "Processed: {processed}, Skipped: {skipped}, Total: {total}".format(
            processed=stats.processed_files,
            skipped=stats.skipped_files,
            total=stats.total_files,
        )
    )
    return stats


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
