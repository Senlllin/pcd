"""Dataset generation utilities for ShapeNet-like and PCN-like structures."""

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np
import open3d as o3d


DEFAULT_CATEGORY_ID = "99999999"
DEFAULT_MODEL_ID_WIDTH = 4


@dataclass
class SplitConfiguration:
    """Holds the proportion of samples assigned to each dataset split."""

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def normalised(self) -> "SplitConfiguration":
        total = self.train + self.val + self.test
        if total <= 0:
            raise ValueError("Split proportions must sum to a positive value")
        return SplitConfiguration(self.train / total, self.val / total, self.test / total)


def create_partial_from_random_crop(
    pcd: o3d.geometry.PointCloud,
    axis: int | None = None,
    keep_range: Tuple[float, float] = (0.5, 0.8),
) -> o3d.geometry.PointCloud:
    """Generate a partial point cloud by cropping along a random axis."""

    if len(pcd.points) == 0:
        raise ValueError("Cannot create a partial cloud from an empty point cloud")

    pts = np.asarray(pcd.points)
    if axis is None:
        axis = np.random.randint(0, 3)

    minimum, maximum = pts[:, axis].min(), pts[:, axis].max()
    if np.isclose(minimum, maximum):
        # Degenerate along the chosen axis; return half the points to mimic
        # partial coverage.
        count = max(1, len(pts) // 2)
        indices = np.random.choice(len(pts), size=count, replace=False)
        partial = o3d.geometry.PointCloud()
        partial.points = o3d.utility.Vector3dVector(pts[indices])
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[indices]
            partial.colors = o3d.utility.Vector3dVector(colors)
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)[indices]
            partial.normals = o3d.utility.Vector3dVector(normals)
        return partial

    low, high = keep_range
    if not 0.0 < low <= high <= 1.0:
        raise ValueError("keep_range must define a subset within (0, 1]")

    cutoff = minimum + np.random.uniform(low, high) * (maximum - minimum)
    mask = pts[:, axis] < cutoff
    if not np.any(mask):
        mask[np.argmin(pts[:, axis])] = True
    partial = o3d.geometry.PointCloud()
    partial.points = o3d.utility.Vector3dVector(pts[mask])
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[mask]
        partial.colors = o3d.utility.Vector3dVector(colors)
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)[mask]
        partial.normals = o3d.utility.Vector3dVector(normals)
    return partial


def generate_shapenet_like_dataset(
    source_dir: pathlib.Path,
    destination_dir: pathlib.Path,
    category_id: str = DEFAULT_CATEGORY_ID,
    model_id_width: int = DEFAULT_MODEL_ID_WIDTH,
) -> List[pathlib.Path]:
    """Organise point clouds into a ShapeNet-like directory structure."""

    destination_dir = destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("*.ply"))
    if not source_files:
        raise ValueError("No .ply files found in source directory")
    created_paths: List[pathlib.Path] = []

    category_dir = destination_dir / category_id
    category_dir.mkdir(parents=True, exist_ok=True)

    for index, file_path in enumerate(source_files, start=1):
        model_id = f"{index:0{model_id_width}d}"
        model_dir = category_dir / model_id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        destination = model_dir / "model_normalized.ply"
        o3d.io.write_point_cloud(str(destination), o3d.io.read_point_cloud(str(file_path)), write_ascii=True)
        created_paths.append(destination)

    return created_paths


def _compute_split_counts(total: int, split: SplitConfiguration) -> Tuple[int, int, int]:
    normalised = split.normalised()
    train_count = math.floor(normalised.train * total)
    val_count = math.floor(normalised.val * total)
    assigned = train_count + val_count
    test_count = max(0, total - assigned)
    return train_count, val_count, test_count


def _save_point_cloud(path: pathlib.Path, pcd: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)


def _split_files(files: Sequence[pathlib.Path], split: SplitConfiguration) -> Tuple[List[pathlib.Path], List[pathlib.Path], List[pathlib.Path]]:
    files = list(files)
    np.random.shuffle(files)
    total = len(files)
    train_count, val_count, test_count = _compute_split_counts(total, split)
    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]
    return train_files, val_files, test_files


def generate_pcn_like_dataset(
    source_dir: pathlib.Path,
    destination_dir: pathlib.Path,
    split: SplitConfiguration | None = None,
    partial_generator: Callable[[o3d.geometry.PointCloud], o3d.geometry.PointCloud] = create_partial_from_random_crop,
) -> None:
    """Generate a PCN-style dataset with partial/complete splits."""

    destination_dir = destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob("*.ply"))
    if not source_files:
        raise ValueError("No .ply files found in source directory")

    split = split.normalised() if split else SplitConfiguration()
    train_files, val_files, test_files = _split_files(source_files, split)

    for split_name, files in (("train", train_files), ("val", val_files), ("test", test_files)):
        for file_path in files:
            complete = o3d.io.read_point_cloud(str(file_path))
            partial = partial_generator(complete)
            filename = file_path.name
            complete_path = destination_dir / split_name / "complete" / filename
            partial_path = destination_dir / split_name / "partial" / filename
            _save_point_cloud(complete_path, complete)
            _save_point_cloud(partial_path, partial)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ShapeNet-like and PCN-like datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("shapenet", help="Create a ShapeNet-style dataset")
    preprocess_parser.add_argument("source", type=pathlib.Path, help="Directory with processed .ply files")
    preprocess_parser.add_argument("destination", type=pathlib.Path, help="Output directory for ShapeNet dataset")
    preprocess_parser.add_argument("--category-id", type=str, default=DEFAULT_CATEGORY_ID, help="Category ID for the dataset")
    preprocess_parser.add_argument(
        "--model-id-width",
        type=int,
        default=DEFAULT_MODEL_ID_WIDTH,
        help="Zero-padding width for generated model IDs",
    )

    pcn_parser = subparsers.add_parser("pcn", help="Create a PCN-style dataset")
    pcn_parser.add_argument("source", type=pathlib.Path, help="Directory with processed .ply files")
    pcn_parser.add_argument("destination", type=pathlib.Path, help="Output directory for the PCN dataset")
    pcn_parser.add_argument("--train", type=float, default=0.8, help="Proportion of samples for the training split")
    pcn_parser.add_argument("--val", type=float, default=0.1, help="Proportion of samples for the validation split")
    pcn_parser.add_argument("--test", type=float, default=0.1, help="Proportion of samples for the test split")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.command == "shapenet":
        generate_shapenet_like_dataset(args.source, args.destination, category_id=args.category_id, model_id_width=args.model_id_width)
    elif args.command == "pcn":
        split = SplitConfiguration(train=args.train, val=args.val, test=args.test)
        generate_pcn_like_dataset(args.source, args.destination, split=split)
    else:  # pragma: no cover - defensive programming
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
