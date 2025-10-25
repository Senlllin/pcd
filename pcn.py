#!/usr/bin/env python3
"""PCN-style dataset builder with configurable category metadata."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import open3d as o3d


DEFAULT_INPUT_DIR = Path(r"C:\Users\zccy2\Desktop\1")
DEFAULT_CATEGORY_ID = "02691156"
DEFAULT_CATEGORY_NAME = "dougong"
MIN_VALID_POINTS = 10
MIN_PARTIAL_RATIO = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PCN-style dataset with partial point clouds."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw point clouds (non-recursive).",
    )
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory where the PCN dataset root will be created.")
    parser.add_argument("--exts", nargs="*", default=(".pcd", ".ply", ".xyz"),
                        help="Accepted input file extensions (with leading dots).")
    parser.add_argument("--out_ext", default=".pcd",
                        choices=(".pcd", ".ply"),
                        help="Output point cloud format.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_partials", type=int, default=8,
                        help="Number of partial point clouds to generate per model.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search for point clouds under the input directory.",
    )
    parser.add_argument("--category_id", default=DEFAULT_CATEGORY_ID,
                        help="Identifier written to category.txt and metadata.")
    parser.add_argument("--category_name", default=DEFAULT_CATEGORY_NAME,
                        help="Category name written to category.txt and metadata.")
    return parser.parse_args()


def load_cloud(path: Path) -> np.ndarray | None:
    try:
        pcd = o3d.io.read_point_cloud(str(path))
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return None
    if pcd.is_empty():
        print(f"[WARN] Empty point cloud skipped: {path}", file=sys.stderr)
        return None
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print(f"[WARN] Empty point cloud skipped: {path}", file=sys.stderr)
        return None
    mask = np.all(np.isfinite(pts), axis=1)
    if not np.all(mask):
        pts = pts[mask]
    if pts.shape[0] < MIN_VALID_POINTS:
        print(f"[WARN] Too few valid points after cleaning ({pts.shape[0]}): {path}", file=sys.stderr)
        return None
    return pts


def normalize_unit_sphere(pts: np.ndarray) -> np.ndarray:
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    norms = np.linalg.norm(centered, axis=1)
    max_norm = float(norms.max(initial=0.0))
    if max_norm < 1e-12:
        scale = 1.0
    else:
        scale = max_norm
    normalized = centered / scale
    # Optional light deduplication by rounding.
    rounded = np.round(normalized, decimals=6)
    unique, unique_idx = np.unique(rounded, axis=0, return_index=True)
    normalized = normalized[sorted(unique_idx)]
    return normalized.astype(np.float32, copy=False)


def stable_model_id(path: Path, pts: np.ndarray) -> str:
    md5 = hashlib.md5()
    md5.update(path.stem.encode("utf-8", errors="ignore"))
    md5.update(pts.astype(np.float32).tobytes())
    return md5.hexdigest()


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(math.floor(0.70 * n))
    n_test = int(math.floor(0.25 * n))
    n_val = n - n_train - n_test
    train_idx = perm[:n_train]
    test_idx = perm[n_train:n_train + n_test]
    val_idx = perm[n_train + n_test:]
    return train_idx, test_idx, val_idx


def generate_partials(pts: np.ndarray, k: int, rng: np.random.Generator) -> List[np.ndarray]:
    if k <= 0:
        return []
    min_points = max(int(math.ceil(pts.shape[0] * MIN_PARTIAL_RATIO)), MIN_VALID_POINTS)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    span = bbox_max - bbox_min
    centroid = pts.mean(axis=0)

    partials: List[np.ndarray] = []

    def try_strategy(strategy_func) -> np.ndarray:
        for _ in range(5):
            mask = strategy_func()
            candidate = pts[mask]
            if candidate.shape[0] >= min_points:
                return candidate
        # fallback to random dropout retaining at least min_points
        count = max(min_points, int(pts.shape[0] * 0.5))
        count = min(count, pts.shape[0])
        idx = rng.choice(pts.shape[0], size=count, replace=False)
        return pts[np.sort(idx)]

    def random_half_space(keep_lower: bool) -> np.ndarray:
        def strategy():
            normal = rng.normal(size=3)
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                normal = np.array([1.0, 0.0, 0.0])
            else:
                normal /= norm
            dots = pts @ normal
            min_d, max_d = dots.min(), dots.max()
            thresh = rng.uniform(min_d + 0.3 * (max_d - min_d), min_d + 0.7 * (max_d - min_d))
            if keep_lower:
                mask = dots <= thresh
            else:
                mask = dots >= thresh
            return mask
        return try_strategy(strategy)

    def axis_crop(axis: int, keep_lower: bool) -> np.ndarray:
        def strategy():
            axis_vals = pts[:, axis]
            min_v, max_v = axis_vals.min(), axis_vals.max()
            cut = rng.uniform(min_v + 0.4 * (max_v - min_v), min_v + 0.7 * (max_v - min_v))
            if keep_lower:
                mask = axis_vals <= cut
            else:
                mask = axis_vals >= cut
            return mask
        return try_strategy(strategy)

    def axis_slab(axis: int) -> np.ndarray:
        def strategy():
            axis_vals = pts[:, axis]
            min_v, max_v = axis_vals.min(), axis_vals.max()
            alpha = rng.uniform(0.0, 0.3)
            beta = rng.uniform(0.7, 1.0)
            lower = min_v + alpha * (max_v - min_v)
            upper = min_v + beta * (max_v - min_v)
            mask = (axis_vals >= lower) & (axis_vals <= upper)
            return mask
        return try_strategy(strategy)

    def spherical_occlusion() -> np.ndarray:
        def strategy():
            center = rng.uniform(bbox_min, bbox_max)
            radius = rng.uniform(0.3, 0.6) * float(np.linalg.norm(span) + 1e-6)
            dist = np.linalg.norm(pts - center, axis=1)
            mask = dist >= radius
            return mask
        return try_strategy(strategy)

    def cone_occlusion() -> np.ndarray:
        def strategy():
            apex = centroid + rng.normal(size=3) * 0.3
            direction = rng.normal(size=3)
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                direction = np.array([0.0, 0.0, 1.0])
                norm = 1.0
            direction /= norm
            angle = rng.uniform(math.radians(20), math.radians(45))
            cos_theta = math.cos(angle)
            vectors = pts - apex
            dist = np.linalg.norm(vectors, axis=1)
            dot = vectors @ direction
            with np.errstate(divide='ignore', invalid='ignore'):
                cos_vals = np.divide(dot, dist, out=np.zeros_like(dot), where=dist > 1e-12)
            mask_remove = (dot > 0) & (cos_vals >= cos_theta)
            mask = ~mask_remove
            return mask
        return try_strategy(strategy)

    def random_dropout() -> np.ndarray:
        def strategy():
            ratio = rng.uniform(0.5, 0.7)
            count = max(min_points, int(pts.shape[0] * ratio))
            count = min(count, pts.shape[0])
            idx = rng.choice(pts.shape[0], size=count, replace=False)
            mask = np.zeros(pts.shape[0], dtype=bool)
            mask[idx] = True
            return mask
        return try_strategy(strategy)

    strategies = [
        lambda: random_half_space(True),
        lambda: random_half_space(False),
        lambda: axis_crop(0, True),
        lambda: axis_crop(1, False),
        lambda: axis_slab(2),
        spherical_occlusion,
        cone_occlusion,
        random_dropout,
    ]

    for i in range(min(k, len(strategies))):
        partials.append(strategies[i]())

    while len(partials) < k:
        partials.append(random_dropout())

    return partials[:k]


def save_cloud(path: Path, pts: np.ndarray, out_ext: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if out_ext == ".ply":
        success = o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)
    else:
        success = o3d.io.write_point_cloud(str(path), pcd)
    if not success:
        raise RuntimeError(f"Failed to write point cloud: {path}")


def list_point_clouds(directory: Path, exts: Sequence[str], recursive: bool) -> List[Path]:
    normalized_exts = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in exts}
    files = []
    iterator = directory.rglob('*') if recursive else directory.iterdir()
    for entry in sorted(iterator):
        if entry.is_file() and entry.suffix.lower() in normalized_exts:
            files.append(entry)
    return files


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    exts: Sequence[str] = args.exts or (".pcd", ".ply", ".xyz")
    out_ext: str = args.out_ext
    seed: int = args.seed
    num_partials: int = args.num_partials
    recursive: bool = args.recursive
    category_id: str = args.category_id
    category_name: str = args.category_name

    if num_partials <= 0:
        print("[WARN] num_partials must be positive; defaulting to 8.")
        num_partials = 8

    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    files = list_point_clouds(input_dir, exts, recursive)
    if not files:
        print(f"No input point clouds found in {input_dir} with extensions {exts}", file=sys.stderr)

    processed = []
    for path in files:
        pts = load_cloud(path)
        if pts is None:
            continue
        normalized = normalize_unit_sphere(pts)
        if normalized.shape[0] < MIN_VALID_POINTS:
            print(f"[WARN] Skipping {path} after normalization: too few points", file=sys.stderr)
            continue
        model_id = stable_model_id(path, normalized)
        processed.append({
            "model_id": model_id,
            "points": normalized,
            "source": path,
        })

    processed.sort(key=lambda item: item["model_id"])
    total = len(processed)

    train_idx, test_idx, val_idx = split_indices(total, seed)

    splits = {
        "train": train_idx,
        "test": test_idx,
        "val": val_idx,
    }

    pcn_root = output_dir / "PCN"
    complete_root = {
        split: pcn_root / split / "complete" / category_id for split in splits
    }
    partial_root = {
        split: pcn_root / split / "partial" / category_id for split in splits
    }

    for path in complete_root.values():
        path.mkdir(parents=True, exist_ok=True)
    # Partial directories will be created per model.

    split_model_ids: dict[str, List[str]] = {split: [] for split in splits}

    for split, indices in splits.items():
        for idx in indices:
            entry = processed[int(idx)]
            model_id = entry["model_id"]
            pts = entry["points"]
            split_model_ids[split].append(model_id)

            model_rng_seed = int(model_id[:16], 16) ^ seed
            model_rng = np.random.default_rng(model_rng_seed)
            partials = generate_partials(pts, num_partials, model_rng)

            complete_path = complete_root[split] / f"{model_id}{out_ext}"
            save_cloud(complete_path, pts, out_ext)

            partial_dir = partial_root[split] / model_id
            for i, part in enumerate(partials):
                partial_path = partial_dir / f"{i:02d}{out_ext}"
                save_cloud(partial_path, part, out_ext)

    # category.txt
    category_txt_path = pcn_root / "category.txt"
    category_txt_path.parent.mkdir(parents=True, exist_ok=True)
    category_txt_path.write_text(f"{category_id} {category_name}\n", encoding="utf-8")

    # PCN.json summary
    summary = {
        "category": {"id": category_id, "name": category_name},
        "seed": seed,
        "out_ext": out_ext,
        "num_partials": num_partials,
        "counts": {split: len(indices) for split, indices in split_model_ids.items()},
        "splits": split_model_ids,
    }
    json_path = pcn_root / "PCN.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Built PCN at {pcn_root}")
    print(f"Category: {category_id} ({category_name})")
    print(
        "Splits: "
        f"train={len(split_model_ids['train'])}, "
        f"test={len(split_model_ids['test'])}, "
        f"val={len(split_model_ids['val'])}"
    )
    print(f"Partials per model: {num_partials}")
    print(f"Format: {out_ext}")


if __name__ == "__main__":
    main()
