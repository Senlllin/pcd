# Point Cloud Dataset Utilities

This repository provides two command line tools for preparing point cloud data:

- `pcd.preprocess`: preprocesses raw point clouds by down-sampling, adding
  uniform noise, and renaming all outputs to a numeric sequence beginning at
  `1001.ply`.
- `pcd.datasets`: organises processed point clouds into ShapeNet-like and
  PCN-like directory structures.

## Requirements

The scripts require Python 3.10+ with the following packages available:

- [Open3D](http://www.open3d.org/)
- NumPy

## Preprocessing workflow

```bash
python -m pcd.preprocess /path/to/pointcloud/folder \
    --sampling-ratio 0.8 \
    --noise-ratio 0.1
```

The command processes every file matching the default extensions (`*.ply`,
`*.pcd`, `*.xyz`, `*.xyzn`, `*.xyzrgb`). Each cloud is down-sampled to retain
approximately 80% of its points, supplemented with uniformly sampled noise
worth 10% of the original point count, and saved as ASCII `.ply`. Afterwards
all `.ply` files within the directory are renamed sequentially starting at
`1001.ply`.

## Dataset generation

ShapeNet-style dataset:

```bash
python -m pcd.datasets shapenet /path/to/processed/folder /output/ShapeNet \
    --category-id 99999999 --model-id-width 4
```

PCN-style dataset:

```bash
python -m pcd.datasets pcn /path/to/processed/folder /output/PCN \
    --train 0.8 --val 0.1 --test 0.1
```

The PCN command automatically generates partial point clouds by cropping each
model along a random axis and stores paired complete/partial samples for the
train/validation/test splits.
