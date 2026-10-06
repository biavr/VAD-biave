"""Logging logic shared by the CLI and the `rerun-importer-nuscenes` plugin.

Every function here takes a plain path (plus plain kwargs) and only calls
`rerun.log(...)` -- it never touches `sys.argv`, spawns a viewer, or decides
where the recording goes. That is the caller's job (see `cli.py` and
`loader_entry.py`), which keeps this module usable from a notebook too.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rerun as rr

from . import dataset_paths
from .colors import annotation_context

log = logging.getLogger(__name__)

# x, y, z, intensity, ring index -- the fixed row layout of nuScenes LIDAR_TOP `.pcd.bin` sweeps.
_LIDAR_POINT_FIELDS = 5

# Sentinel used by tools/data_converter/occupancy_gt_generator.py for cells with no lidar returns.
_OCCUPANCY_IGNORE_CLASS = 255


def log_lidarseg(
    path: Path,
    *,
    dataroot: Path | None = None,
    entity_path_prefix: str = "",
    static: bool = False,
) -> None:
    """Log a nuScenes `*_lidarseg.bin` file as a semantic 3D point cloud.

    The file itself only holds one class id per point. When the matching
    LIDAR_TOP sweep can be found next to it (see
    `dataset_paths.find_point_cloud`), points are drawn at their real
    position. Otherwise this logs a warning and falls back to a flat grid
    layout, so the class labels are still visible even without geometry.
    """
    entity = f"{entity_path_prefix}/lidarseg" if entity_path_prefix else "world/lidar/semantic"

    labels = np.fromfile(path, dtype=np.uint8)
    rr.log(entity, rr.AnnotationContext(annotation_context()), static=True)

    point_cloud = dataset_paths.find_point_cloud(path, dataroot=dataroot)
    if point_cloud is None:
        log.warning(
            "could not find the LIDAR_TOP sweep paired with %s "
            "(pass --dataroot, or check the file sits inside a standard nuScenes layout); "
            "falling back to a synthetic grid with no real geometry",
            path,
        )
        rr.log(f"{entity}/_source", rr.TextLog(
            "no matching point cloud found -- showing labels on a synthetic grid, not real positions"
        ), static=True)
        positions = _synthetic_grid(len(labels))
        rr.log(entity, rr.Points3D(positions, class_ids=labels), static=static)
        return

    positions = _load_lidar_xyz(point_cloud.points_path)
    n = min(len(positions), len(labels))
    if len(positions) != len(labels):
        log.warning(
            "point/label count mismatch for %s: %d points vs %d labels; truncating to %d",
            path, len(positions), len(labels), n,
        )
    rr.log(f"{entity}/_source", rr.TextLog(f"positions from {point_cloud.points_path}"), static=True)
    rr.log(entity, rr.Points3D(positions[:n], class_ids=labels[:n]), static=static)


def occupancy_grid_to_boxes(
    class_grid: np.ndarray,
    height_grid: np.ndarray,
    pc_range,
    cell_size,
):
    """Convert a `(H, W)` class/height occupancy grid into `rr.Boxes3D` arrays.

    Shared by `log_occupancy` (reads a GT `.npz` off disk) and
    `log_occupancy_grid` (takes already-computed arrays, e.g. a live model
    prediction) so both log identically-shaped boxes and can be told apart
    only by entity path / color, not by geometry. Returns `(mins, sizes,
    class_ids)`, or `None` if every cell is `_OCCUPANCY_IGNORE_CLASS`.
    """
    xmin, ymin, zmin, _xmax, _ymax, _zmax = pc_range
    cell_x, cell_y = cell_size

    rows, cols = np.nonzero(class_grid != _OCCUPANCY_IGNORE_CLASS)
    if len(rows) == 0:
        return None

    xs = xmin + cols * cell_x
    ys = ymin + rows * cell_y
    # Cell tops are measured/predicted heights and can't be trusted to sit
    # above zmin (e.g. a below-range outlier, or an untrained model
    # predicting something wild); clip so no box gets a negative height.
    box_heights = np.clip(height_grid[rows, cols] - zmin, a_min=0.0, a_max=None)

    mins = np.stack([xs, ys, np.full_like(xs, zmin)], axis=1).astype(np.float32)
    sizes = np.stack([np.full_like(xs, cell_x), np.full_like(xs, cell_y), box_heights], axis=1).astype(np.float32)
    return mins, sizes, class_grid[rows, cols]


def log_occupancy_grid(
    entity: str,
    class_grid: np.ndarray,
    height_grid: np.ndarray,
    pc_range,
    cell_size,
    *,
    static: bool = False,
) -> bool:
    """Log an already-computed occupancy grid (GT or a model prediction) as 3D boxes at `entity`.

    Does not log an `AnnotationContext` itself (callers logging both a GT and
    a predicted grid at sibling paths should log one shared context over
    their common parent instead, since both use the same 32-class palette).
    Returns whether anything was logged (`False` if every cell was ignored).
    """
    boxes = occupancy_grid_to_boxes(class_grid, height_grid, pc_range, cell_size)
    if boxes is None:
        return False
    mins, sizes, class_ids = boxes
    rr.log(entity, rr.Boxes3D(mins=mins, sizes=sizes, class_ids=class_ids), static=static)
    return True


def log_occupancy(
    path: Path,
    *,
    entity_path_prefix: str = "",
    static: bool = False,
) -> None:
    """Log a BEV occupancy-GT `.npz` file (see `tools/data_converter/occupancy_gt_generator.py`) as 3D boxes.

    Each file holds a `(H, W)` `class_grid` (a lidarseg class id per cell, or
    255 where the cell has no lidar returns) and a same-shaped `height_grid`
    (the max point height seen in that cell's column, in meters, or NaN
    where empty), plus the `pc_range`/`cell_size` needed to place cells in
    space. One box is logged per occupied cell: it spans the cell's full
    footprint in x/y, and in z it runs from `pc_range`'s zmin up to that
    cell's recorded height -- so a box's height *is* the occupancy height
    for that cell, exactly like the generator's `height_grid` describes.
    Empty (255) cells are skipped entirely.
    """
    entity = f"{entity_path_prefix}/occupancy" if entity_path_prefix else "world/occupancy"

    with np.load(path) as data:
        class_grid = data["class_grid"]
        height_grid = data["height_grid"]
        pc_range = data["pc_range"]
        cell_size = data["cell_size"]

    rr.log(entity, rr.AnnotationContext(annotation_context()), static=True)

    if not log_occupancy_grid(entity, class_grid, height_grid, pc_range, cell_size, static=static):
        log.warning("%s has no occupied cells", path)


def _load_lidar_xyz(points_path: Path) -> np.ndarray:
    raw = np.fromfile(points_path, dtype=np.float32).reshape(-1, _LIDAR_POINT_FIELDS)
    return np.ascontiguousarray(raw[:, :3])


def _synthetic_grid(n: int, spacing: float = 0.2) -> np.ndarray:
    """Arrange `n` points on a flat square grid, purely so labels remain visible without geometry."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
    grid = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side)], axis=1)
    return (grid[:n] * spacing).astype(np.float32)
