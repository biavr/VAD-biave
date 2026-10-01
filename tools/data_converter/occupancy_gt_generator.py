"""Generate 2.5D BEV occupancy ground truth from nuScenes-lidarseg.

For every keyframe sample, this rasterizes the labelled LIDAR_TOP point
cloud into a bird's-eye-view grid. Each (x, y) cell stores:
  - class_grid[y, x]:  the majority lidarseg class among the points that
                        fall in that cell's vertical column (uint8, 0-31 per
                        the official nuScenes-lidarseg category indices, or
                        IGNORE_CLASS=255 if the cell has no points).
  - height_grid[y, x]: the max point height (z, meters, in the LIDAR_TOP
                        sensor frame) in that column, or NaN if empty.

Points and grids live in the ego vehicle's LIDAR_TOP sensor frame, matching
the `box_type_3d='LiDAR'` convention already used by VADCustomNuScenesDataset,
so a cell at grid[i, j] and a VAD detection/map target at the same (x, y)
describe the same physical location.

Why this script doesn't use `nusc.lidarseg*`:
The devkit only loads lidarseg metadata when `{version}/lidarseg.json` and a
`category.json` with an `index` field are present. Neither exists in this
dataset root (only the plain object-detection `category.json` and the raw
`lidarseg/{version}/*.bin` label files were installed), so `NuScenes.lidarseg`
is unavailable here. Instead we read the `.bin` label files directly by their
standard naming convention (`{lidar_sample_data_token}_lidarseg.bin`) and use
the hardcoded, version-independent 32-class index table below -- this table
has been fixed since the nuScenes-lidarseg release and does not depend on the
missing metadata files.

Usage:
    python tools/data_converter/occupancy_gt_generator.py \\
        --data-root /workspace/datasets/nuscenes/v1.0-trainval \\
        --version v1.0-trainval \\
        --out-dir /workspace/datasets/nuscenes/v1.0-trainval/occupancy_gt
"""
import argparse
import os
from os import path as osp

import numpy as np
from mmengine.utils import track_parallel_progress, track_iter_progress
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

# Official nuScenes-lidarseg category table (index -> name), fixed since the
# lidarseg extension's release. Index 0 ("noise") marks spurious/unlabelled
# lidar returns; it is treated like any other class when voting for a cell's
# majority label (see module docstring).
LIDARSEG_IDX2NAME = {
    0: 'noise',
    1: 'animal',
    2: 'human.pedestrian.adult',
    3: 'human.pedestrian.child',
    4: 'human.pedestrian.construction_worker',
    5: 'human.pedestrian.personal_mobility',
    6: 'human.pedestrian.police_officer',
    7: 'human.pedestrian.stroller',
    8: 'human.pedestrian.wheelchair',
    9: 'movable_object.barrier',
    10: 'movable_object.debris',
    11: 'movable_object.pushable_pullable',
    12: 'movable_object.trafficcone',
    13: 'static_object.bicycle_rack',
    14: 'vehicle.bicycle',
    15: 'vehicle.bus.bendy',
    16: 'vehicle.bus.rigid',
    17: 'vehicle.car',
    18: 'vehicle.construction',
    19: 'vehicle.emergency.ambulance',
    20: 'vehicle.emergency.police',
    21: 'vehicle.motorcycle',
    22: 'vehicle.trailer',
    23: 'vehicle.truck',
    24: 'flat.driveable_surface',
    25: 'flat.other',
    26: 'flat.sidewalk',
    27: 'flat.terrain',
    28: 'static.manmade',
    29: 'static.other',
    30: 'static.vegetation',
    31: 'vehicle.ego',
}
NUM_LIDARSEG_CLASSES = len(LIDARSEG_IDX2NAME)  # 32
IGNORE_CLASS = 255  # sentinel for cells with no lidar returns


def lidarseg_bin_path(data_root: str, version: str, lidar_sd_token: str) -> str:
    """Path of a keyframe's per-point label file, by nuScenes' fixed naming."""
    return osp.join(data_root, 'lidarseg', version, f'{lidar_sd_token}_lidarseg.bin')


def load_lidarseg_labels(bin_path: str) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.uint8)


def voxelize_bev(points_xyz: np.ndarray,
                  labels: np.ndarray,
                  pc_range,
                  cell_size):
    """Rasterize a labelled point cloud into a class+height BEV grid.

    Args:
        points_xyz: (N, 3) float array, LIDAR_TOP sensor frame.
        labels: (N,) uint8 array, per-point lidarseg class index.
        pc_range: (xmin, ymin, zmin, xmax, ymax, zmax).
        cell_size: (cell_x, cell_y) in meters.

    Returns:
        class_grid: (H, W) uint8, IGNORE_CLASS where no points fall.
        height_grid: (H, W) float32, NaN where no points fall.
        Row i corresponds to y in [ymin + i*cell_y, ymin + (i+1)*cell_y);
        column j corresponds to x in [xmin + j*cell_x, xmin + (j+1)*cell_x).
    """
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    cell_x, cell_y = cell_size
    W = int(round((xmax - xmin) / cell_x))
    H = int(round((ymax - ymin) / cell_y))

    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    in_range = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) & \
               (z >= zmin) & (z < zmax)
    x, y, z, labels = x[in_range], y[in_range], z[in_range], labels[in_range]

    class_grid = np.full((H, W), IGNORE_CLASS, dtype=np.uint8)
    height_grid = np.full((H, W), np.nan, dtype=np.float32)
    if x.size == 0:
        return class_grid, height_grid

    col = np.floor((x - xmin) / cell_x).astype(np.int64).clip(0, W - 1)
    row = np.floor((y - ymin) / cell_y).astype(np.int64).clip(0, H - 1)
    flat_idx = row * W + col
    num_cells = H * W

    # Max height per occupied cell.
    height_flat = np.full(num_cells, -np.inf, dtype=np.float32)
    np.maximum.at(height_flat, flat_idx, z.astype(np.float32))
    occupied = np.isfinite(height_flat)
    height_flat[~occupied] = np.nan

    # Majority class per occupied cell: tally per-cell, per-class counts,
    # then take the argmax class for each occupied cell.
    counts = np.zeros((num_cells, NUM_LIDARSEG_CLASSES), dtype=np.int32)
    np.add.at(counts, (flat_idx, labels.astype(np.int64)), 1)
    class_flat = np.full(num_cells, IGNORE_CLASS, dtype=np.uint8)
    class_flat[occupied] = counts[occupied].argmax(axis=1).astype(np.uint8)

    return class_flat.reshape(H, W), height_flat.reshape(H, W)


def _process_one(args):
    (data_root, version, lidar_sd_token, lidar_path, out_path,
     pc_range, cell_size) = args
    bin_path = lidarseg_bin_path(data_root, version, lidar_sd_token)
    if not osp.exists(bin_path):
        return ('missing_labels', lidar_sd_token)

    labels = load_lidarseg_labels(bin_path)
    points = LidarPointCloud.from_file(lidar_path).points[:3, :].T  # (N, 3)
    if points.shape[0] != labels.shape[0]:
        return ('length_mismatch', lidar_sd_token)

    class_grid, height_grid = voxelize_bev(points, labels, pc_range, cell_size)
    np.savez_compressed(
        out_path,
        class_grid=class_grid,
        height_grid=height_grid,
        pc_range=np.array(pc_range, dtype=np.float32),
        cell_size=np.array(cell_size, dtype=np.float32),
    )
    return ('ok', lidar_sd_token)


def generate_occupancy_gt(data_root: str,
                          version: str,
                          out_dir: str,
                          pc_range,
                          cell_size,
                          nproc: int = 1):
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for sample in nusc.sample:
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_path = nusc.get_sample_data_path(lidar_sd_token)
        out_path = osp.join(out_dir, f"{sample['token']}.npz")
        tasks.append((data_root, version, lidar_sd_token, lidar_path,
                      out_path, pc_range, cell_size))

    print(f'Generating occupancy GT for {len(tasks)} keyframes -> {out_dir}')
    if nproc > 1:
        results = track_parallel_progress(_process_one, tasks, nproc=nproc)
    else:
        results = [_process_one(t) for t in track_iter_progress(tasks)]

    status_counts = {}
    problems = []
    for status, token in results:
        status_counts[status] = status_counts.get(status, 0) + 1
        if status != 'ok':
            problems.append((status, token))

    print('Done:', status_counts)
    if problems:
        print(f'{len(problems)} keyframes were skipped, e.g.:')
        for status, token in problems[:10]:
            print(f'  [{status}] lidar sample_data token={token}')


def parse_args():
    parser = argparse.ArgumentParser(description='Generate BEV occupancy GT from nuScenes-lidarseg')
    parser.add_argument('--data-root', type=str,
                        default='/workspace/datasets/nuscenes/v1.0-trainval')
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Defaults to <data-root>/occupancy_gt')
    parser.add_argument('--pc-range', type=float, nargs=6,
                        default=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                        metavar=('XMIN', 'YMIN', 'ZMIN', 'XMAX', 'YMAX', 'ZMAX'),
                        help='Matches VAD_base_stage_1_updated.py point_cloud_range by default')
    parser.add_argument('--cell-size', type=float, nargs=2, default=[0.5, 0.5],
                        metavar=('CELL_X', 'CELL_Y'))
    parser.add_argument('--nproc', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out_dir = args.out_dir or osp.join(args.data_root, 'occupancy_gt')
    generate_occupancy_gt(
        data_root=args.data_root,
        version=args.version,
        out_dir=out_dir,
        pc_range=args.pc_range,
        cell_size=args.cell_size,
        nproc=args.nproc,
    )
