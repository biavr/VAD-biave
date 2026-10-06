# nuscenes_rerun

A small [Rerun](https://rerun.io) plugin + CLI for visualizing nuScenes data.
It ships with support for `*_lidarseg.bin` semantic-lidar label files and for
`occupancy_gt/*.npz` BEV occupancy grids, and is structured so new file types
(camera images, trajectories, detections, ...) can be added later without
touching the existing code.

## Installing rerun (already done here)

`rerun-sdk` was already `pip install`ed (version `0.38.1`, in
`/workspace/.local/lib/python3.10/site-packages`), which is what makes
`import rerun` work. The one thing that *wasn't* set up is that pip puts its
CLI binaries in `~/.local/bin`, and that directory wasn't on `$PATH` -- so
`rerun` itself wasn't runnable from a plain shell. No package was changed to
fix this, only `$PATH`:

```bash
bash install.sh
```

This appends `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc` (only if
it isn't already there) and symlinks this plugin's entry point into
`~/.local/bin`. Open a new shell (or `source ~/.bashrc`) afterwards.

Verify with:

```bash
rerun --version
```

## Usage

### As a Rerun plugin: `rerun <path>`

Once `install.sh` has been run, Rerun's own CLI picks up any file it doesn't
natively understand and hands it to this plugin automatically:

```bash
rerun /workspace/datasets/nuscenes/lidarseg/v1.0-trainval/0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin
rerun /workspace/datasets/nuscenes/v1.0-trainval/occupancy_gt/000681a060c04755a1537cf83b53ba57.npz
```

(Note: current Rerun versions take the path as a plain positional argument,
not `--file <path>` -- `rerun <path>` is the actual invocation. `--file`
doesn't exist as a flag on rerun 0.38.)

Under the hood this works because Rerun's ["external
importer"](https://www.rerun.io/docs/concepts/logging-and-ingestion/importers/overview)
mechanism scans `$PATH` for executables named `rerun-importer-<name>`. This
repo ships one: [`rerun-importer-nuscenes`](rerun-importer-nuscenes), a thin
shim that hands off to [`nuscenes_rerun/loader_entry.py`](nuscenes_rerun/loader_entry.py).
If a file doesn't match any of our loaders, it exits with Rerun's
"incompatible" status code so Rerun can try something else instead of
erroring out.

### As a standalone CLI

For direct/scripted use (custom options, saving to a file, connecting to an
already-running viewer, etc.) without depending on Rerun's plugin discovery:

```bash
# Spawn a viewer:
python3 -m nuscenes_rerun lidarseg <path_to_lidarseg.bin>

# Save to a .rrd file instead:
python3 -m nuscenes_rerun lidarseg <path_to_lidarseg.bin> --save out.rrd

# Point at a dataset root that can't be auto-detected:
python3 -m nuscenes_rerun lidarseg <path_to_lidarseg.bin> --dataroot /data/nuscenes

# BEV occupancy grid, as boxes:
python3 -m nuscenes_rerun occupancy <path_to_occupancy_gt>.npz
```

Run `python3 -m nuscenes_rerun <loader> --help` for the full option list.

## What gets logged for `lidarseg`

A `*_lidarseg.bin` file only stores one `uint8` class id per point -- it has
no xyz positions. To draw a real point cloud, the loader looks up the
matching `LIDAR_TOP` sweep through nuScenes' own `sample_data.json` (the
lidarseg filename, minus `_lidarseg.bin`, *is* that sweep's `sample_data`
token) and reads point positions from there. See
[`dataset_paths.py`](nuscenes_rerun/dataset_paths.py) for the exact lookup;
it only checks a couple of well-known candidate paths and never scans the
dataset tree, and it uses an `mmap` + byte-search lookup rather than
`json.load`-ing the whole table, since `sample_data.json` can be well over a
gigabyte on a full trainval install.

If the pairing can't be resolved (e.g. `--dataroot` wasn't enough to find
it), the loader logs a warning and falls back to laying the same labeled
points out on a flat synthetic grid, so you still see something rather than
an error.

Points are logged with `class_ids` against an `rr.AnnotationContext` built
from [`colors.py`](nuscenes_rerun/colors.py) (the official nuScenes-lidarseg
32-class palette), so the Rerun Viewer colors them and shows class names on
hover for free, at entity path `world/lidar/semantic`.

## What gets logged for `occupancy`

Files under `occupancy_gt/` (produced by
[`tools/data_converter/occupancy_gt_generator.py`](../data_converter/occupancy_gt_generator.py))
hold a 2D `class_grid` (one lidarseg class id per BEV cell, or `255` where
the cell had no lidar returns) and a same-shaped `height_grid` (the max
point height seen in that cell, or `NaN` if empty), plus `pc_range` and
`cell_size` to place cells in space.

One `rr.Boxes3D` box is logged per non-empty cell: it covers the cell's full
footprint in x/y, and runs in z from `pc_range`'s `zmin` up to that cell's
recorded height -- so the box's height directly *is* the occupancy height
for that cell. Boxes use `class_ids` against the same `colors.py` palette as
`lidarseg` (both use the same 32 nuScenes-lidarseg classes), so the two
loaders share colors/labels automatically. Empty (`255`) cells are skipped.
Logged at entity path `world/occupancy`.

## Visualizing a model's predictions over a sequence: `nuscenes_rerun.sequence`

The two loaders above only show what's already on disk (raw GT files). To
compare a trained model's live occupancy predictions against GT, across a
run of consecutive frames instead of one sample at a time, use the separate
`sequence` tool instead (it needs to build the actual model, so it doesn't
fit the "recognize this file, log it" shape the `LOADERS` registry above is
for):

```bash
cd tools/visualization
python3 -m nuscenes_rerun.sequence \
    ../../projects/configs/VAD/VAD_tiny_stage_1_with_occ.py \
    /workspace/logs/outputs_tiny_stage_1_with_occ/20261005_103643/epoch_3.pth \
    --num-frames 20 --save sequence.rrd
```

This builds the dataset + model once, then walks forward through
`--num-frames` consecutive samples (following each sample's `next` token, so
it correctly crosses the dataset's own sample ordering) logging, per frame
on a shared `frame` timeline:

- all 6 camera images, at `camera/<CAM_NAME>`
- GT occupancy (when that sample has any -- see caveat below), at
  `world/occupancy_gt`
- the model's live predicted occupancy (every frame, dense by construction),
  at `world/occupancy_pred`

GT and predicted boxes use the exact same box geometry and the same
`colors.py` palette as the `occupancy` loader above, so the two are visually
comparable, just toggle one or the other's visibility in the Viewer to
compare a cell at a time, or leave both on and look for color mismatches.

Without `--start-token`, it starts at the first sample that actually has
occupancy GT (so frame 0 isn't empty); pass one to start somewhere specific.

Open the result with `rerun sequence.rrd` (or drop `--save` to spawn a
viewer directly; on a remote workstation, use `--connect` to a viewer you've
already pointed a forwarded port at, the same way the live dashboard's port
needs forwarding).

**Caveat: occupancy GT coverage is sparse (~6% of samples in this dataset
right now, per `occupancy_gt_generator.py`'s completeness note), and not
concentrated in runs** -- across a 20-frame sequence, expect GT on maybe 1-2
frames, not every frame. The predicted occupancy is still logged every
frame regardless (it's a dense model output, not limited by GT
availability), so you can still watch it stay stable/jump around frame to
frame even without a GT overlay on most of them.

## Adding a new visualizer

1. Add a `log_<thing>(path, **opts)` function to `core.py` (or a new sibling
   module) that only calls `rr.log(...)` -- no argument parsing, no
   `rr.init`/`spawn`/`save` decisions.
2. Register it in [`nuscenes_rerun/loaders/__init__.py`](nuscenes_rerun/loaders/__init__.py):
   add a `Loader(name=..., can_load=..., log=...)` entry.

That's it: a new `python -m nuscenes_rerun <name> ...` subcommand and a new
`rerun <path>` file type both appear automatically, since the CLI
(`cli.py`) and the plugin (`loader_entry.py`) both just iterate over
`LOADERS`.

## Layout

```
nuscenes_rerun/
├── cli.py            personalized CLI (`python -m nuscenes_rerun ...`)
├── loader_entry.py   entry point used by the rerun-importer-nuscenes plugin
├── loaders/          registry: which loader handles which file
├── core.py           the actual rr.log(...) calls, one function per data kind
├── dataset_paths.py  nuScenes file lookups (lidarseg <-> LIDAR_TOP pairing)
└── colors.py         nuScenes-lidarseg 32-class id -> (name, RGB) table
rerun-importer-nuscenes  executable Rerun discovers on $PATH
install.sh               one-time PATH + symlink setup
```
