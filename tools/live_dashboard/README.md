# Live training dashboard

A standalone local web page that shows live-updating loss curves for an
in-progress VAD training run.

It works by reading the same `vis_data/scalars.json` file mmengine's own
logger already writes to during training, polling it every few seconds. It
is **not** wired into `train_new.py`, any config, or the training loop in
any way — it's a separate process that only reads files training already
produces. That means:

- No config or pipeline changes were needed to add it, and none are needed
  to use it.
- It works with every existing config (base, tiny, tiny+occ, anything
  future) unmodified, since they all go through the same mmengine logger.
- Starting or stopping it has zero effect on training. You can launch it
  before training starts, long after it's finished, or not at all.

## Usage

Start training as usual, in its own terminal:

```
CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29500 tools/train_new.py projects/configs/VAD/VAD_tiny_stage_1_updated.py --launcher pytorch
```

In a **second** terminal, start the dashboard, pointed at that config's
`work_dir` (the `work_dir = '...'` line at the bottom of the config file):

```
python tools/live_dashboard/server.py --work-dir /workspace/logs/outputs_tiny_stage_1
```

It prints something like:

```
Watching /workspace/logs/outputs_tiny_stage_1
Live dashboard: http://localhost:8787
Ctrl+C to stop.
```

Open that URL in a browser. On a remote workstation accessed through VS
Code, the Ports panel will usually offer to forward it automatically; if
not, forward it manually (`ssh -L 8787:localhost:8787 ...`).

Leave it running — it keeps polling until you `Ctrl+C` it. The two
processes are independent, so order doesn't matter and either can be
restarted without touching the other.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--work-dir` | *(required)* | A config's `work_dir`. The dashboard watches this exact directory. |
| `--port` | `8787` | Local port to serve on. Change it if that port is taken, or to run more than one dashboard at once (e.g. watching base and tiny in parallel). |

## What it shows

- **Total loss**, as a large chart, with dashed vertical lines marking
  epoch boundaries and green dots marking each epoch whose checkpoint
  (`epoch_N.pth`) has actually been saved to disk.
- A grid of smaller charts for the individual loss terms: detection
  (`loss_cls`, `loss_bbox`), agent trajectory (`loss_traj`,
  `loss_traj_cls`), map (`loss_map_cls`, `loss_map_pts`, `loss_map_dir`),
  the occupancy head if present (`loss_occ_cls`, `loss_occ_height`), and
  `grad_norm` as a sanity check for training stability.
- Terms that are pinned at exactly `0.0` by design for the current stage
  (`loss_map_bbox`, `loss_map_iou`, every `loss_plan_*` term in stage 1) are
  left out — there's nothing to show.

Hover any chart for the exact value and step at that point.

### How it picks which run to show

mmengine creates a new timestamped folder (e.g. `20261004_112917/`) under
`work_dir` every time you launch training. The dashboard always shows the
**newest** one it finds there, checked fresh on every poll. If you stop
training and relaunch it under the same `work_dir`, the dashboard switches
to the new run automatically — no restart needed.

Checkpoint files are searched for anywhere under `work_dir` (including
subfolders), since it's common to move `epoch_*.pth` files around to keep
them organized.

## What this is not

This only shows the *loss values* the model is optimizing. It cannot tell
you whether predicted boxes, map vectors, or trajectories are actually
good — that needs either a working evaluation metric (not currently wired
up for this custom VAD setup; see the stage-1 config comments) or visual
inspection of predictions, which is a separate, heavier task than this
lightweight dashboard.

## Troubleshooting

- **"Waiting for the first logged iteration…"** — either training hasn't
  reached its first logging interval yet (default: 50 iterations), or
  `--work-dir` doesn't match the config's actual `work_dir`, or no
  timestamped run folder exists there yet.
- **"connection lost, retrying…"** — the dashboard server process stopped
  or crashed. Restart it; it'll pick back up from wherever the log file
  currently is.
- **Port already in use** — pass a different `--port`.
