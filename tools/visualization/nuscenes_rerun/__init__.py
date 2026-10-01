"""nuscenes_rerun: Rerun (https://rerun.io) visualizers for nuScenes data.

This package is deliberately split so the same logging code can be driven
from three places without duplication:

- ``python -m nuscenes_rerun <loader> <path> [options]``   -- the CLI, see `cli.py`.
- ``rerun <path>``                                          -- via the
  `rerun-importer-nuscenes` executable next to this package, see
  `loader_entry.py`.
- directly, by importing e.g. `nuscenes_rerun.core.log_lidarseg`.

To add support for a new kind of file, add one function to (or a new module
next to) `core.py` and register it in `loaders.py`; the CLI subcommand and
the `rerun <path>` integration both pick it up automatically.
"""
