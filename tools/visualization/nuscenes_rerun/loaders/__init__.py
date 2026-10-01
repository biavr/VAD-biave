"""Registry mapping file paths to the loader that knows how to visualize them.

Both the CLI (`nuscenes_rerun.cli`) and the `rerun <path>` plugin
(`nuscenes_rerun.loader_entry`) dispatch through `LOADERS` /
`find_loader()`. To support a new kind of file:

1. Write a `log_<thing>(path, **opts)` function in `core.py` (or a new
   sibling module).
2. Add a `Loader(...)` entry below with a `can_load` predicate and a thin
   `log` wrapper that pulls the right fields off the CLI/plugin `args`
   namespace.

That's it -- a new CLI subcommand and a new `rerun <path>` file type appear
automatically, with no changes needed anywhere else.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np

from .. import core

_OCCUPANCY_REQUIRED_KEYS = {"class_grid", "height_grid", "pc_range", "cell_size"}


class Loader(NamedTuple):
    name: str
    can_load: Callable[[Path], bool]
    log: Callable[[Path, argparse.Namespace], None]


def _can_load_lidarseg(path: Path) -> bool:
    return path.is_file() and path.name.endswith("_lidarseg.bin")


def _log_lidarseg(path: Path, args: argparse.Namespace) -> None:
    dataroot = getattr(args, "dataroot", None)
    core.log_lidarseg(
        path,
        dataroot=Path(dataroot) if dataroot else None,
        entity_path_prefix=getattr(args, "entity_path_prefix", "") or "",
        static=getattr(args, "static", False),
    )


def _can_load_occupancy(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".npz":
        return False
    try:
        with np.load(path) as data:
            return _OCCUPANCY_REQUIRED_KEYS.issubset(data.files)
    except Exception:
        # Not a readable/zip-format npz, or some other unrelated .npz file.
        return False


def _log_occupancy(path: Path, args: argparse.Namespace) -> None:
    core.log_occupancy(
        path,
        entity_path_prefix=getattr(args, "entity_path_prefix", "") or "",
        static=getattr(args, "static", False),
    )


LOADERS: list[Loader] = [
    Loader(name="lidarseg", can_load=_can_load_lidarseg, log=_log_lidarseg),
    Loader(name="occupancy", can_load=_can_load_occupancy, log=_log_occupancy),
]


def find_loader(path: Path) -> Loader | None:
    """Return the first registered loader willing to handle `path`, if any."""
    return next((loader for loader in LOADERS if loader.can_load(path)), None)
