"""Locate the nuScenes files that pair with a `*_lidarseg.bin` label file.

A lidarseg ``.bin`` file only stores one class id per point (as ``uint8``);
it carries no point positions. nuScenes links each one back to a real LIDAR
sweep through ``sample_data.json``: the lidarseg filename (with the
``_lidarseg.bin`` suffix stripped) *is* the ``sample_data`` token of that
sweep, and that record's ``filename`` field points at the raw
``samples/LIDAR_TOP/*.pcd.bin`` file which holds ``(x, y, z, intensity,
ring)`` float32 rows in the same point order as the labels.

This module only ever checks a short list of well-known candidate paths --
it never walks the dataset tree looking for files, since a real nuScenes
export has hundreds of thousands of files under ``samples/`` and
``sweeps/`` and a recursive search there would be very slow.
"""

from __future__ import annotations

import json
import mmap
from dataclasses import dataclass
from pathlib import Path

LIDARSEG_SUFFIX = "_lidarseg.bin"


@dataclass
class NuScenesPointCloud:
    """A LIDAR_TOP sweep resolved for a given lidarseg label file."""

    points_path: Path
    dataroot: Path


def sample_data_token_from_lidarseg_path(lidarseg_path: Path) -> str:
    """Recover the `sample_data` token encoded in a lidarseg filename."""
    name = lidarseg_path.name
    if not name.endswith(LIDARSEG_SUFFIX):
        raise ValueError(f"not a nuScenes lidarseg file (expected *{LIDARSEG_SUFFIX}): {lidarseg_path}")
    return name[: -len(LIDARSEG_SUFFIX)]


def guess_dataroot(lidarseg_path: Path) -> Path | None:
    """Guess the dataroot from the conventional `<dataroot>/lidarseg/<version>/*_lidarseg.bin` layout."""
    parents = lidarseg_path.resolve().parents
    # parents[0] = <dataroot>/lidarseg/<version>, parents[1] = <dataroot>/lidarseg
    if len(parents) < 3 or parents[1].name != "lidarseg":
        return None
    return parents[2]


def _candidate_sample_data_tables(dataroot: Path, version: str) -> list[Path]:
    """Places `sample_data.json` may live, trying the layouts nuScenes exports use.

    Standard nuScenes devkit layout keeps it at ``<dataroot>/<version>/sample_data.json``.
    Some mirrors nest the metadata one level deeper as
    ``<dataroot>/<version>/<version>/sample_data.json`` alongside ``samples/`` and
    ``sweeps/`` in the outer ``<version>`` directory. Both are returned (in that
    order) since a real installation can have *both* -- e.g. a small,
    training-pipeline-filtered table at the outer path that doesn't happen to
    contain every ``sample_data`` token, alongside the full devkit table nested
    one level in.
    """
    return [
        candidate
        for candidate in (
            dataroot / version / "sample_data.json",
            dataroot / version / version / "sample_data.json",
        )
        if candidate.is_file() and candidate.stat().st_size > 0
    ]


def _find_record_by_token(table_path: Path, token: str) -> dict | None:
    """Look up one `{"token": ..., ...}` record in a (possibly huge) json array file.

    `sample_data.json` tables can be well over a gigabyte with millions of
    flat (non-nested) records, so this avoids `json.load`-ing the whole file
    just to find one row: it `mmap`s the file and does a byte-string search
    for the token, then decodes only the enclosing `{...}` object. This is
    safe because these tables are arrays of *flat* dicts -- no field value
    ever contains a literal `{` or `}`.
    """
    for needle in (f'"token": "{token}"'.encode(), f'"token":"{token}"'.encode()):
        with table_path.open("rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                idx = mm.find(needle)
                if idx == -1:
                    continue
                start = mm.rfind(b"{", 0, idx)
                end = mm.find(b"}", idx)
                if start == -1 or end == -1:
                    continue
                return json.loads(mm[start : end + 1])
    return None


def find_point_cloud(lidarseg_path: Path, dataroot: Path | None = None) -> NuScenesPointCloud | None:
    """Resolve the LIDAR_TOP sweep paired with a lidarseg label file, if possible.

    Returns ``None`` (rather than raising) whenever any part of the lookup
    fails, so callers can fall back to a degraded visualization instead of
    crashing on datasets that are laid out differently or incomplete.
    """
    lidarseg_path = lidarseg_path.resolve()
    if dataroot is None:
        dataroot = guess_dataroot(lidarseg_path)
    if dataroot is None or not dataroot.is_dir():
        return None

    try:
        token = sample_data_token_from_lidarseg_path(lidarseg_path)
    except ValueError:
        return None

    version = lidarseg_path.parent.name  # e.g. "v1.0-trainval"
    for table_path in _candidate_sample_data_tables(dataroot, version):
        record = _find_record_by_token(table_path, token)
        if record is None or "filename" not in record:
            continue

        # `record["filename"]` is relative to whichever directory directly
        # holds `samples/` and `sweeps/`. Try the metadata table's own
        # directory, then its parent, then the dataroot itself, covering the
        # layouts described in `_candidate_sample_data_tables`.
        for base in (table_path.parent, table_path.parent.parent, dataroot):
            points_path = base / record["filename"]
            if points_path.is_file():
                return NuScenesPointCloud(points_path=points_path, dataroot=dataroot)

    return None
