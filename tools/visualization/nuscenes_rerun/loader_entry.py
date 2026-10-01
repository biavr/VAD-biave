"""Entry point for `rerun-importer-nuscenes`, an external importer plugin.

Rerun invokes any executable on `$PATH` named `rerun-importer-<name>` (or
the deprecated `rerun-loader-<name>`) with the path of a file it doesn't
natively understand, plus a handful of recommended-setting flags. See:
https://www.rerun.io/docs/concepts/logging-and-ingestion/importers/overview

If none of `nuscenes_rerun.loaders.LOADERS` recognize the file, this exits
with `rr.EXTERNAL_IMPORTER_INCOMPATIBLE_EXIT_CODE` so Rerun moves on to
other loaders (or its own built-ins) instead of reporting an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rerun as rr

from .loaders import find_loader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str)
    parser.add_argument("--application-id", type=str, default=None, help="suggested application id")
    parser.add_argument("--opened-application-id", type=str, default=None, help="currently open application id, if any")
    parser.add_argument("--recording-id", type=str, default=None, help="suggested recording id")
    parser.add_argument("--opened-recording-id", type=str, default=None, help="currently open recording id, if any")
    parser.add_argument("--entity-path-prefix", type=str, default="", help="prefix for all logged entity paths")
    parser.add_argument("--static", action="store_true", default=False, help="log data as static")
    parser.add_argument("--time_sequence", type=str, action="append", default=[], help="e.g. sim_frame=42 (repeatable)")
    parser.add_argument("--time_duration_nanos", type=str, action="append", default=[], help="e.g. sim_time=123 (repeatable)")
    parser.add_argument("--time_timestamp_nanos", type=str, action="append", default=[], help="e.g. sim_time=1709203426123456789 (repeatable)")
    # Our own extension: rerun's spec doesn't define this, but nothing stops
    # a loader from accepting extra flags of its own.
    parser.add_argument("--dataroot", type=str, default=None, help="nuScenes dataset root, if it can't be auto-detected")
    return parser


def _set_time_from_args(args: argparse.Namespace) -> None:
    for entry in args.time_sequence:
        timeline, _, value = entry.partition("=")
        if value:
            rr.set_time(timeline, sequence=int(value))
    for entry in args.time_duration_nanos:
        timeline, _, value = entry.partition("=")
        if value:
            rr.set_time(timeline, duration=1e-9 * int(value))
    for entry in args.time_timestamp_nanos:
        timeline, _, value = entry.partition("=")
        if value:
            rr.set_time(timeline, timestamp=1e-9 * int(value))


def main() -> None:
    args = build_parser().parse_args()
    path = Path(args.path)

    loader = find_loader(path)
    if loader is None:
        # Not one of ours: let Rerun try other loaders / its own built-ins.
        sys.exit(rr.EXTERNAL_IMPORTER_INCOMPATIBLE_EXIT_CODE)

    rr.init(args.application_id or "nuscenes_rerun", recording_id=args.recording_id)
    rr.stdout()

    if not args.static:
        _set_time_from_args(args)

    loader.log(path, args)


if __name__ == "__main__":
    main()
