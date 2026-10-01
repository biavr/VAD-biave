"""Personalized CLI for the nuScenes Rerun visualizers.

Examples
--------
Spawn a viewer showing the semantic point cloud for one lidarseg file::

    python -m nuscenes_rerun lidarseg \\
        /workspace/datasets/nuscenes/lidarseg/v1.0-trainval/0a0c9ff1674645fdab2cf6d7308b9269_lidarseg.bin

Save the recording instead of opening a viewer::

    python -m nuscenes_rerun lidarseg <path> --save out.rrd

Every entry in `nuscenes_rerun.loaders.LOADERS` becomes its own subcommand
automatically -- adding a new loader (camera images, trajectories, ...)
does not require touching this file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rerun as rr

from .loaders import LOADERS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nuscenes_rerun",
        description="Log nuScenes data into a Rerun recording, viewable with the Rerun Viewer.",
    )
    subparsers = parser.add_subparsers(dest="loader", required=True)

    for loader in LOADERS:
        sub = subparsers.add_parser(loader.name, help=f"Visualize a {loader.name} file")
        sub.add_argument("path", type=str, help="Path to the input file")
        sub.add_argument(
            "--dataroot", default=None,
            help="nuScenes dataset root, if it can't be auto-detected from the file's location",
        )
        sub.add_argument(
            "--entity-path-prefix", default="",
            help="Prefix for all logged entity paths",
        )
        sub.add_argument(
            "--static", action="store_true",
            help="Log data as static (timeless) instead of at the current time",
        )
        output = sub.add_mutually_exclusive_group()
        output.add_argument("--save", default=None, help="Save the recording to this .rrd file instead of spawning a viewer")
        output.add_argument("--connect", default=None, help="Connect to an existing Rerun viewer/server (e.g. rerun+http://127.0.0.1:9876/proxy) instead of spawning one")
        sub.set_defaults(_loader=loader)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    loader = args._loader

    rr.init(f"nuscenes_rerun_{loader.name}", spawn=args.save is None and args.connect is None)
    if args.save:
        rr.save(args.save)
    elif args.connect:
        rr.connect_grpc(args.connect)

    loader.log(Path(args.path), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
