#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import os.path as osp
import types
import argparse
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

# Automatic Path Resolution
current_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.abspath(osp.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Basic MMCV Utils Compatibility Layer for internal metrics tracking
from mmengine.utils.version_utils import digit_version
from mmengine.utils.dl_utils.parrots_wrapper import TORCH_VERSION

try:
    import mmcv.utils
except ImportError:
    import mmcv
    mmcv.utils = types.ModuleType('mmcv.utils')
    sys.modules['mmcv.utils'] = mmcv.utils

mmcv.utils.TORCH_VERSION = TORCH_VERSION
mmcv.utils.digit_version = digit_version

def parse_args():
    parser = argparse.ArgumentParser(description='MMEngine Standard Test Script')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the directory to save evaluation metrics')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--eval', type=str)
    parser.add_argument('--tmpdir', type=str)
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint
    cfg.launcher = args.launcher

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    main()