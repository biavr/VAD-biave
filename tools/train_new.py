# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
#  Further modified for mmengine/mmcv2 Runner-based training
# ---------------------------------------------

from __future__ import division

# --- THE INTERNAL VAD COMPATIBILITY PATCH ---
# These shims exist because projects/mmdet3d_plugin/ still imports from the
# legacy mmcv-1.x API (mmcv.runner.BaseModule, force_fp32, ext_loader, etc.)
# which doesn't exist in the real mmcv 2.1.0 / mmdet3d 1.4.0 installed here.
# Keep this block even though main() below no longer calls most of these
# symbols directly -- the plugin modules imported via custom_imports/plugin_dir
# still need mmcv.runner / mmcv.utils / ext_loader to resolve.
import sys
import types
import os
import torch
import mmengine

# 1. Bridge the Utilities
import mmengine.utils
sys.modules['mmcv.utils'] = mmengine.utils
mmengine.utils.TORCH_VERSION = torch.__version__
mmengine.utils.IS_CUDA_AVAILABLE = torch.cuda.is_available()
mmengine.utils.IS_MLU_AVAILABLE = False
mmengine.utils.IS_ROCM_AVAILABLE = False
mmengine.utils.digit_version = mmengine.utils.digit_version

# 2. Bridge the Runner (Dist + Load)
runner_compat = types.ModuleType('mmcv.runner')
import mmengine.dist
import mmengine.hub
runner_compat.get_dist_info = mmengine.dist.get_dist_info
runner_compat.init_dist = mmengine.dist.init_dist
runner_compat.load_url = lambda url, **kwargs: torch.hub.load_state_dict_from_url(url, **kwargs)
sys.modules['mmcv.runner'] = runner_compat

# 3. Bridge MMCV Main
import mmcv
mmcv.Config = mmengine.Config
mmcv.print_log = mmengine.logging.print_log
mmcv.DictAction = mmengine.ConfigDict

# 4. Bridge the Extension Loader
ext_loader = types.ModuleType('ext_loader')
ext_loader.load_ext = lambda name, funcs: torch.ops.mmcv if hasattr(torch.ops, 'mmcv') else None

# Inject into every possible path VAD might look
sys.modules['mmcv.utils.ext_loader'] = ext_loader
sys.modules['mmdet.utils.ext_loader'] = ext_loader
sys.modules['mmdet3d.utils.ext_loader'] = ext_loader

# This handles the relative import 'from ..utils import ext_loader'
# by pretending there is a top-level utils module with ext_loader in it
import mmengine.utils
mmengine.utils.ext_loader = ext_loader

# get_root_logger / collect_env bridges -- kept in case plugin code calls
# mmdet3d.utils.get_root_logger internally; Runner uses MMLogger natively
# so main() itself no longer needs these directly.
import mmengine.logging
from mmengine.utils.dl_utils import collect_env as mmengine_collect_env

def manual_get_root_logger(log_file=None, log_level='INFO', name='mmdet'):
    return mmengine.logging.MMLogger.get_instance(name, log_file=log_file, log_level=log_level)

import mmdet3d.utils
mmdet3d.utils.get_root_logger = manual_get_root_logger
mmdet3d.utils.collect_env = mmengine_collect_env

# --------------------------------------------

import argparse
import warnings
from os import path as osp

from mmengine import Config, DictAction
from mmengine.runner import Runner

import cv2
cv2.setNumThreads(1)

sys.path.append('')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    print("Starting....")
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plugin/xx, registry will be updated
    if cfg.get('plugin', False):
        import importlib
        if cfg.get('plugin_dir', None):
            plugin_dir = cfg.plugin_dir
        else:
            # import dir is the dirpath for the config file
            plugin_dir = osp.dirname(args.config)
        _module_path = plugin_dir.rstrip('/').replace('/', '.')
        importlib.import_module(_module_path)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                 osp.splitext(osp.basename(args.config))[0])

    # resume handling: mmengine Runner takes a bool `resume` + `load_from` path
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.load_from = args.resume_from
        cfg.resume = True

    # gpu-ids / --gpus only make sense for the old non-distributed manual loop;
    # under Runner, device placement for single-process runs is just "cuda:0"
    # relative to whatever CUDA_VISIBLE_DEVICES exposes, and multi-GPU is
    # driven entirely by the launcher (torchrun) + --launcher pytorch.
    if args.gpu_ids is not None or args.gpus is not None:
        warnings.warn(
            '--gpu-ids/--gpus are ignored under Runner-based training. '
            'Select GPUs via CUDA_VISIBLE_DEVICES and set --launcher pytorch '
            'with torchrun for multi-GPU.')

    # launcher: 'none' = single process, 'pytorch' = torchrun-launched DDP
    cfg.launcher = args.launcher

    # auto-scale LR with world size when distributed
    if args.autoscale_lr:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if 'optim_wrapper' in cfg and 'optimizer' in cfg.optim_wrapper:
            base_lr = cfg.optim_wrapper.optimizer['lr']
            cfg.optim_wrapper.optimizer['lr'] = base_lr * world_size / 8
            print(f"Auto-scaled LR: {base_lr} -> "
                  f"{cfg.optim_wrapper.optimizer['lr']} (world_size={world_size})")
        else:
            warnings.warn(
                '--autoscale-lr requested but cfg.optim_wrapper.optimizer '
                'not found; skipping LR scaling.')

    # seeding + determinism, handled natively by Runner via cfg.randomness
    cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)

    # validation toggle
    if args.no_validate:
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None

    print(f"Distributed launcher: {cfg.launcher}")
    print("Building runner...")
    runner = Runner.from_cfg(cfg)

    print("Start training...")
    runner.train()


if __name__ == '__main__':
    main()