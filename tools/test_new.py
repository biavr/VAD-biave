import sys
import os
import os.path as osp

current_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.abspath(osp.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import types
import argparse

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

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.registry import DefaultScope

import projects.mmdet3d_plugin

def parse_args():
    parser = argparse.ArgumentParser(description='MMEngine Test Detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--work-dir', help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # Placeholder args to support arguments typed in the command line
    parser.add_argument('--eval', type=str, help='legacy eval option placeholder')
    parser.add_argument('--tmpdir', type=str, help='legacy tmpdir option placeholder')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()

    # Load the updated MMEngine-friendly config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Set up the target execution workspace directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # Inject the evaluated model checkpoint target file paths
    cfg.load_from = args.checkpoint
    
    # Configure launcher environments
    if args.launcher == 'none':
        cfg.launcher = 'none'
    else:
        cfg.launcher = args.launcher

    # Build the MMEngine Runner natively for evaluation
    runner = Runner.from_cfg(cfg)

    # Kick off testing
    runner.test()

if __name__ == '__main__':
    main()