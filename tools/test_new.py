import sys
import os
import os.path as osp

current_dir = osp.dirname(osp.abspath(__file__))
root_dir = osp.abspath(osp.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

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

    # 1. Load the updated MMEngine-friendly config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 2. Set up the target execution workspace directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    # 3. Inject the evaluated model checkpoint target file paths
    cfg.load_from = args.checkpoint
    
    # 4. Configure launcher environments
    if args.launcher == 'none':
        cfg.launcher = 'none'
    else:
        cfg.launcher = args.launcher

    # 5. Build the MMEngine Runner natively for evaluation
    runner = Runner.from_cfg(cfg)

    # 6. Kick off testing (this reads test_dataloader & test_evaluator from your updated config)
    runner.test()

if __name__ == '__main__':
    main()