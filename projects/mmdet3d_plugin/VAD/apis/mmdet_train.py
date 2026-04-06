import random
import warnings
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from mmengine.model import BaseModel
from mmengine.model.wrappers import MMDistributedDataParallel
from mmdet.registry import HOOKS
from mmengine.registry import build_from_cfg, RUNNERS
from mmengine.runner import Runner
from mmengine.dist import get_dist_info
from mmengine.optim import build_optim_wrapper, OptimWrapper
from mmengine.config import Config

from mmdet3d.registry import DATASETS
from mmengine.logging import MMLogger
import time
import os.path as osp
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset

def custom_train_detector(model,
                         dataset,
                         cfg,
                         distributed=False,
                         validate=False,
                         timestamp=None,
                         eval_model=None,
                         meta=None):
    # Prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    # print("CFG.DATA:", cfg.data)
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.train_dataloader.get('sampler', dict(type='DefaultSampler')),
            nonshuffler_sampler=dict(type='DefaultSampler', shuffle=False),
        ) for ds in dataset
    ]

    # Put model on GPUs
    if distributed:
        print("Using Distributed Data Parallel")
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        print("Using Data Parallel")
        if not isinstance(model, BaseModel):
        # Ensure the model inherits from MMEngine's BaseModel 
        # so 'train_step' is naturally available.
            pass
        # Put the model on the GPU
        model = model.cuda()

    # Build Optimizer
    opt_cfg = cfg.get('optimizer', cfg.get('optim_wrapper', {}).get('optimizer', None))
    if opt_cfg is None:
        raise AttributeError("Could not find optimizer config in 'optimizer' or 'optim_wrapper'")
    
    wrapper_cfg = dict(optimizer=opt_cfg)
    optimizer = build_optim_wrapper(model, wrapper_cfg)
    
    # Ensure optimizer is an OptimWrapper
    if not isinstance(optimizer, OptimWrapper):
        optimizer = OptimWrapper(optimizer)

    # 1. Bridge the train_cfg for MMEngine Runner
    if 'train_cfg' not in cfg:
        cfg.train_cfg = dict(
            by_epoch=True, 
            max_epochs=cfg.get('total_epochs', cfg.get('max_epochs', 24)), 
            val_interval=1
        )

    # 2. SANITIZE CONFIG (The Shadow Config Fix)
    # This prevents yapf SyntaxError by removing live objects from the version used for logging
    def sanitize_dict(d):
        new_dict = {}
        for k, v in d.items():
            if isinstance(v, dict):
                new_dict[k] = sanitize_dict(v)
            elif isinstance(v, (str, int, float, bool, list, tuple, type(None))):
                new_dict[k] = v
            else:
                new_dict[k] = f"<{type(v).__name__} object>"
        return new_dict

    # Create a clean config object for the Runner's internal logging
    clean_dict = sanitize_dict(cfg.to_dict())
    shadow_cfg = Config(clean_dict)

    # 3. Build Runner
    # We pass live objects as arguments, but the shadow_cfg for metadata logging
    runner = Runner(
        model=model,
        optim_wrapper=optimizer,
        train_dataloader=data_loaders[0],
        work_dir=cfg.work_dir,
        train_cfg=cfg.train_cfg,
        log_processor=cfg.get('log_processor', dict(window_size=50)),
        cfg=shadow_cfg, 
    )

    # runner.timestamp = timestamp

    # Note: Modern MMEngine Runner handles hooks automatically via config.
    # If your config has 'custom_hooks', 'checkpoint_config', etc., 
    # the Runner will pick them up from shadow_cfg.

    # 4. Start Training
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    # In MMEngine, we use .train() instead of .run()
    runner.train()