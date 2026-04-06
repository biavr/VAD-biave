from mmengine.registry import Registry, DATA_SAMPLERS, build_from_cfg

# SAMPLER = Registry('sampler')

SAMPLER = Registry('sampler', parent=DATA_SAMPLERS, scope='projects')

def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
