# from .mmdet_train import custom_train_detector
# from mmseg.apis import train_segmentor
# from mmdet.apis import train_detector

# def custom_train_model(model,
#                 dataset,
#                 cfg,
#                 distributed=False,
#                 validate=False,
#                 timestamp=None,
#                 eval_model=None,
#                 meta=None):
#     """A function wrapper for launching model training according to cfg.

#     Because we need different eval_hook in runner. Should be deprecated in the
#     future.
#     """
#     if cfg.model.type in ['EncoderDecoder3D']:
#         assert False
#     else:
#         custom_train_detector(
#             model,
#             dataset,
#             cfg,
#             distributed=distributed,
#             validate=validate,
#             timestamp=timestamp,
#             eval_model=eval_model,
#             meta=meta)


# def train_model(model,
#                 dataset,
#                 cfg,
#                 distributed=False,
#                 validate=False,
#                 timestamp=None,
#                 meta=None):
#     """A function wrapper for launching model training according to cfg.

#     Because we need different eval_hook in runner. Should be deprecated in the
#     future.
#     """
#     if cfg.model.type in ['EncoderDecoder3D']:
#         train_segmentor(
#             model,
#             dataset,
#             cfg,
#             distributed=distributed,
#             validate=validate,
#             timestamp=timestamp,
#             meta=meta)
#     else:
#         train_detector(
#             model,
#             dataset,
#             cfg,
#             distributed=distributed,
#             validate=validate,
#             timestamp=timestamp,
#             meta=meta)


# 1. Remove mmseg and mmdet.apis imports
# 2. Use MMEngine's Runner instead
from mmengine.runner import Runner
from .mmdet_train import custom_train_detector

def custom_train_model(model,
                       dataset,
                       cfg,
                       distributed=False,
                       validate=False,
                       timestamp=None,
                       eval_model=None,
                       meta=None):
    """
    In OpenMMLab 2.0, we prioritize the custom_train_detector which 
    has been modernized to use the MMEngine Runner.
    """
    # Simply delegate to your custom detector trainer
    return custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)

def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """
    Standard MMEngine training entry point.
    'train_segmentor' and 'train_detector' are now functionally 
    equivalent to Runner.train().
    """
    # Instead of branching between mmseg and mmdet, we use the unified Runner
    runner = Runner.from_cfg(cfg)
    runner.train()