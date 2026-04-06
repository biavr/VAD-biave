import sys
import types
import mmengine
import mmengine.dist
import mmengine.hub
import mmengine.logging
import torch

# --- 1. Create a "Frankenstein" mmcv.runner module ---
# In MMCV 1.x, runner handled everything. In 2.x, it's split.
runner_compat = types.ModuleType('mmcv.runner')

# Move Distributed functions from mmengine.dist
runner_compat.get_dist_info = mmengine.dist.get_dist_info
runner_compat.init_dist = mmengine.dist.init_dist

# Move URL functions (and use our manual torch fix)
def manual_load_url(url, model_dir=None, map_location=None, check_hash=False, file_name=None):
    return torch.hub.load_state_dict_from_url(url, model_dir=model_dir, map_location=map_location)

runner_compat.load_url = manual_load_url

# Inject the "Frankenstein" module into the system
sys.modules['mmcv.runner'] = runner_compat

# --- 2. Keep the other bridges ---
sys.modules['mmcv.utils'] = mmengine.utils
import mmcv
mmcv.Config = mmengine.Config
mmcv.print_log = mmengine.logging.print_log
mmcv.DictAction = mmengine.ConfigDict

print("Updated OpenMMLab 2.0 Patch: Distributed Training Bridge Active.")

# --- Add to the "mmcv.utils" section of your fix_imports.py ---

# MLU is for Cambricon chips; set to False since you are on CUDA
mmengine.utils.IS_MLU_AVAILABLE = False 

# While we're at it, VAD often checks for these too:
mmengine.utils.IS_CUDA_AVAILABLE = torch.cuda.is_available()
mmengine.utils.IS_ROCM_AVAILABLE = False # AMD chips

from mmcv.ops import __path__ as mmcv_ops_path

# Create a fake ext_loader module
ext_loader = types.ModuleType('ext_loader')

# Define the load_ext function that VAD expects
def load_ext(name, funcs):
    import importlib
    # In MMCV 2.x, ops are already pre-loaded or accessible via mmcv._ext
    try:
        ext = importlib.import_module('mmcv._ext')
        return ext
    except ImportError:
        # Fallback for newer mmcv versions
        return None

ext_loader.load_ext = load_ext

# Inject it into the system at the expected path
sys.modules['mmcv.utils.ext_loader'] = ext_loader
# Some VAD versions look here too:
sys.modules['mmdet.utils.ext_loader'] = ext_loader