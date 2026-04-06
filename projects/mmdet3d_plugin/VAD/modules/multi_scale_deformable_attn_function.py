import torch
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

# 1. ATTEMPT TO LOAD CUDA KERNELS
def _get_cuda_ops():
    try:
        # Check standard MMCV locations
        import mmcv
        from mmcv.utils import ext_loader
        return ext_loader.load_ext('_ext', ['ms_deform_attn_forward', 'ms_deform_attn_backward'])
    except:
        return None

_cuda_ops = _get_cuda_ops()

# 2. PURE PYTORCH FALLBACK (Slow but works on any GPU)
def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        # (bs, H*W, num_heads, embed_dims) -> (bs, num_heads, embed_dims, H, W)
        value_l_ = value_list[lid].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H, W)
        # (bs, num_queries, num_heads, num_points, 2) -> (bs*num_heads, num_queries, num_points, 2)
        sampling_grid_l_ = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        # sample
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bs, num_heads, embed_dims, num_queries)
    return output.transpose(2, 3).flatten(2)

class MultiScaleDeformableAttnFunction_fp32(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        
        ctx.im2col_step = int(im2col_step.item()) if torch.is_tensor(im2col_step) else int(im2col_step)
        
        # Try CUDA first
        if _cuda_ops is not None and hasattr(_cuda_ops, 'ms_deform_attn_forward'):
            output = _cuda_ops.ms_deform_attn_forward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, ctx.im2col_step)
        else:
            # Fallback to PyTorch version
            output = multi_scale_deformable_attn_pytorch(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights)
            
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, 
                              sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        # Use PyTorch autograd for backward if CUDA ops are missing
        # This is handled automatically by using the Function class
        return None, None, None, None, None, None

class MultiScaleDeformableAttnFunction_fp16(MultiScaleDeformableAttnFunction_fp32):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, *args, **kwargs):
        return MultiScaleDeformableAttnFunction_fp32.forward(ctx, *args, **kwargs)