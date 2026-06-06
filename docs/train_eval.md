# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

## Train VAD with 8 GPUs 
```shell
cd /path/to/VAD
conda activate vad
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 --master_port=2333 tools/train.py projects/configs/VAD/VAD_tiny_e2e_updated.py --launcher none --deterministic --work-dir /workspace/logs/outputs

CUBLAS_WORKSPACE_CONFIG=:4096:8 
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 --master_port=2333 tools/train.py projects/configs/VAD/VAD_tiny_e2e_updated.py --launcher none --work-dir /workspace/logs/outputs

CUDA_VISIBLE_DEVICES=2 python tools/train.py projects/configs/VAD/VAD_tiny_e2e_updated.py --work-dir /workspace/logs/outputs

2>&1 | tee logs.txt
```

**NOTE**: We release two types of training configs: the end-to-end configs and the two-stage (stage-1: Perception & Prediction; stage-2: Planning) configs. They should produce similar results. The two-stage configs are recommended because you can just train the stage-1 model once and use it as a pre-train model for stage-2.

## Eval VAD with 1 GPU
```shell
cd /path/to/VAD
conda activate vad
CUDA_VISIBLE_DEVICES=2 python tools/test.py projects/configs/VAD/VAD_base.py /path/to/ckpt.pth --launcher none --eval bbox --tmpdir tmp

CUDA_VISIBLE_DEVICES=2 python tools/test.py /workspace/VAD/projects/configs/VAD/VAD_tiny_e2e_updated.py /workspace/logs/outputs_19_05_nuscenes_mini/epoch_15.pth --launcher none --eval bbox --tmpdir tmp

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python tools/test_new.py /workspace/VAD/projects/configs/VAD/VAD_tiny_e2e_updated.py /workspace/logs/outputs_19_05_nuscenes_mini/epoch_15.pth --launcher none --eval bbox --tmpdir tmp

CUDA_VISIBLE_DEVICES=2 python tools/test_new.py /workspace/VAD/projects/configs/VAD/VAD_tiny_e2e_updated.py /workspace/logs/outputs_19_05_nuscenes_mini/epoch_15.pth --launcher none --eval bbox --tmpdir tmp

```

**NOTE**: Using distributed mode (multi GPUs) for evaluation will lead to inaccurate results, so make sure to use non-distributed mode (1 GPU) for evaluation.

## Reproduce results with pre-trained weights
If you want to reproduce results with pre-trained weights, please change the `img_norm_cfg` setting in your config file to following:

 ``` 
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
```

this is the original setting when we train the model, but we have update it in the recent commit according to this [issue](https://github.com/hustvl/VAD/issues/9)'s advice. If you use the new `img_norm_cfg` config, you will get wrong metric results and visualizations.