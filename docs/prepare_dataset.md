

## NuScenes
Download nuScenes V1.0 full dataset data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data as follows.


**Download CAN bus expansion**
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*

Directly download the [train](https://drive.google.com/file/d/1OVd6Rw2wYjT_ylihCixzF6_olrAQsctx/view?usp=sharing) file and [val](https://drive.google.com/file/d/16DZeA-iepMCaeyi57XSXL3vYyhrOQI9S/view?usp=sharing) file from google drive, or generate by yourself:
```
CUDA_VISIBLE_DEVICES=0 python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path /workspace/datasets/nuscenes/nuscenes --out-dir /workspace/datasets/nuscenes/nuscenes --extra-tag vad_nuscenes --version v1.0-mini --canbus /workspace/datasets/nuscenes/nuscenes

CUDA_VISIBLE_DEVICES=1 python tools/data_converter/vad_nuscenes_converter.py nuscenes --root-path /workspace/datasets/nuscenes/v1.0-trainval --out-dir /workspace/datasets/nuscenes/v1.0-trainval --extra-tag vad_nuscenes --version v1.0 --canbus /workspace/datasets/nuscenes
```

Using the above code will generate `vad_nuscenes_infos_temporal_{train,val}.pkl`.

**Folder structure**
```
VAD
├── projects/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-19c8e357.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── vad_nuscenes_infos_temporal_train.pkl
|   |   ├── vad_nuscenes_infos_temporal_val.pkl
```
