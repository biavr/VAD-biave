import mmengine
# Path to your training pkl
data = mmengine.load('/workspace/datasets/nuscenes/nuscenes/vad_nuscenes_infos_temporal_train.pkl')
# Check the first sample's annotations
first_info = data['infos'][0]
if 'gt_names' in first_info:
    print("Categories found in pkl:", set(first_info['gt_names']))
else:
    # Some versions store it inside 'annos'
    print("Categories found in pkl:", set(first_info['annos']['gt_names_3d']))