# Visualization

We provide the script to visualize the VAD prediction to a video [here](../tools/analysis_tools/visualization.py).

## Visualize prediction

```shell
cd /path/to/VAD/
conda activate vad
python tools/analysis_tools/visualization.py --result-path /workspace/logs/outputs_tiny_stage_1_with_occ/results_nusc_15ep.pkl --save-path /workspace/logs/outputs_tiny_stage_1_with_occ
```

The inference results is a prefix_results_nusc.pkl automaticly saved to the work_dir after running evaluation. It's a list of prediction results for each validation sample.
