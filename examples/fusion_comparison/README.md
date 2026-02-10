# Fusion Strategy Comparison Experiment

This example demonstrates how to compare different cooperative perception fusion strategies using OpenCOOD. We compare **Late Fusion** (minimal bandwidth) vs **Intermediate Fusion** (higher accuracy).

## Key Concept: Transform-before-send vs Send-before-transform

OpenCOOD supports two spatial alignment strategies for intermediate fusion, controlled by the `proj_first` parameter:

### Transform-before-send (`proj_first: true`)
- Point clouds are projected to ego coordinate frame **before** feature extraction
- Features are already spatially aligned
- Used by: F-Cooper, basic intermediate fusion
- Simpler fusion logic, but assumes synchronous data

### Send-before-transform (`proj_first: false`)
- Features extracted in each CAV's local coordinate frame
- Features warped to ego frame **during** fusion using transformation matrix
- Used by: V2VNet, V2X-ViT, CoAlign
- Enables spatial correction for time-delayed data

See `opencood/data_utils/datasets/intermediate_fusion_dataset.py:36-42` for implementation.

## Configuration Differences

| Parameter | Late Fusion | Intermediate (F-Cooper) |
|-----------|-------------|-------------------------|
| `fusion.core_method` | `LateFusionDataset` | `IntermediateFusionDataset` |
| `fusion.args.proj_first` | N/A | `true` |
| `model.core_method` | `point_pillar` | `point_pillar_fcooper` |
| Bandwidth | ~0.024 Mbps | ~72 Mbps |

## Prerequisites

1. Install OpenCOOD following the main [README](../../README.md)
2. Download OPV2V validation data to `opv2v_data_dumping/validate/`
3. Download pretrained checkpoints from [UCLA Box](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V)

## Running the Experiment

### Option 1: Using the Script
```bash
# Edit checkpoint paths in run_fusion_comparison.sh first
chmod +x examples/fusion_comparison/run_fusion_comparison.sh
./examples/fusion_comparison/run_fusion_comparison.sh
```

### Option 2: Manual Execution
```bash
# Late Fusion
python opencood/tools/inference.py \
    --model_dir /path/to/late_fusion_checkpoint \
    --fusion_method late

# Intermediate Fusion (F-Cooper)
python opencood/tools/inference.py \
    --model_dir /path/to/fcooper_checkpoint \
    --fusion_method intermediate
```

## Expected Results (OPV2V Default Towns)

| Method | AP@0.7 | Bandwidth |
|--------|--------|-----------|
| Late Fusion (PointPillar) | 78.1% | 0.024 Mbps |
| Intermediate (F-Cooper) | 79.0% | 72.08 Mbps |
| Intermediate (F-Cooper, compressed) | 78.8% | 1.12 Mbps |

## Understanding the Results

**Why Intermediate Fusion Performs Better:**
- Shares richer information (feature maps vs detection boxes)
- Can resolve occlusions through feature-level fusion

**Bandwidth-Accuracy Tradeoff:**
- Late fusion: Minimal bandwidth, limited by per-agent detection quality
- Intermediate: Better accuracy but ~3000x bandwidth increase
- Set `compression: 64` in config to reduce bandwidth to ~1.12 Mbps

## Troubleshooting

### 1. CUDA Out of Memory
Reduce `max_voxel_test` in the config (try 40000 instead of 70000).

### 2. FileNotFoundError for data
Verify `validate_dir` path in your checkpoint's `config.yaml`. Path should be absolute or relative to working directory.

### 3. Model architecture mismatch
Ensure checkpoint matches the fusion method:
- Late fusion checkpoint for `--fusion_method late`
- F-Cooper checkpoint for `--fusion_method intermediate`

### 4. spconv version errors
Some checkpoints require specific spconv versions (1.2.1 vs 2.x). Check the benchmark table for version info.

### 5. Batch size assertion error
Inference requires `batch_size=1`. This is handled automatically by the inference script.

## Citation

If using this experiment in your research, please cite:
```bibtex
@inproceedings{xu2022opencood,
  author = {Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, Jiaqi Ma},
  title = {OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication},
  booktitle = {ICRA},
  year = {2022}
}
```
