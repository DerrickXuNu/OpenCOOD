# OpenCOOD

OpenCOOD is an <strong>Open</strong> <strong>COO</strong>perative <strong>D</strong>etection framework for autonomous driving. It is also the official implementation of the <strong> ICRA 2022  </strong>
paper [OPV2V.](https://arxiv.org/abs/2109.07644)

<p align="center">
<img src="images/demo1.gif" width="600" alt="" class="img-responsive">
<img src="images/camera_demo.gif" width="600"  alt="" class="img-responsive">
</p>

## News
**01/31/2022**: Our paper *OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication* has been accpted by ICRA2022!

**09/21/2021**: OPV2V dataset is public available: https://mobility-lab.seas.ucla.edu/opv2v/

## Features
- **Provide easy data API for the Vehicle-to-Vehicle (V2V) multi-modal perception dataset [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/)**

    It currently provides easy API to load LiDAR data from multiple agents simultaneously in a structured format and
convert to PyTorch Tesnor directly for model use. 
- **Provide multiple SOTA 3D detection backbone**
    
    It supports state-of-the-art LiDAR detector including [PointPillar](https://arxiv.org/abs/1812.05784), [Pixor](https://arxiv.org/abs/1902.06326), [VoxelNet](https://arxiv.org/abs/1711.06396), and [SECOND](https://www.mdpi.com/1424-8220/18/10/3337).
- **Support most common fusion strategies**
  
    It includes 3 most common fusion strategies: early fusion, late fusion, and intermediate fusion across different agents.
- **Support several SOTA multi-agent visual fusion model** 

    It supports the most recent multi-agent perception algorithms (currently up to Sep. 2021) including [Attentive Fusion](https://arxiv.org/abs/2109.07644),
    [Cooper (early fusion)](https://arxiv.org/abs/1905.05265), [F-Cooper](https://arxiv.org/abs/1909.06459), etc. We will keep updating
    the newest algorithms.
- **Provide a convenient log replay toolbox for OPV2V dataset (coming soon)**

    It also provides an easy tool to replay the original OPV2V dataset. More importantly, it allows users to enrich the original dataset by
 attaching new sensors or define additional tasks (e.g. tracking, prediction)
    without changing the events in the initial dataset (e.g. positions and number of all vehicles, traffic speed).


## Installation
Please refer to [data introduction](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html)
and [installation](https://opencood.readthedocs.io/en/latest/md_files/installation.html) guide to prepare
data and install OpenCOOD. To see more details of OPV2V data, please check [our website.](https://mobility-lab.seas.ucla.edu/opv2v/)

## Quick Start
### Data sequence visualization
To quickly visualize the LiDAR stream in the OPV2V dataset, first modify the `validate_dir`
in your `opencood/hypes_yaml/visualization.yaml` to the opv2v data path on your local machine, e.g. `opv2v/validate`,
and the run the following commond:
```python
cd ~/OpenCOOD
python opencood/visualization/vis_data_sequence.py 
```
### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```python
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/second_early_fusion.yaml`, meaning you want to train
an early fusion model which utilizes SECOND as the backbone. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `opv2v_data_dumping/test`.

```python
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud. 

The evaluation results  will be dumped in the model directory.

## Benchmark and model zoo
We currently provide 3D LiDAR detection benchmark on OPV2V dataset, please refer to [benchmark](https://opencood.readthedocs.io/en/latest/md_files/lidar_benchmark.html). More
benchmark results of different data modalities/tasks will be revealed soon.

## Tutorials
We have a series of tutorials to help you understand OpenCOOD more. Please check the series of our [tutorials](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html).


## Citation
 If you are using our OpenCOOD framework or OPV2V dataset for your research, please cite the following paper:
 ```bibtex
@inproceedings{xu2022opencood,
  author = {Runsheng Xu, Hao Xiang, Xin Xia, Xu Han, Jinlong Li, Jiaqi Ma},
  title = {OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication},
  booktitle = {2022 IEEE International Conference on Robotics and Automation (ICRA)},
  year = {2022}}
```

## Future Plans
- [ ] Provide camera APIs for OPV2V
- [ ] Provide the log replay toolbox
- [ ] More SOTA models


## Contributors
OpenCOOD is supported by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/). <br>

### Lab Principal Investigator:
- Dr. Jiaqi Ma ([linkedin](https://www.linkedin.com/in/jiaqi-ma-17037838/),
               [UCLA Samueli](https://samueli.ucla.edu/people/jiaqi-ma/))

### Project Lead: <br>
 - Runsheng Xu ([linkedin](https://www.linkedin.com/in/runsheng-xu/), [github](https://github.com/DerrickXuNu))  <br>
 - Hao Xiang ([linkedin](https://www.linkedin.com/in/hao-xiang-42bb5a1b2/), [github](https://github.com/XHwind))
