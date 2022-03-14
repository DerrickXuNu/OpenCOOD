# OpenCOOD
[![Documentation Status](https://readthedocs.org/projects/opencood/badge/?version=latest)](https://opencood.readthedocs.io/en/latest/?badge=latest) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

OpenCOOD is an <strong>Open</strong> <strong>COO</strong>perative <strong>D</strong>etection framework for autonomous driving. It is also the official implementation of the <strong> ICRA 2022  </strong>
paper [OPV2V.](https://arxiv.org/abs/2109.07644)

<p align="center">
<img src="images/demo1.gif" width="600" alt="" class="img-responsive">
<img src="images/camera_demo.gif" width="600"  alt="" class="img-responsive">
</p>

## News
**03/10/2022**: Results and pretrained weights for intermeidate Fusion with **compression** are provided.

**02/20/2022**: [F-Cooper](https://arxiv.org/abs/1909.06459) now is supported and the results/traiend model can be found in the benchmark table.

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
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE}]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'constant', 'intensity' or 'z-value'.


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
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.

The evaluation results  will be dumped in the model directory.

## Benchmark and model zoo
### Results on OPV2V dataset (AP@0.7 for no-compression/ compression)

|                    | Backbone   | Fusion Strategy  | Bandwidth (Megabit), <br/> before/after compression| Default Towns    |Culver City| Download |
|--------------------| --------   | ---------------  | ---------------                | -------------    |-----------| -------- |
| Naive Late         | PointPillar        | Late      |    **0.024**/**0.024** |   0.781/0.781        | 0.668/0.668         |    [url](https://drive.google.com/file/d/1WTKooW6k0exLqoIE5Czqy6ptycYlgKZz/view?usp=sharing)   |
| [Cooper](https://arxiv.org/abs/1905.05265)       | PointPillar        | Early  |   7.68/7.68   | 0.800/x         | 0.696/x       | [url](https://drive.google.com/file/d/1N1p6syxGSKD18ELgtBQoSuUzR8tX1JeE/view?usp=sharing)     | 
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)         | PointPillar        | Intermediate  | 126.8/1.98   | **0.815**/**0.810**       | **0.735**/**0.731**        | [url](https://drive.google.com/file/d/1u4w13SDzdGq6Irh2PHxT-qIlNXRT3z6Z/view?usp=sharing)     | 
| [F-Cooper](https://arxiv.org/abs/1909.06459)         | PointPillar        | Intermediate  | 72.08/1.12    | 0.790/0.788     | 0.728/0.726        | [url](https://drive.google.com/file/d/1CjXu9Y2ZTzJA6Oo3hnqFhbTqBVKq3mQb/view?usp=sharing)     | 
| Naive Late         | VoxelNet        | Late  | **0.024**/**0.024**    | 0.738/0.738          | 0.588/0.588        | [url]()    |
| Cooper    | VoxelNet        | Early   |   7.68/7.68  | 0.758/x        | 0.677/x        | [url](https://drive.google.com/file/d/14WD7iLLyyCJJ3lApbYYdr5KOUM1ACnve/view?usp=sharing)     | 
| Attentive Fusion        | VoxelNet        | Intermediate |   576.71/1.12   | **0.864**/**0.852**        | **0.775**/**0.746**       | [url](https://drive.google.com/file/d/16q8CfcB8dS4EVhJMvvEfn0gM2ynxZB3E/view?usp=sharing)      | 
| Naive Late         | SECOND        | Late |  **0.024**/**0.024**    |  0.775/0.775        |0.682/0.682        | [url](https://drive.google.com/file/d/1VG_FKe1mKagPVGXH7UGHpyaM5q3cxtD8/view?usp=sharing)      |
| Cooper    | SECOND        | Early  |   7.68/7.68   |  0.813/x       |  0.738/x     | [url](https://drive.google.com/file/d/1Z9io1VNcU-urcRW8l0ogWCTVCB53mw4N/view?usp=sharing)     | 
| Attentive         | SECOND        | Intermediate |  63.4/0.99     |   **0.826**/**0.783**     | **0.760**/**0.760**    | [url](https://drive.google.com/file/d/1zEB8EyZ0X-WQykHFOM0pVwI8jXunRz1Z/view?usp=sharing)      | 
| Naive Late         | PIXOR        | Late |    **0.024**/**0.024** |    0.578/0.578       |  0.360/0.360      | [url]()      |
| Cooper    | PIXOR        | Early |   7.68/7.68    |   0.678/x      | **0.558**/x      | [url](https://drive.google.com/file/d/1ZDLjtizZCuV6D92LloEPKRIw-LqxfE1j/view?usp=sharing)     | 
| Attentive         | PIXOR        | Intermediate  |   313.75/1.22  |  **0.687**/**0.612**      | 0.546/**0.492**       | [url]()      |


**Note**: 
* We suggest using **PointPillar** as the backbone when you are creating your method and try to compare with
our benchmark, as we implement most of the SOTA methods with this backbone only.
* We assume the transimssion rate is 27Mbp/s. Considering the frequency of LiDAR is 10Hz, the 
bandwidth requirement should be less than **2.7Mbp** to avoid severe delay. 
* A 'x' in the benchmark table represents the bandwidth requirement is too large, which 
can not be considered to employ in practice.
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
- [ ] More SOTA models, such as [DiscoNet](https://arxiv.org/pdf/2111.00643.pdf) and [V2VNet](https://arxiv.org/abs/2008.07519).


## Contributors
OpenCOOD is supported by the [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/). We also appreciate the great work from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), as 
part of our works use their framework.<br>

### Lab Principal Investigator:
- Dr. Jiaqi Ma ([linkedin](https://www.linkedin.com/in/jiaqi-ma-17037838/),
               [UCLA Samueli](https://samueli.ucla.edu/people/jiaqi-ma/))

### Project Lead: <br>
 - Runsheng Xu ([linkedin](https://www.linkedin.com/in/runsheng-xu/), [github](https://github.com/DerrickXuNu))  <br>
 - Hao Xiang ([linkedin](https://www.linkedin.com/in/hao-xiang-42bb5a1b2/), [github](https://github.com/XHwind))
