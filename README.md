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
**09/15/2022**: Powered by OpenCOOD, the paper *Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps* has been accepted by **NeuRIPS 2022**!

**09/06/2022**: Powered by OpenCOOD, our paper *CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers* has been accepted by **CoRL2022**!

**07/06/2022**: Powered by OpenCOOD, our paper *V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer* has been accepted by **ECCV2022**!

**01/31/2022**: Our paper *OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication* has been accpted by ICRA2022!

**09/21/2021**: OPV2V dataset is public available: https://mobility-lab.seas.ucla.edu/opv2v/

## Features
- Provide easy data API for multiple popular multi-agent perception dataset:
  - [x] [OPV2V [ICRA2022]](https://mobility-lab.seas.ucla.edu/opv2v/)
  - [ ] [V2XSet [ECCV2022]]()

- Provide multiple SOTA 3D detection backbone:
    - [X] [PointPillar](https://arxiv.org/abs/1812.05784)
    - [X] [Pixor](https://arxiv.org/abs/1902.06326)
    - [X] [VoxelNet](https://arxiv.org/abs/1711.06396)
    - [X] [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)
- Support  SOTA multi-agent perception models:
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [Cooper [ICDCS]](https://arxiv.org/abs/1905.05265)
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [V2VNet [ECCV2022]](https://arxiv.org/abs/2008.07519)
    - [x] [FPV-RCNN [RAL2022]](https://arxiv.org/pdf/2109.11615.pdf)
    - [ ] [DiscoNet [NeurIPS2022]](https://arxiv.org/abs/2111.00643)
    - [ ] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
- **Provide a convenient log replay toolbox for OPV2V dataset.** More importantly, it allows users to enrich the original dataset by
    attaching new sensors or define additional tasks (e.g. tracking, prediction)
    without changing the events in the initial dataset (e.g. positions and number of all vehicles, traffic speed).

## Data Downloading
All the data can be downloaded from [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu). If you have a good internet, you can directly
download the complete large zip file such as `train.zip`. In case you suffer from downloading large fiels, we also split each data set into small chunks, which can be found 
in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:
```python
cat train.zip.parta* > train.zip
unzip train.zip
```

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
### Results on OPV2V LiDAR-track (AP@0.7 for no-compression/ compression)

|                    | Backbone   | Fusion Strategy  | Bandwidth (Megabit), <br/> before/after compression| Default Towns    |Culver City| Download |
|--------------------| --------   | ---------------  | ---------------                | -------------    |-----------| -------- |
| Naive Late         | PointPillar        | Late      |    **0.024**/**0.024** |   0.781/0.781        | 0.668/0.668         |    [url](https://drive.google.com/file/d/1WTKooW6k0exLqoIE5Czqy6ptycYlgKZz/view?usp=sharing)   |
| [Cooper](https://arxiv.org/abs/1905.05265)       | PointPillar        | Early  |   7.68/7.68   | 0.800/x         | 0.696/x       | [url](https://drive.google.com/file/d/1N1p6syxGSKD18ELgtBQoSuUzR8tX1JeE/view?usp=sharing)     | 
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)         | PointPillar        | Intermediate  | 126.8/1.98   | 0.815/0.810       | 0.735/0.731        | [url](https://drive.google.com/file/d/1u4w13SDzdGq6Irh2PHxT-qIlNXRT3z6Z/view?usp=sharing)     | 
| [F-Cooper](https://arxiv.org/abs/1909.06459)         | PointPillar        | Intermediate  | 72.08/1.12    | 0.790/0.788     | 0.728/0.726        | [url](https://drive.google.com/file/d/1CjXu9Y2ZTzJA6Oo3hnqFhbTqBVKq3mQb/view?usp=sharing)     | 
| [V2VNet](https://arxiv.org/abs/2008.07519)         | PointPillar        | Intermediate  | 72.08/1.12    | **0.822**/0.814     | 0.734/0.729    | [url](https://drive.google.com/file/d/14xl_gNEIHcDw-SvQyO1ioQwyzGym-tKX/view?usp=sharing)     | 
| [FPV-RCNN](https://arxiv.org/abs/2109.11615)         | PV-RCNN        | Intermediate(2 stage)  | 0.24/0.24    | 0.820/**0.820**     | **0.763**/**0.763**    | [url](https://drive.google.com/file/d/1iOVi7holJ-Cu2P3dRv5HmOWlB5lkLukJ/view)     | 
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


### Results of BEV semantic segmentation on OPV2V camera-track (IoU)

|                    | Backbone   | Fusion Strategy  | Vehicles| Road Surface   |Lane| Download |
|--------------------| --------   | ---------------  | ---------------                | -------------    |-----------| -------- |
| No Fusion        | CVT        | No Fusion      |    37.7 |   57.8        | 43.7     |    [None]()   |
| Map Fusion      | CVT        | Late  |   45.1   |  60.0     | 44.1      | [None]()     | 
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)         | CVT        | Intermediate  | 51.9  |60.5       | 46.2        | [None]()     | 
| [F-Cooper](https://arxiv.org/abs/1909.06459)         | CVT        | Intermediate  |52.5    | 60.4    | 46.5       | [None]()     | 
| [V2VNet](https://arxiv.org/abs/2008.07519)         | CVT        | Intermediate  | 53.5     | 60.2     | 47.5   | [None]()     | 
| [DiscoNet](https://arxiv.org/abs/2109.11615)         | CVT       | Intermediate  | 52.9   |  60.7   | 45.8    | [None]()     | 
| [FuseBEVT](https://arxiv.org/pdf/2207.02202.pdf)        | CVT        | Intermediate  | 59.0     | 62.1        | 49.2      | [url]()    |
| [CoBEVT](https://arxiv.org/pdf/2207.02202.pdf)        | SinBEVT        | Intermediate  | **60.4**     | **63.0**          | **53.0**      | [url](https://drive.google.com/drive/folders/1NLzyvMFxuv8Qy52q_OzcNsugTS5JacAT)    |

**Note**: 
To play with OPV2V camera data, please check here: https://github.com/DerrickXuNu/CoBEVT

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

## Relevant Projects
OpenCOOD has supported several projects in cooperative perception field.

**Where2comm: Communication-Efficient Collaborative Perception via Spatial Confidence Maps** <br>
Yue Hu, Shaoheng Fang, Zixing Lei, Yiqi Zhong, Siheng Chen<br>
*Neurips 2022* <br>
[[Paper]](https://arxiv.org/abs/2209.12836) [[Code]](https://github.774.gs/MediaBrain-SJTU/Where2comm)

**CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers** <br>
Runsheng Xu*, Zhengzhong Tu*, Hao Xiang, Wei Shao, Bolei Zhou, Jiaqi Ma <br>
*CoRL2022* <br>
[[Paper]](https://arxiv.org/abs/2207.02202) [[Code]](https://github.com/DerrickXuNu/CoBEVT)

**V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer** <br>
Runsheng Xu*, Hao Xiang*, Zhengzhong Tu*, Xin Xia, Ming-Hsuan Yang, Jiaqi Ma <br>
*ECCV2022* <br>
[[Paper]](https://arxiv.org/abs/2203.10638) [[Code]](https://github.com/DerrickXuNu/v2x-vit) [[Talk]](https://course.zhidx.com/c/MmQ1YWUyMzM1M2I3YzVlZjE1NzM=)

**OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with Vehicle-to-Vehicle Communication** <br>
Runsheng Xu*, Hao Xiang*, Xin Xia, Jinlong Li, Jiaqi Ma <br>
*ICRA2022* <br>
[[Paper]](https://arxiv.org/abs/2109.07644) [[Website]](https://mobility-lab.seas.ucla.edu/opv2v/) [[Code]](https://github.com/DerrickXuNu/OpenCOOD)
