# Installation

* [__System/Hardware Requirements__](#requirements)
* [__Local Installation__](#local-installation)
    * [__1. CARLA installation__](#1-carla-installation0911-required)
	    * [1.1 Package installation](#11-package-installation)  
	    * [1.2 Build from source](#12-build-from-source)  

    * [__2. Install OpenCDA__](#opencda-installation)
    * [__3. Install Pytorch and Yolov5 (Optional)__](#3-install-pytorch-and-yolov5optional)
    * [__4. Install Sumo (Optional)__](#4-install-sumooptional)




---
## System/Hardware Requirements
To get started, the following requirements should be fulfilled.
* __System requirements.__ OpenCOOD is tested under Ubuntu 18.04
* __Adequate GPU.__ A minimum of 6GB gpu is recommended.
* __Disk Space.__ Estimate 100GB of space is recommended for data downoading.
* __Python__ Python3.7 is required.


---
## Installation
### 1. Dependency Installation
First, download OpenCOOD github to your local folder if you haven't done it yet.
```sh
git clonehttps://github.com/DerrickXuNu/OpenCOOD.git
cd OenCOOD
```
Next we create a conda environment and install the requirements.

```sh
conda env create -f environment.yml
conda activate opencood
python setup.py opencood
```

If conda install failed,  install through pip
```sh
pip install -r requirement.txt
```

### 2. Pytorch Installation (>=1.8)
Go to https://pytorch.org/ to install pytorch cuda version.

### 3. Spconv (1.2.1 requred)
OpenCOOD currently uses the old spconv version to generate voxel features. We will 
upgrade to spconv 2.0 in the short future. To install spconv 1.2.1, please follow the guide in https://github.com/traveller59/spconv/tree/v1.2.1.

#### Tips for installing spconv 1.2.1:
1. make sure your cmake version >= 3.13.2
2. CUDNN and CUDA runtime library (use `nvcc --version` to check) needs to be installed on your machine.


