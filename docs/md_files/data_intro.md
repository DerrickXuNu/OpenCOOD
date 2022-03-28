## Data Preparation

---
All the data can be downloaded from [google drive](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu). If you have a good internet, you can directly
download the complete large zip file such as `train.zip`. In case you suffer from downloading large fiels, we also split each data set into small chunks, which can be found 
in the directory ending with `_chunks`, such as `train_chunks`. After downloading, please run the following command to each set to merge those chunks together:
```python
cat train.zip.parta* > train.zip
unzip train.zip
```
After downloading is finished, please make the file structured as following:

```sh
OpenCOOD # root of your OpenCOOD
├── opv2v_data_dumping # the downloaded opv2v data
│   ├── train
│   ├── validate
│   ├── test
│   ├── test_culvercity
├── opencood # the core codebase

```


## Data Introduction

---

OPV2V data is structured as following:

```sh
opv2v_data_dumping
├── train # data for training
│   ├── 2021_08_22_21_41_24  # scenario folder
│     ├── data_protocol.yaml # the simulation parameters used to collect the data in Carla
│     └──  1732 # The connected automated vehicle's id 
│       └── 00000.pcd - 00700.pcd # the point clouds data from timestamp 0 to 700
│       ├── 00000.yaml - 00700.yaml # corresponding metadata for each timestamp
│       ├── 00000_camera0.png - 00700_camera0.png # frontal camera images
│       ├── 00000_camera1.png - 00700_camera1.png # right rear camera images
│       ├── 00000_camera2.png - 00700_camera2.png # left rear camera images
│       └── 00000_camera3.png - 00700_camera3.png # back camera images
├── validate  
├── test
├── test_culvercity
```

### 1. Data Split
OPV2V dataset can be divided into 4 different folders: `train`, `validation`, `test`, and `test_culvercity`
- `train`: contains all training data
- `validate`: used for validation during training
- `test`: test set for calrla default towns
- `test_culvercity`: test set for the digital town of Culver City, Los Angeles

### 2. Scenario Database
OPV2V has 73 scenarios in total, where each of them contains data stream from different agents across different timestamps.
Each scenario is named by the time it was gathered, e.g., `2021_08_22_21_41_24`.

### 3. Agent Contents
Under each scenario folder,  the data of every intelligent agent~(i.e. connected automated vehicle) appearing in the current scenario is saved in different folders. Each folder is named by the agent's unique id, e.g., 1732.

In each agent folder, data across different timestamps will be saved. Those timestamps are represented by five digits integers
as the prefix of the filenames (e.g., 00700.pcd). There are three types of files inside the agent folders: LiDAR point clouds (`.pcd` files), camera images (`.png` files), and metadata (`.yaml` files).

#### 3.1 Lidar point cloud
The LiDAR data is saved with Open3d package and has a postfix ".pcd" in the name. 

#### 3.2 Camera images
Each CAV is equipped with 4 RGB cameras (check https://mobility-lab.seas.ucla.edu/opv2v/ to see the mounted positions of these cameras) to capture the 360 degree of view of the surrounding scene.`camera0`, `camera1`, `camera2`, and `camera3` represent the front, right rear, left rear, and back cameras respectively.

#### 3.3  Data Annotation
All the metadata is saved in yaml files. It records the following important information at the current timestamp:
- **ego information**:  Current ego pose with and without GPS noise under Carla world coordinates, ego speed in km/h, the LiDAR pose, and future planning trajectories. 
- **calibration**: The intrinsic matrix and extrinsic matrix from each camera to the LiDAR sensor.
- **objects annotation**: The pose and velocity of each surrounding human driving vehicle that has at least one point hit by the agent's LiDAR sensor. See [data annotation section](data_annotation_tutorial.md) for more details. 

### 4. Data Collection Protocol
Besides agent contents, every scenario database also has a yaml file named `data_protocol.yaml`. 
This yaml file records the simulation configuration to collect the current scenario. It is used to log replay
the data and enable users to add/modify sensors for new tasks without changing the original events.

To help users collect customized data in CARLA (e.g., different sensor configurations and modalities) with similar data format and structure, we have released our OPV2V data collection code in the `feature/data_collection` branch of [OpenCDA](https://github.com/ucla-mobility/OpenCDA/tree/feature/data_collection). Users can refer to its [documentation](https://opencda-documentation.readthedocs.io/en/latest/md_files/introduction.html) for detailed instructions.
