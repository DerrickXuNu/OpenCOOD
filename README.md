# OpenCOOD

OpenCOOD is an <strong>Open</strong> <strong>COO</strong>perative <strong>D</strong>etection framework in autonomous driving. It is also the official implementation of the <strong> ICRA 2022  </strong>
paper [OPV2V.](https://arxiv.org/abs/2109.07644)

<p align="center">
<img src="images/demo1.gif" width="600" alt="" class="img-responsive">
<img src="images/camera_demo.gif" width="600"  alt="" class="img-responsive">
</p>

## Features
- **Provide easy data API for the Vehicle-to-Vehicle (V2V) multi-modal [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/) dataset**

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
- **Provide a convenient log replay toolbox for OPV2V dataset**

    It also provides an easy tool to replay the origin data collected OPV2V dataset. More importantly, it allows users attaching new sensors and tasks (e.g. tracking, prediction)
    without changing the origin data distribution.


