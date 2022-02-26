## Benchmark: 3D LiDAR  Detection

---
### Results on OPV2V dataset (AP@0.7 for no-compression/ compression)

|                    | Backbone   | Fusion Strategy  | Bandwidth (Megabit), <br/> before/after compression| Default Towns    |Culver City| Download |
|--------------------| --------   | ---------------  | ---------------                | -------------    |-----------| -------- |
| Naive Late         | PointPillar        | Late      |    **0.024**/**0.024** |   0.781/0.781        | 0.668/0.668         |    [url](https://drive.google.com/file/d/1WTKooW6k0exLqoIE5Czqy6ptycYlgKZz/view?usp=sharing)   |
| [Cooper](https://arxiv.org/abs/1905.05265)       | PointPillar        | Early  |   7.68/7.68   | 0.800/x         | 0.696/x       | [url](https://drive.google.com/file/d/1N1p6syxGSKD18ELgtBQoSuUzR8tX1JeE/view?usp=sharing)     | 
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)         | PointPillar        | Intermediate  | 126.8/1.981     | **0.815**       | **0.735**         | [url](https://drive.google.com/file/d/1QBcNQso1zISqf4Fw18FvWLQdDL6Rx-Sr/view?usp=sharing)     | 
| [F-Cooper](https://arxiv.org/abs/1909.06459)         | PointPillar        | Intermediate  |     | 0.790     | 0.728        | [url](https://drive.google.com/file/d/1CjXu9Y2ZTzJA6Oo3hnqFhbTqBVKq3mQb/view?usp=sharing)     | 
| Naive Late         | VoxelNet        | Late  |    | 0.738          | 0.588        | [url]()    |
| Cooper    | VoxelNet        | Early   |     | 0.758        | .677        | [url](https://drive.google.com/file/d/14WD7iLLyyCJJ3lApbYYdr5KOUM1ACnve/view?usp=sharing)     | 
| Attentive Fusion        | VoxelNet        | Intermediate |      | **0.864**        | **0.775**        | [url](https://drive.google.com/file/d/1QoEvuZtXfC5U5-HAbnyeJKAiAN54MidY/view?usp=sharing)      | 
| Naive Late         | SECOND        | Late |     |  0.775         |0.682        | [url](https://drive.google.com/file/d/1VG_FKe1mKagPVGXH7UGHpyaM5q3cxtD8/view?usp=sharing)      |
| Cooper    | SECOND        | Early  |      |  0.813       |  0.738     | [url](https://drive.google.com/file/d/1Z9io1VNcU-urcRW8l0ogWCTVCB53mw4N/view?usp=sharing)     | 
| Attentive         | SECOND        | Intermediate |      |   **0.826**     | **0.760**     | [url](https://drive.google.com/file/d/107005eltMD9bmb1RHz4ZWWZQT1TP6Gp0/view?usp=sharing)      | 
| Naive Late         | PIXOR        | Late |     |    0.578       |  0.360      | [url]()      |
| Cooper    | PIXOR        | Early |       |   0.678      | **0.558**      | [url](https://drive.google.com/file/d/1ZDLjtizZCuV6D92LloEPKRIw-LqxfE1j/view?usp=sharing)     | 
| Attentive         | PIXOR        | Intermediate  |     |  **0.687**      | 0.546       | [url]()      |


**Note**: 
* We suggest using **PointPillar** as the backbone when you are creating your method and try to compare with
our benchmark, as we implement most of the SOTA methods with this backbone only.