## Benchmark: 3D LiDAR  Detection

---
### Results on OPV2V dataset (AP@0.5/0.7)

|                    | Backbone   | Fusion Strategy  | Default Towns    |Culver City| Download |
|--------------------| --------   | ---------------  | -------------    |-----------| -------- |
| Naive Late         | PointPillar        | Late      |   0.858/0.781        | 0.799/0.668         |    [url](https://drive.google.com/file/d/1WTKooW6k0exLqoIE5Czqy6ptycYlgKZz/view?usp=sharing)   |
| [Cooper](https://arxiv.org/abs/1905.05265)       | PointPillar        | Early        | 0.891/0.800         | 0.829/0.696       | [url](https://drive.google.com/file/d/1N1p6syxGSKD18ELgtBQoSuUzR8tX1JeE/view?usp=sharing)     | 
| [Attentive Fusion](https://arxiv.org/abs/2109.07644)         | PointPillar        | Intermediate       | **0.908**/**0.815**       | **0.854**/**0.735**         | [url](https://drive.google.com/file/d/1QBcNQso1zISqf4Fw18FvWLQdDL6Rx-Sr/view?usp=sharing)     | 
| [F-Cooper](https://arxiv.org/abs/1909.06459)         | PointPillar        | Intermediate       | 0.887/0.790     | 0.845/0.728        | [url](https://drive.google.com/file/d/1CjXu9Y2ZTzJA6Oo3hnqFhbTqBVKq3mQb/view?usp=sharing)     | 
| Naive Late         | VoxelNet        | Late      | 0.801/0.738          | 0.722/0.588        | [url]()    |
| Cooper    | VoxelNet        | Early        | 0.852/0.758        | 0.815/0.677        | [url](https://drive.google.com/file/d/14WD7iLLyyCJJ3lApbYYdr5KOUM1ACnve/view?usp=sharing)     | 
| Attentive Fusion        | VoxelNet        | Intermediate       | **0.906**/**0.864**        | **0.854**/**0.775**        | [url](https://drive.google.com/file/d/1QoEvuZtXfC5U5-HAbnyeJKAiAN54MidY/view?usp=sharing)      | 
| Naive Late         | SECOND        | Late      |  0.846/0.775         | 0.808/0.682        | [url](https://drive.google.com/file/d/1VG_FKe1mKagPVGXH7UGHpyaM5q3cxtD8/view?usp=sharing)      |
| Cooper    | SECOND        | Early        |  0.877/0.813       |  0.821/0.738     | [url](https://drive.google.com/file/d/1Z9io1VNcU-urcRW8l0ogWCTVCB53mw4N/view?usp=sharing)     | 
| Attentive         | SECOND        | Intermediate       |   **0.893/0.826**     | **0.875/0.760**     | [url](https://drive.google.com/file/d/107005eltMD9bmb1RHz4ZWWZQT1TP6Gp0/view?usp=sharing)      | 
| Naive Late         | PIXOR        | Late      |    0.769/0.578       |  0.622/0.360      | [url]()      |
| Cooper    | PIXOR        | Early        |   0.810/0.678      | **0.734/0.558**      | [url](https://drive.google.com/file/d/1ZDLjtizZCuV6D92LloEPKRIw-LqxfE1j/view?usp=sharing)     | 
| Attentive         | PIXOR        | Intermediate       |  **0.815/0.687**      | 0.716/0.546       | [url]()      |


**Note**: We suggest using **PointPillar** as the backbone when you are creating your method and try to compare with
our benchmark, as we implement most of the SOTA methods with this backbone only.