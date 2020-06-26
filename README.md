# JL-DCF-pytorch

Code of JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection(CVPR2020)  [PDF](https://arxiv.org/pdf/2004.08515v1)
# Requirements
* python 3.6 <br>
* pytorch 1.5.0 <br>
* torchvision 0.6.1 <br>
* cuda 10.0
# Usage
## Train 
* Downloading the pre-trained backbone and put in it the pretrained file folder
* downloading the train set and modify the 'train_root' and 'train_list' in the main.py
* set 'mode' to 'train'
* run main.py
## Test 
* Downloading the pre-trained JL-DCF model and modify the 'model' in the main.py
* Modify the 'test_folder' in the main.py to your testing results save floader
* Modify the 'sal_mode' to select testing dataset
* set 'mode' to 'test'
* run main.py
## learning curve
The training logs is saveing in the 'log' folder, if you want to see the learning curve, you can get it by using:<br>
` tensorboard --logdir your-log-path`
# Pre-trained model
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), 提取码 rllb<br>
[JL-DCF-pytorch](https://pan.baidu.com/s/1EZQZBKzSbZjNdJluKsRd9g), 提取码  nqrh<br>
# pre-computed saliency maps
[pre-computed saliency maps](https://pan.baidu.com/s/1gaIucFyCWlE4f1qhPKzzTw), 提取码 5hl9<br>
# Dataset
* [training dataset](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), 提取码 i4mi<br>
* [testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), 提取码  1ju8<br>
# Performance

| Dataset |      | Caffe | resnet50 | pre-computed saliency map | resnet101 |
| ------- | ---- | ----- | -------- | ------------------------- | --------- |
| NJU2K   | S    | 0.903 | 0.906    | 0.902                     | 0.906     |
|         | F    | 0.903 | 0.907    | 0.904                     | 0.906     |
|         | E    | 0.944 | 0.946    | 0.944                     | 0.943     |
|         | M    | 0.043 | 0.040    | 0.041                     | 0.041     |
| NLPR    | S    | 0.925 | 0.931    | 0.925                     | 0.928     |
|         | F    | 0.916 | 0.921    | 0.918                     | 0.915     |
|         | E    | 0.962 | 0.967    | 0.963                     | 0.964     |
|         | M    | 0.022 | 0.021    | 0.022                     | 0.023     |
| STERE   | S    | 0.905 | 0.907    | 0.903                     | 0.901     |
|         | F    | 0.901 | 0.905    | 0.904                     | 0.895     |
|         | E    | 0.946 | 0.948    | 0.947                     | 0.943     |
|         | M    | 0.042 | 0.038    | 0.040                     | 0.042     |
| RGBD135 | S    | 0.929 | 0.928    | 0.931                     | 0.937     |
|         | F    | 0.919 | 0.921    | 0.923                     | 0.934     |
|         | E    | 0.968 | 0.967    | 0.968                     | 0.975     |
|         | M    | 0.022 | 0.021    | 0.021                     | 0.020     |
| LFSD    | S    | 0.862 | 0.853    | 0.862                     | 0.862     |
|         | F    | 0.866 | 0.860    | 0.867                     | 0.868     |
|         | E    | 0.901 | 0.894    | 0.902                     | 0.902     |
|         | M    | 0.071 | 0.076    | 0.070                     | 0.072     |
| SIP     | S    | 0.879 | 0.888    | 0.880                     | 0.878     |
|         | F    | 0.885 | 0.892    | 0.889                     | 0.887     |
|         | E    | 0.923 | 0.931    | 0.925                     | 0.926     |
|         | M    | 0.051 | 0.046    | 0.049                     | 0.051     |

# Citation
Please cite our paper if you find the work useful:<br>
@InProceedings{Fu_2020_CVPR,<br>
author = {Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao},<br>
title = {JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},<br>
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
year = {2020}<br>
}
