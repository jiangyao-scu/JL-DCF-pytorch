# JL-DCF-pytorch

Code of JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection(CVPR2020)  [PDF](https://arxiv.org/pdf/2004.08515v1)
# Requirements
* python 3.6 <br>
* pytorch 1.5.0 <br>
* torchvision 0.6.1 <br>
* cuda 10.0
# Usage
## Train 
* Downloading the pre-trained backbone and put in it the 'pretrained' file folder
* downloading the train set and modify the 'train_root' and 'train_list' in the main.py
* set 'mode' to 'train'
* run main.py
## Test 
* Downloading the pre-trained JL-DCF model and modify the 'model' in the main.py
* Modify the 'test_folder' in the main.py to your testing results save floader
* Modify the 'sal_mode' to select testing dataset(NJU2K,NLPR,STERE,RGBD135,LFSD,SIP)
* set 'mode' to 'test'
* run main.py
## learning curve
The training logs saves in the 'log' folder, if you want to see the learning curve, you can get it by using:<br>
` tensorboard --logdir your-log-path`
# Pre-trained model
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password:  rllb<br>
[JL-DCF-pytorch](https://pan.baidu.com/s/1Vu8uUpo3pVd-sXniVlXEAA), password:  b5r6<br>
# pre-computed saliency maps
[pre-computed saliency maps](https://pan.baidu.com/s/1gaIucFyCWlE4f1qhPKzzTw), password:  5hl9<br>
# Dataset
* [training dataset](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
* [testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
# Performance

| dataset |      | Caffe | pre-computed saliency map | resnet101 |
| ------- | ---- | ----- | ------------------------- | --------- |
| NJU2K   | S    | 0.903 |        0.902              | 0.918     |
|         | F    | 0.903 |        0.904              | 0.918     |
|         | E    | 0.944 |        0.944              | 0.952     |
|         | M    | 0.043 |        0.041              | 0.037     |
| NLPR    | S    | 0.925 |        0.925              | 0.934     |
|         | F    | 0.916 |        0.918              | 0.923     |
|         | E    | 0.962 |        0.963              | 0.967     |
|         | M    | 0.022 |        0.022              | 0.022     |
| STERE   | S    | 0.905 |        0.903              | 0.907     |
|         | F    | 0.901 |        0.904              | 0.901     |
|         | E    | 0.946 |        0.947              | 0.944     |
|         | M    | 0.042 |        0.040              | 0.043     |
| RGBD135 | S    | 0.929 |        0.931              | 0.093     |
|         | F    | 0.919 |        0.923              | 0.919     |
|         | E    | 0.968 |        0.968              | 0.967     |
|         | M    | 0.022 |        0.021              | 0.022     |
| LFSD    | S    | 0.862 |        0.862              | 0.870     |
|         | F    | 0.866 |        0.867              | 0.868     |
|         | E    | 0.901 |        0.902              | 0.902     |
|         | M    | 0.071 |        0.070              | 0.072     |
| SIP     | S    | 0.879 |        0.880              | 0.884     |
|         | F    | 0.885 |        0.889              | 0.892     |
|         | E    | 0.923 |        0.925              | 0.929     |
|         | M    | 0.051 |        0.049              | 0.049     |

# Citation
Please cite our paper if you find the work useful:<br>
@InProceedings{Fu_2020_CVPR,<br>
author = {Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao},<br>
title = {JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},<br>
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
year = {2020}<br>
}
