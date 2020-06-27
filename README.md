# JL-DCF-pytorch

Code of JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection(CVPR2020)  [PDF](https://arxiv.org/pdf/2004.08515v1)
# Requirements
* python 3.6 <br>
* pytorch 1.5.0 <br>
* torchvision 0.6.1 <br>
* cuda 10.0
# Usage
This is the pytorch implemention of JL-DCF. We have trained and tested it on windows (win10 + cuda 10 + python 3.6 + pytorch 1.5), it should also work on Linux but we didn't try. 
## Train 
* Downloading the pre-trained backbone(resnet101,vgg_conv1) and put in it the 'pretrained' file folder
* downloading the train set and modify the 'train_root' and 'train_list' in the main.py
* set 'mode' to 'train'
* run main.py
## Test 
* Downloading the testing dataset and put it in the 'dataset/test/' folder 
* Downloading the trained JL-DCF model and modify the 'model' to it saveing path in the main.py
* Modify the 'test_folder' in the main.py to your testing results save floader
* Modify the 'sal_mode' to select testing dataset(NJU2K,NLPR,STERE,RGBD135,LFSD,SIP)
* set 'mode' to 'test'
* run main.py
## learning curve
The training logs saves in the 'log' folder, if you want to see the learning curve, you can get it by using:<br>
` tensorboard --logdir your-log-path`
# Pre-trained model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password:  rllb<br>
# trained model for testing
[JL-DCF-pytorch](https://pan.baidu.com/), password:  b5r6<br>
# pre-computed saliency maps
[pre-computed saliency maps come from author](https://pan.baidu.com/s/1gaIucFyCWlE4f1qhPKzzTw), password:  5hl9<br>
[pre-computed saliency maps come from pytorch implementation of JL-DCF](), password:  XXXX<br>
# Dataset
* [training dataset with Flip horizontally](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
* [testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
# Performance
This is the performance of JL-DCF on pytorch
|      | JNJU2K | NLPR  | STERE | RGBD135 | LFSD  | SIP   |
| ---- | ------ | ----- | ----- | ------- | ----- | ----- |
| S    | 0.910  | 0.929 | 0.899 | 0.930   | 0.863 | 0.887 |
| F    | 0.911  | 0.916 | 0.894 | 0.918   | 0.868 | 0.894 |
| E    | 0.946  | 0.962 | 0.940 | 0.962   | 0.902 | 0.932 |
| M    | 0.043  | 0.024 | 0.048 | 0.022   | 0.079 | 0.050 |

# Citation
Please cite our paper if you find the work useful:<br>
@InProceedings{Fu_2020_CVPR,<br>
author = {Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao},<br>
title = {JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},<br>
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
year = {2020}<br>
}
