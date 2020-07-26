# JL-DCF-pytorch

Code of JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection(CVPR2020)  [PDF](https://arxiv.org/pdf/2004.08515v1)

# Requirements
* python 3.6 <br>
* pytorch 1.5.0 <br>
* torchvision 0.6.1 <br>
* cuda 10.0

# Usage
This is the pytorch implementation of JL-DCF. We have trained and tested it on windows (win10 + cuda 10 + python 3.6 
+ pytorch 1.5), it should also work on Linux but we didn't try. 

## Train 
* Downloading the pre-trained backbone(resnet101,vgg_conv1) and put in it the 'pretrained' file folder
* downloading the train set and modify the 'train_root' and 'train_list' in the `main.py`
* set 'mode' to 'train'
* run `main.py`

## Test 
* Downloading the testing dataset and put it in the 'dataset/test/' folder 
* Downloading the trained JL-DCF model and modify the 'model' to it saveing path in the `main.py`
* Modify the 'test_folder' in the main.py to your testing results save folder
* Modify the 'sal_mode' to select testing dataset(NJU2K, NLPR, STERE, RGBD135, LFSD, and SIP)
* set 'mode' to 'test'
* run `main.py`

## Learning curve
The training logs saves in the 'log' folder. If you want to see the learning curve, you can get it by using: ` tensorboard --logdir your-log-path`

# Pre-trained model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password: rllb<br>

# Trained model for testing
[JL-DCF-pytorch](https://pan.baidu.com/s/1KoxUvnnM5zJoFPEkrv7b1Q), password: jdpb<br>

# Pre-computed saliency maps
[pre-computed saliency maps come from author](https://pan.baidu.com/s/1gaIucFyCWlE4f1qhPKzzTw), password:  5hl9<br>
[pre-computed saliency maps come from pytorch implementation of JL-DCF](https://pan.baidu.com/s/1FmubauYT2N6BH2NWGxzpTQ), password:  b71v<br>

# Dataset
* [training dataset with flip horizontally](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
* [testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>

# Performance
This is the performance of JL-DCF(pyotrch implementation).
| datasets | metrics | pytorch |
| -------- | ------- | ------- |
| NJU2K    | S       | 0.917   |
|          | F       | 0.919   |
|          | E       | 0.950   |
|          | M       | 0.037   |
| NLPR     | S       | 0.931   |
|          | F       | 0.920   |
|          | E       | 0.964   |
|          | M       | 0.022   |
| STERE    | S       | 0.906   |
|          | F       | 0.903   |
|          | E       | 0.946   |
|          | M       | 0.040   |
| RGBD135  | S       | 0.934   |
|          | F       | 0.928   |
|          | E       | 0.967   |
|          | M       | 0.020   |
| LFSD     | S       | 0.862   |
|          | F       | 0.861   |
|          | E       | 0.894   |
|          | M       | 0.074   |
| SIP      | S       | 0.879   |
|          | F       | 0.889   |
|          | E       | 0.925   |
|          | M       | 0.050   |

# Citation
Please cite our paper if you find the work useful:<br>

        @InProceedings{Fu_2020_CVPR,
        author = {Keren Fu, Deng-Ping Fan, Ge-Peng Ji, Qijun Zhao},
        title = {JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},
        booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2020}
        }
