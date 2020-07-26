# JL-DCF-pytorch

Pytorch implementation for JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection (CVPR2020) [PDF](https://arxiv.org/pdf/2004.08515v1)

# Requirements
* Python 3.6 <br>
* Pytorch 1.5.0 <br>
* Torchvision 0.6.1 <br>
* Cuda 10.0

# Usage
This is the Pytorch implementation of JL-DCF. It has been trained and tested on Windows (Win10 + Cuda 10 + Python 3.6 + Pytorch 1.5),
and it should also work on Linux but we didn't try. 

## To Train 
* Download the pre-trained ImageNet backbone (resnet101 and vgg_conv1, whereas the latter already exists in the folder), and put in it the 'pretrained' folder
* Download the training dataset and modify the 'train_root' and 'train_list' in the `main.py`
* Set 'mode' to 'train'
* Run `main.py`

## To Test 
* Download the testing dataset and have it in the 'dataset/test/' folder 
* Download the already-trained JL-DCF pytorch model and modify the 'model' to its saving path in the `main.py`
* Modify the 'test_folder' in the `main.py` to the testing results saving folder you want
* Modify the 'sal_mode' to select one testing dataset (NJU2K, NLPR, STERE, RGBD135, LFSD or SIP)
* Set 'mode' to 'test'
* Run `main.py`

## Learning curve
The training log is saved in the 'log' folder. If you want to see the learning curve, you can get it by using: ` tensorboard --logdir your-log-path`

# Pre-trained ImageNet model for training
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[vgg_conv1](https://pan.baidu.com/s/1CJyNALzPIAiHrDSMcRO2yA), password: rllb<br>

# Trained model for testing
Baidu Pan: [JL-DCF-pytorch](https://pan.baidu.com/s/1KoxUvnnM5zJoFPEkrv7b1Q), password: jdpb<br>
Google Drive: Coming soon<br>

# JL-DCF-pytorch saliency maps
Baidu Pan: [Saliency maps](https://pan.baidu.com/s/1dhuL2108DxLdAj4J59gAaw), password:  gtkc<br>
Google Drive: Coming soon<br>

# Dataset
Baidu Pan:<br>
[Training dataset (with horizontal flip)](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
[Testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
Google Drive:<br>
Coming soon<br>

# Performance
Below is the performance of JL-DCF-pyotrch (Pytorch implementation). Generally, the performance of Pytorch implementation is comparable to, and even slightly better than the previous Caffe implementation reported in the paper. This is probably due to the differences between deep learning platforms. Also, due to the randomness in the training process, the obtained results will fluctuate slightly.
| Datasets | Metrics | Pytorch |
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
