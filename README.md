# JL-DCF-pytorch

Pytorch implementation for JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection (CVPR2020) [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf)][[中文版](http://dpfan.net/wp-content/uploads/cvpr2020JLDCF_CameraReady_Chinese_final.pdf)]

# Requirements
* Python 3.6 <br>
* Pytorch 1.5.0 <br>
* Torchvision 0.6.1 <br>
* Cuda 10.0

# Usage
This is the Pytorch implementation of JL-DCF. It has been trained and tested on Windows (Win10 + Cuda 10 + Python 3.6 + Pytorch 1.5),
and it should also work on Linux but we didn't try. 

## To Train 
* Download the pre-trained ImageNet backbone (resnet101 and vgg_conv1, whereas the latter already exists in the folder), and put it in the 'pretrained' folder
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
Google Drive: https://drive.google.com/open?id=12u37yz-031unDPJoKaZ0goK8BtPP-6Cj<br>

# JL-DCF-pytorch saliency maps
Baidu Pan: [Saliency maps](https://pan.baidu.com/s/1IzAjbbhoAdhsg-2B_gSwqw), password: 4nqr<br>
Google Drive: https://drive.google.com/open?id=1mHMN36aI5zNt50DQBivSDyYvCQ9eeGhP<br>

# Dataset
Baidu Pan:<br>
[Training dataset (with horizontal flip)](https://pan.baidu.com/s/1vrVcRFTMRO5v-A6Q2Y3-Nw), password:  i4mi<br>
[Testing datadet](https://pan.baidu.com/s/13P-f3WbA76NVtRePcFbVFw), password:   1ju8<br>
Google Drive:<br>
[Training dataset (with horizontal flip)](https://drive.google.com/open?id=12ais7wZhTjaFO4BHJyYyNuzzM312EWCT)<br>
[Testing datadet](https://drive.google.com/open?id=18ALe_HBuNjVTB_US808d8ZKfpd_mwLy5)<br>

# Performance
Below is the performance of JL-DCF-pyotrch (Pytorch implementation). Generally, the performance of Pytorch implementation is comparable to, and even slightly better than the previous [Caffe implementation](https://github.com/kerenfu/JLDCF/) reported in the paper. This is probably due to the differences between deep learning platforms. Also, due to the randomness in the training process, the obtained results will fluctuate slightly.

| Datasets | Metrics | Pytorch |
| -------- | ------- | ------- |
| NJU2K    |S-measure| 0.917   |
|          | maxF    | 0.919   |
|          | maxE    | 0.950   |
|          | MAE     | 0.037   |
| NLPR     |S-measure| 0.931   |
|          | maxF    | 0.920   |
|          | maxE    | 0.964   |
|          | MAE     | 0.022   |
| STERE    |S-measure| 0.906   |
|          | maxF    | 0.903   |
|          | maxE    | 0.946   |
|          | MAE     | 0.040   |
| RGBD135  |S-measure| 0.934   |
|          | maxF    | 0.928   |
|          | maxE    | 0.967   |
|          | MAE     | 0.020   |
| LFSD     |S-measure| 0.862   |
|          | maxF    | 0.861   |
|          | maxE    | 0.894   |
|          | MAE     | 0.074   |
| SIP      |S-measure| 0.879   |
|          | maxF    | 0.889   |
|          | maxE    | 0.925   |
|          | MAE     | 0.050   |  

# Citation
Please cite our paper if you find the work useful: 

	@inproceedings{Fu2020JLDCF,
  	title={JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection},
  	author={Fu, Keren and Fan, Deng-Ping and Ji, Ge-Peng and Zhao, Qijun},
  	booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  	pages={3052--3062},
  	year={2020}
	}
        
	@article{Fu2021siamese,
  	title={Siamese Network for RGB-D Salient Object Detection and Beyond},
  	author={Fu, Keren and Fan, Deng-Ping and Ji, Ge-Peng and Zhao, Qijun and Shen, Jianbing},
  	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  	year={2021}
	}
# Benchmark RGB-D SOD
The complete RGB-D SOD benchmark can be found in this page  
http://dpfan.net/d3netbenchmark/
