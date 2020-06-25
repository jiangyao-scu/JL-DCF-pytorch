# JL-DCF-pytorch

Code of JL-DCF: Joint Learning and Densely-Cooperative Fusion Framework for RGB-D Salient Object Detection(CVPR2020)
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
`<tensorboard --logdir your-log-path>`
# Pre-trained model
[resnet101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)<br>
[JL-DCF-pytorch](https://baidu.com)<br>
# pre-computed saliency maps
[pre-computed saliency maps](https://baidu.com)
# Performance


# Citation
@InProceedings{Wu_2019_CVPR,<br>
author = {Wu, Zhe and Su, Li and Huang, Qingming},<br>
title = {Cascaded Partial Decoder for Fast and Accurate Salient Object Detection},<br>
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},<br>
month = {June},<br>
year = {2019}<br>
}
