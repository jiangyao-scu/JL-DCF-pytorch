to 蒋遥
这是JL-DCF的pytorch版本
我已经完成代码部分，经过多次调试，应该是没有网络结构的错误
调试有2个月多，效果尚可
但是离caffe版本效果还差一点点
你的任务傅老师应该交代过了
就是以resnet101作为backbone调试出不低于caffe的效果
代码细节问题随时交流
how to train:python main.py --arch resnet --train_root ./data/RGBDcollection --train_list ./data/RGBDcollection/train.lst
how to test:python main.py --mode=test --model=./results/..... --test_fold=./results/.... --sal_mode=STERE
