import argparse
import os
from dataset import get_loader
from solver import Solver
import time


def get_test_info(config):
    if config.sal_mode == 'NJU2K':
        image_root = 'dataset/test/NJU2K_test/'
        image_source = 'dataset/test/NJU2K_test/test.lst'
    elif config.sal_mode == 'STERE':
        image_root = 'dataset/test/STERE/'
        image_source = 'dataset/test/STERE/test.lst'
    elif config.sal_mode == 'RGBD135':
        image_root = 'dataset/test/RGBD135/'
        image_source = 'dataset/test/RGBD135/test.lst'
    elif config.sal_mode == 'LFSD':
        image_root = 'dataset/test/LFSD/'
        image_source = 'dataset/test/LFSD/test.lst'
    elif config.sal_mode == 'NLPR':
        image_root = 'dataset/test/NLPR/'
        image_source = 'dataset/test/NLPR/test.lst'
    elif config.sal_mode == 'SIP':
        image_root = 'dataset/test/SIP/'
        image_source = 'dataset/test/SIP/test.lst'
    elif config.sal_mode == 'ReDWeb-S':
        image_root = 'dataset/test/ReDWeb-S/'
        image_source = 'dataset/test/ReDWeb-S/test.lst'
    else:
        raise Exception('Invalid config.sal_mode')

    config.test_root = image_root
    config.test_list = image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)

        if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
            os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        get_test_info(config)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_folder): os.makedirs(config.test_folder)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':
    resnet101_path = 'pretrained/resnet101-5d3b4d8f.pth'
    resnet50_path = 'pretrained/resnet50-19c8e357.pth'
    vgg16_path = 'pretrained/vgg16-397923af.pth'
    densenet161_path = 'pretrained/densenet161-8d451a50.pth'
    pretrained_path = {'resnet101': resnet101_path, 'resnet50': resnet50_path, 'vgg16': vgg16_path,
                       'densenet161': densenet161_path}

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.00005)  # Learning rate resnet:5e-5
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--device_id', type=str, default='cuda:0')

    # Training settings
    parser.add_argument('--arch', type=str, default='vgg'
                        , choices=['resnet', 'vgg','densenet'])  # resnet, vgg or densenet
    parser.add_argument('--pretrained_model', type=str, default=pretrained_path)  # pretrained backbone model
    parser.add_argument('--epoch', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')  # pretrained JL-DCF model
    parser.add_argument('--save_folder', type=str, default='checkpoints/')
    parser.add_argument('--epoch_save', type=int, default=5)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)
    parser.add_argument('--network', type=str, default='vgg16'
                        , choices=['resnet50', 'resnet101', 'vgg16', 'densenet161'])  # Network Architecture

    # Train data
    parser.add_argument('--train_root', type=str, default='/dataset/RGBDcollection')
    parser.add_argument('--train_list', type=str, default='/dataset/RGBDcollection/train.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default='checkpoints/vgg16.pth')  # Snapshot
    parser.add_argument('--test_folder', type=str, default='test/vgg16/LFSD/')  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='LFSD',
                        choices=['NJU2K', 'NLPR', 'STERE', 'RGBD135', 'LFSD', 'SIP', 'ReDWeb-S'])  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    get_test_info(config)

    main(config)
