import argparse
import os
from dataset.dataset import get_loader
from solver import Solver
import time


def get_test_info(config):
    if config.sal_mode == 'NJU2K':
        image_root = 'D:/work/python/NJU2K_test/'
        image_source = 'D:/work/python/NJU2K_test/test.lst'
    elif config.sal_mode == 'STERE':
        image_root = 'D:/work/python/STERE/'
        image_source = 'D:/work/python/STERE/test.lst'
    elif config.sal_mode == 'RGBD135':
        image_root = 'D:/work/python/RGBD135/'
        image_source = 'D:/work/python/RGBD135/test.lst'
    elif config.sal_mode == 'LFSD':
        image_root = 'D:/work/python/LFSD/'
        image_source = 'D:/work/python/LFSD/test.lst'
    elif config.sal_mode == 'NLPR':
        image_root = 'D:/work/python/NLPR/'
        image_source = 'D:/work/python/NLPR/test.lst'
    elif config.sal_mode == 'SIP':
        image_root = 'D:/work/python/SIP/'
        image_source = 'D:/work/python/SIP/test.lst'


    config.test_root = image_root
    config.test_list = image_source

def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        """
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.mkdir("%s/run-%d" % (config.save_folder, run))
        os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        """
        if not os.path.exists("%s/demo-%s" % (config.save_folder, time.strftime("%d"))):
            os.mkdir("%s/demo-%s" % (config.save_folder, time.strftime("%d")))
        config.save_folder = "%s/demo-%s" % (config.save_folder, time.strftime("%d"))
        train = Solver(train_loader, None, config)
        train.train()
    elif config.mode == 'test':
        # config.test_root, config.test_list =
        get_test_info(config)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")
def run(train,epoch):
    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    # resnet_path = './JL-DCF/weight/demo-28/epoch_60.pth'
    resnet_path = './dataset/pretrained/resnet101-5d3b4d8f.pth'
    # resnet_path = './dataset/pretrained/resnet101.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-9)  # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str,default='')  # ./JL-DCF/weight/demo-28/epoch_60.pth ./JL-DCF/demo-15/epoch_18.pth ./JL-DCF/weight/demo-18-1/epoch_10.pth
    parser.add_argument('--save_folder', type=str, default='JL-DCF/weight/')
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='D:/work/python/RGBDcollection')
    parser.add_argument('--train_list', type=str, default='D:/work/python/RGBDcollection/train1.lst')

    # Testing settings
    model = 'JL-DCF/weight/demo-24/epoch_'+str(epoch)+'.pth'
    test_flod = 'test/demo24new1/'+str(epoch)+'/STERE/'
    parser.add_argument('--model', type=str, default=model)  # Snapshot
    parser.add_argument('--test_fold', type=str, default=test_flod)  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='STERE')  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default=train, choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    # test_root, test_list =
    get_test_info(config)
    # config.test_root = test_root
    # config.test_list = test_list

    main(config)

if __name__ == '__main__':
    # a=[6,8,9,13]
    # for i in range(2,6):
    #     run('test',i)
    # run('test',3)
    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    # resnet_path = './JL-DCF/weight/demo-28/epoch_60.pth'
    resnet_path = './dataset/pretrained/resnet101-5d3b4d8f.pth'
    # resnet_path = './dataset/pretrained/resnet101.pth'

    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-9) # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005) # Weight decay
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet') # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=1) # only support 1 now
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='') # ./JL-DCF/weight/demo-28/epoch_60.pth ./JL-DCF/demo-15/epoch_18.pth ./JL-DCF/weight/demo-18-1/epoch_10.pth
    parser.add_argument('--save_folder', type=str, default='JL-DCF/weight/')
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=50)

    # Train data
    parser.add_argument('--train_root', type=str, default='D:/work/python/RGBDcollection')
    parser.add_argument('--train_list', type=str, default='D:/work/python/RGBDcollection/train1.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default='JL-DCF/weight/demo-22/epoch_40.pth') # Snapshot
    parser.add_argument('--test_fold', type=str, default='test/demo22/40/LFSD/') # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='LFSD') # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    # test_root, test_list =
    get_test_info(config)
    # config.test_root = test_root
    # config.test_list = test_list

    main(config)
