import os

import cv2
import torch
from torch.utils import data

import numpy as np
import random


random.seed(10)
normalization = True
class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)




    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        de_name = self.sal_list[item % self.sal_num].split()[1]
        gt_name = self.sal_list[item % self.sal_num].split()[2]
        sal_image = load_image(os.path.join(self.sal_root, im_name))

        sal_depth = load_depth(os.path.join(self.sal_root, de_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))

        #sal_image, sal_depth, sal_label = cv_random_flip(sal_image, sal_depth, sal_label)
        # sal_image, sal_depth, sal_label = cv_random_crop(sal_image, sal_depth, sal_label)

        sal_image = sal_image.transpose((2,0,1))
        sal_depth = sal_depth.transpose((2, 0, 1))
        sal_label = sal_label[...,np.newaxis]
        sal_label = sal_label.transpose((2, 0, 1))

        sal_image = torch.Tensor(sal_image)
        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)

        sample = {'sal_image': sal_image, 'sal_depth': sal_depth, 'sal_label': sal_label}
        return sample

    def __len__(self):
        return self.sal_num


class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item].split()[0]))
        depth = load_depth_test(os.path.join(self.data_root, self.image_list[item].split()[1]))
        image = torch.Tensor(image)
        depth = torch.Tensor(depth)
        return {'image': image, 'name': self.image_list[item % self.image_num].split()[0].split('/')[1], 'size': im_size, 'depth': depth}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=True):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader


def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    # in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(im, (320, 320))
    # if normalization == True:
    #     in_ = in_[:, :, ::-1]
    #     in_ = in_ / 255.0
    #     in_ -= np.array((0.485, 0.456, 0.406))
    #     in_ /= np.array((0.229, 0.224, 0.225))
    # in_ -= np.array((104.00699, 116.66877, 122.67892))
    # in_ = in_.transpose((2,0,1))
    # print(in_.shape)
    return in_


def load_depth(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    # in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(im,(320,320))
    # if normalization == True:
    #     in_ = in_[:, :, ::-1]
    #     in_ = in_ / 255.0
    #     in_ -= np.array((0.485, 0.456, 0.406))
    #     in_ /= np.array((0.229, 0.224, 0.225))
    #in_ -= np.array((104.00699, 116.66877, 122.67892))
    # in_ = in_.transpose((2,0,1))
    return in_

def load_depth_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_,(320,320))
    # if normalization == True:
    #     in_ = in_[:, :, ::-1]
    #     in_ = in_ / 255.0
    #     in_ -= np.array((0.485, 0.456, 0.406))
    #     in_ /= np.array((0.229, 0.224, 0.225))
    #in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_


def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (320, 320))
    # if normalization == True:
    #     in_ = in_[:, :, ::-1]
    #     in_ = in_ / 255.0
    #     in_ -= np.array((0.485, 0.456, 0.406))
    #     in_ /= np.array((0.229, 0.224, 0.225))
    #in_ -= np.array((104.00699, 116.66877, 122.67892))
    #in_ = in_[:,:,::-1].copy()
    #in_ -= np.array((0.406, 0.456, 0.485))
    #in_ /= np.array((0.229, 0.224, 0.225))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label,(320,320))
    label = label / 255.0
    # label = label[...,np.newaxis]
    return label
''' 
   if len(label.shape) == 3:  # 有的label是三通道，范围0-255.单通道范围0or1。所以只有三通道需要除255
        label = label[:,:,0]
        label = label / 255.
'''


def cv_random_flip(img, depth, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:, :, ::-1]
        depth = depth[:, :, ::-1]
        label = label[:, :, ::-1]
    return img, depth,


def cv_random_crop(image, depth, label):

    top = random.randint(0, 20)
    left = random.randint(0, 20)

    image = image[top: top + 300, left: left + 300,:]
    depth = depth[top: top + 300, left: left + 300,:]
    label = label[top: top + 300, left: left + 300]
    image = cv2.resize(image,(320,320))
    depth = cv2.resize(depth, (320, 320))
    label = cv2.resize(label, (320, 320))
    return image,depth,label

if __name__ == '__main__' :
    path = r'D:\pytorch\JL-DCF\data\RGBDcollection_crop\LR/'
    # flst = os.listdir(path)
    # for f in flst:
    #
    #     path_lr = os.path.join(path,f)
    #
    #     path_depth = path_lr.replace("ori.jpg","Depth.png").replace("LR","depth")
    #     path_gt = path_lr.replace("ori.jpg", "GT.png").replace("LR","GT")
    #
    #
    #     img = cv2.imread(path_lr)
    #     depth = cv2.imread(path_depth)
    #     gt = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
    #     img = np.array(img, dtype = np.float32)
    #     depth = np.array(depth, dtype=np.float32)
    #     gt = np.array(gt, dtype=np.float32)
    #     size = img.shape[:2]
    #     h = size[0]
    #     w = size[1]
    #     h_new = int(size[0]*0.9)
    #     w_new = int(size[1]*0.9)
    #     top = random.randint(0, h-h_new)
    #     left = random.randint(0, w-w_new)
    #
    #     img = img[top: top + h_new, left: left + w_new, :]
    #     depth = depth[top: top + h_new, left: left + w_new, :]
    #     gt = gt[top: top + h_new, left: left + w_new]
    #
    #     save_lr = path_lr.replace("_ori.jpg", "_crop_ori.jpg")
    #     save_depth = path_lr.replace("_ori.jpg","_crop_Depth.png").replace("LR","depth")
    #     save_gt = path_lr.replace("_ori.jpg", "_crop_GT.png").replace("LR","GT")
    #     print(save_lr)
    #     cv2.imwrite(save_lr, img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
    #     cv2.imwrite(save_depth, depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    #     cv2.imwrite(save_gt, gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
