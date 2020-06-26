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

        sal_image, sal_depth, sal_label = cv_random_crop(sal_image, sal_depth, sal_label)
        sal_image = sal_image.transpose((2,0,1))
        sal_depth = sal_depth.transpose((2, 0, 1))
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
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_, (320, 320))
    if normalization == True:
        in_ = in_[:, :, ::-1]
        in_ = in_ / 255.0
        in_ -= np.array((0.485, 0.456, 0.406))
        in_ /= np.array((0.229, 0.224, 0.225))
    return in_


def load_depth(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_,(320,320))
    if normalization == True:
        in_ = in_[:, :, ::-1]
        in_ = in_ / 255.0
        in_ -= np.array((0.485, 0.456, 0.406))
        in_ /= np.array((0.229, 0.224, 0.225))
    return in_

def load_depth_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ = cv2.resize(in_,(320,320))
    if normalization == True:
        in_ = in_[:, :, ::-1]
        in_ = in_ / 255.0
        in_ -= np.array((0.485, 0.456, 0.406))
        in_ /= np.array((0.229, 0.224, 0.225))
    in_ = in_.transpose((2,0,1))
    return in_


def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ = cv2.resize(in_, (320, 320))
    if normalization == True:
        in_ = in_[:, :, ::-1]
        in_ = in_ / 255.0
        in_ -= np.array((0.485, 0.456, 0.406))
        in_ /= np.array((0.229, 0.224, 0.225))
    in_ = in_.transpose((2, 0, 1))
    return in_, im_size


def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    label = np.array(im, dtype=np.float32)
    label = cv2.resize(label,(320,320))
    label = label / 255.0
    label = label[...,np.newaxis]
    return label



def cv_random_crop(image, depth, label):
    top = random.randint(0, 20)
    left = random.randint(0, 20)

    image = image[top: top + 300, left: left + 300,:]
    depth = depth[top: top + 300, left: left + 300,:]
    label = label[top: top + 300, left: left + 300,:]
    image = cv2.resize(image,(320,320))
    depth = cv2.resize(depth, (320, 320))
    label = cv2.resize(label, (320, 320))
    label = label[..., np.newaxis]
    return image,depth,label
