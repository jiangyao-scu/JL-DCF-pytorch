import torch
import random
# from collections import OrderedDict
from torch.nn import utils, functional as F
# from torch.optim import Adam
# from torch.autograd import Variable
from torch.backends import cudnn
from networks.JL_DCF import build_model#, weights_init
# import scipy.misc as sm
import numpy as np
import os
# import torchvision.utils as vutils
import cv2
# import math
import time
# from networks.quickdraw import showt,shown
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('run'+time.strftime("%d-%m"))


np.set_printoptions(threshold=np.inf)
size = (320, 320)
size_coarse = (20, 20)


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [100]
        self.build_model()
        self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model))
         #为SCRN载入参数修改
        #self.net.base.load_state_dict(torch.load(r'D:\pytorch\JL-DCF\dataset\pretrained\resnet50_GCPR.pt'),strict=False)



        # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()
        # use_global_stats = True
        # self.net.apply(weights_init)
        if self.config.load == '':
            self.net.JLModule.load_pretrained_model(self.config.pretrained_model)  # load pretrained backbone
        else:
            # pretrained_dict = torch.load(self.config.load)
            # model_dict = self.net.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # model_dict.update(pretrained_dict)
            # self.net.load_state_dict(model_dict)
            self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model


        self.lr = self.config.lr
        self.wd = self.config.wd

        #保持caffe各层相同学习率，weight_decay
        """
        params = []
        resnet_params = dict(self.net.base.resnet.named_parameters())
        cp_params = dict(self.net.base.CP.named_parameters())
        vgg_params = dict(self.net.base.vgg_conv1.named_parameters())
        fusion_params = dict(self.net.fusion.named_parameters())
        score_params = dict(self.net.score.named_parameters())
        for k,v in resnet_params.items():
            params += [{'params':[v],'lr':self.lr}]

        def get_para(dict,params=params):
            for k, v in dict.items():
                if 'bias' in k:
                    params += [{'params': [v], 'lr': 2 * self.lr}]
                else:
                    params += [{'params': [v], 'lr': self.lr, 'weight_decay': self.wd}]
        get_para(cp_params)
        get_para(vgg_params)
        get_para(fusion_params)
        get_para(score_params)
        """
        #self.optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad,self.net.parameters()), lr=self.lr, momentum=0.9)
        #self.optimizer = Adam(filter(lambda p:p.requires_grad,self.net.parameters()), lr=0.0001, weight_decay=self.wd, betas=(0.99, 0.999) )
        self.optimizer = torch.optim.Adadelta(filter(lambda p:p.requires_grad,self.net.parameters()),lr=0.01, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']),\
                                            data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device('cuda:0')
                    # cudnn.benchmark = True
                    images = images.to(device)
                    depth = depth.to(device)

                input = torch.cat((images, depth), dim=0)
                preds,pred_coarse = self.net(input)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min())/(pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_fold, name[:-4] + '_GT.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        file = open('JL-DCF/result19.txt', 'a')
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        # setup_seed(128)
        aveGrad = 0
        self.optimizer.zero_grad()
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            # self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                # print(i)
                sal_image, sal_depth, sal_label = data_batch['sal_image'], data_batch['sal_depth'], data_batch['sal_label']
                # print(sal_image.size())
                # print(sal_depth.size())
                # print(sal_label.size())
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device('cuda:0')
                    # cudnn.benchmark = True
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)
                    #setup_seed(10)

                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                sal_label_coarse = torch.cat((sal_label_coarse,sal_label_coarse), dim=0)
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final, sal_coarse = self.net(sal_input)
                # sal_final, sal_coarse,x5,x4,x3,x2 = self.net(sal_input)
                # print(sal_final.shape)
                # print(sal_coarse.shape)
                # print(x5.shape)
                # print(x4.shape)
                # print(x3.shape)
                # print(x2.shape)。






                sal_loss_coarse = F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
                sal_loss_final = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                # loss_x5 = F.binary_cross_entropy_with_logits(x5, sal_label, reduction='sum')
                # loss_x4 = F.binary_cross_entropy_with_logits(x4, sal_label, reduction='sum')
                # loss_x3 = F.binary_cross_entropy_with_logits(x3, sal_label, reduction='sum')
                # loss_x2 = F.binary_cross_entropy_with_logits(x2, sal_label, reduction='sum')
                #print('final:',end="")
                #print(sal_loss_final)
                #print('coarse:',end="")
                #print(sal_loss_coarse)


                sal_loss_fuse = sal_loss_final + 256 * sal_loss_coarse #+ loss_x2+loss_x3+loss_x4+loss_x5

                # print(sal_loss_fuse)
                '''

                if sal_loss_fuse>10000:
                    #outlier_img = np.squeeze(sal_image).cpu().data.numpy().
                    outlier_gt = np.squeeze(sal_label).cpu().data.numpy()
                    outlier_ped = np.squeeze(torch.sigmoid(sal_final)).cpu().data.numpy()
                    print(sal_loss_fuse)
                    print(sal_loss_final)
                    print(256 * sal_loss_coarse)
                    shown(outlier_gt)
                    shown(outlier_ped)

                sal_loss_fuse.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                '''
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)  # 积累多少样本
                # if (i+1)%2==1:
                #     last_loss = sal_loss
                # if (i+1)%2==0:
                #     sal_loss = (sal_loss+last_loss)/2
                #     r_sal_loss +=last_loss.data
                #     r_sal_loss += sal_loss.data
                #
                #     sal_loss.backward()
                #     # self.optimizer.step()
                #     # self.optimizer.zero_grad()
                #     aveGrad += 1
                r_sal_loss += sal_loss.data

                sal_loss.backward()
                # self.optimizer.step()
                # self.optimizer.zero_grad()
                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size)))
                    file.write('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f\n' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size)))
                    print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      (epoch) * len(self.train_loader.dataset) + i)
                    r_sal_loss = 0



            if (epoch) % self.config.epoch_save == 0 :
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch))

        # save model
        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)
        file.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))
