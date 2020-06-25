import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

from .resnet import resnet101_locate
#from .vgg import vgg16_locate

k=64


class CMLayer(nn.Module):
    def __init__(self):
        super(CMLayer, self).__init__()

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            part1 = list_x[i][0]
            part2 = list_x[i][1]
            sum = (part1 + part2 + (part1 * part2)).unsqueeze(dim=0)
            resl.append(sum)
        return resl


class FAModule(nn.Module):
    def __init__(self):
        super(FAModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k/4), 1), self.relu)
        self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k/2), 1), self.relu, nn.Conv2d(int(k/2), int(k/4), 3, 1, 1), self.relu)
        self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k/4), 1), self.relu, nn.Conv2d(int(k/4), int(k/4), 5, 1, 2), self.relu)
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k/4), 1), self.relu)

    def forward(self, x_cm, x_fa ):
        # element-wise addition
        x = x_cm

        for i in range(len(x_fa)):
            x += x_fa[i]
        # aggregation
        x_branch1 = self.conv_branch1(x)
        x_branch2 = self.conv_branch2(x)
        x_branch3 = self.conv_branch3(x)
        x_branch4 = self.conv_branch4(x)

        x = torch.cat((x_branch1, x_branch2, x_branch3, x_branch4), dim=1)
        return x




class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x


class JL_DCF(nn.Module):
    def __init__(self, base_model_cfg, JLModule, cm_layers, feature_aggregation_module, JL_score_layers,DCF_score_layers,upsampling):
        super(JL_DCF, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.JLModule = JLModule
        self.FA = nn.ModuleList(feature_aggregation_module)
        self.upsampling = nn.ModuleList(nn.ModuleList(upsampling[i]) for i in range(0,4))
        self.score_JL = JL_score_layers
        self.score_DCF = DCF_score_layers
        self.cm = cm_layers

    def forward(self, x):
        x = self.JLModule(x)
        x_cm = self.cm(x)
        s_coarse = self.score_JL(x[5])
        x_cm = x_cm[::-1]
        x_fa = []
        x_fa_temp = []
        x_fa.append(self.FA[4](x_cm[1], x_cm[0]))
        x_fa_temp.append(x_fa[0])
        for i in range(len(x_cm)-2):
            for j in range(len(x_fa)):
                x_fa_temp[j] = self.upsampling[i][i-j](x_fa[j])
            # for a in x_fa_temp:
            #     print(a.shape)
            x_fa.append(self.FA[3-i](x_cm[i+2], x_fa_temp))
            x_fa_temp.append(x_fa[-1])

        s_final = self.score_DCF(x_fa[-1])
        return s_final, s_coarse

# class MODE(nn.Module):
#     def __init__(self, jl_dcf):
#         super(MODE, self).__init__()
#         self.jl_dcf = jl_dcf
#         self.conv1 = nn.Conv2d(k, 1, 1, 1)
#         self.conv2 = nn.Conv2d(k, 1, 1, 1)
#         self.conv3 = nn.Conv2d(k, 1, 1, 1)
#         self.conv4 = nn.Conv2d(k, 1, 1, 1)
#
#     def forward(self, x):
#         s_final, s_coarse  = self.jl_dcf(x)
#         return s_final, s_coarse

def build_model(base_model_cfg='resnet'):
    feature_aggregation_module = []
    for i in range(5):
        feature_aggregation_module.append(FAModule())
    upsampling = []
    for i in range(0,4):
        upsampling.append([])
        for j in range(0,i+1):
            upsampling[i].append(nn.ConvTranspose2d(64,64,kernel_size=2**(j+2),stride=2**(j+1),padding=2**(j)))
    if base_model_cfg == 'vgg':
        return JL_DCF(base_model_cfg, vgg16_locate(), CMLayer(), DCFlLayer(), ScoreLayer(k))
    elif base_model_cfg == 'resnet':
        return JL_DCF(base_model_cfg, resnet101_locate(), CMLayer(), feature_aggregation_module, ScoreLayer(k),ScoreLayer(k),upsampling)
        # return JL_DCF(base_model_cfg, resnet101_locate(), CMLayer(), DCFlLayer(), ScoreLayer(k))



