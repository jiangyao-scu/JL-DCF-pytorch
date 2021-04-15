import torch
from torch import nn
import torch.nn.functional as F
import re

from .resnet import ResNet, Bottleneck
from .vgg import *
from .densenet import densenet161

k = 64


class JLModule(nn.Module):
    def __init__(self, backbone):
        super(JLModule, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(1024, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2048, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)
        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self, x):
        # put tensor from Resnet backbone to compress model
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)
        for i in range(5):
            feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract  # list of tensor that compress model output


class JLModuleVGG(nn.Module):
    def __init__(self, backbone):
        super(JLModuleVGG, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(256, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(512, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # for vgg16, the layers' names are different with the pretrained vgg16.pth, so the names should to be changed.
        renamed_dict = dict()
        for k, v in pretrained_dict.items():
            k = k.replace('features', 'layers')
            if k in model_dict:
                renamed_dict[k] = v
        # print('vgg16, pretrained:', pretrained_dict.items())
        # print('vgg16, renamed:', renamed_dict.items())
        model_dict.update(renamed_dict)
        # print('model, update:', model_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        # put tensor from VGG backbone to compress model
        feature_extract = []
        x = self.backbone(x)
        for i in range(6):
            feature_extract.append(self.CP[i](x[i]))
        return feature_extract  # list of tensor that compress model output

class JLModuleDensenet(nn.Module):
    def __init__(self, backbone):
        super(JLModuleDensenet, self).__init__()
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.vgg_conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=True))
        cp = []

        cp.append(nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(96, 128, 3, 1, 1), self.relu, nn.Conv2d(128, 128, 3, 1, 1), self.relu,
                                nn.Conv2d(128, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(384, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(768, 256, 5, 1, 2), self.relu, nn.Conv2d(256, 256, 5, 1, 2), self.relu,
                                nn.Conv2d(256, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2112, 512, 5, 1, 2), self.relu, nn.Conv2d(512, 512, 5, 1, 2), self.relu,
                                nn.Conv2d(512, k, 3, 1, 1), self.relu))
        cp.append(nn.Sequential(nn.Conv2d(2208, 512, 7, 1, 6, 2), self.relu, nn.Conv2d(512, 512, 7, 1, 6, 2),
                                self.relu, nn.Conv2d(512, k, 3, 1, 1), self.relu))
        self.CP = nn.ModuleList(cp)

    def load_pretrained_model(self, model_path):

        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        self.backbone.load_state_dict(state_dict)

        self.vgg_conv1.load_state_dict(torch.load('pretrained/vgg_conv1.pth'), strict=True)

    def forward(self, x):
        # put tensor from Resnet backbone to compress model
        feature_extract = []
        feature_extract.append(self.CP[0](self.vgg_conv1(x)))
        x = self.backbone(x)

        for i in range(5):
            feature_extract.append(self.CP[i + 1](x[i]))
        return feature_extract  # list of tensor that compress model output

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
        self.conv_branch1 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu)
        self.conv_branch2 = nn.Sequential(nn.Conv2d(k, int(k / 2), 1), self.relu,
                                          nn.Conv2d(int(k / 2), int(k / 4), 3, 1, 1), self.relu)
        self.conv_branch3 = nn.Sequential(nn.Conv2d(k, int(k / 4), 1), self.relu,
                                          nn.Conv2d(int(k / 4), int(k / 4), 5, 1, 2), self.relu)
        self.conv_branch4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), nn.Conv2d(k, int(k / 4), 1), self.relu)

    def forward(self, x_cm, x_fa):
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
    def __init__(self, base_model_cfg, JLModule, cm_layers, feature_aggregation_module, JL_score_layers,
                 DCF_score_layers, upsampling):
        super(JL_DCF, self).__init__()
        self.base_model_cfg = base_model_cfg
        self.JLModule = JLModule
        self.FA = nn.ModuleList(feature_aggregation_module)
        self.upsampling = nn.ModuleList(nn.ModuleList(upsampling[i]) for i in range(0, 4))
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
        for i in range(len(x_cm) - 2):
            for j in range(len(x_fa)):
                x_fa_temp[j] = self.upsampling[i][i - j](x_fa[j])
            x_fa.append(self.FA[3 - i](x_cm[i + 2], x_fa_temp))
            x_fa_temp.append(x_fa[-1])

        s_final = self.score_DCF(x_fa[-1])
        return s_final, s_coarse


def build_model(network='resnet101', base_model_cfg='resnet'):
    feature_aggregation_module = []
    for i in range(5):
        feature_aggregation_module.append(FAModule())
    upsampling = []
    for i in range(0, 4):
        upsampling.append([])
        for j in range(0, i + 1):
            upsampling[i].append(
                nn.ConvTranspose2d(k, k, kernel_size=2 ** (j + 2), stride=2 ** (j + 1), padding=2 ** (j)))
    if base_model_cfg == 'resnet':
        parameter = [3, 4, 23, 3] if network == 'resnet101' else [3, 4, 6, 3]
        backbone = ResNet(Bottleneck, parameter)
        return JL_DCF(base_model_cfg, JLModule(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
    elif base_model_cfg == 'vgg':
        backbone = vgg(network=network)
        return JL_DCF(base_model_cfg, JLModuleVGG(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                      ScoreLayer(k), upsampling)
    elif base_model_cfg == 'densenet':
        backbone = densenet161()
        return JL_DCF(base_model_cfg, JLModuleDensenet(backbone), CMLayer(), feature_aggregation_module, ScoreLayer(k),
                  ScoreLayer(k), upsampling)
