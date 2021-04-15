import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, init_weights=False):
        super(VGG, self).__init__()
        # self.features = features
        # self.layers = []
        # for layer in features:
        #     self.layers.append(nn.Sequential(layer))
        self.layers = features
        self.layers = nn.ModuleList(self.layers)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        tmp_x = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            # only for vgg16
            if i == 3 or i == 8 or i == 15 or i == 22 or i == 29 or i == 30:
                tmp_x.append(x)
        return tmp_x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    layer = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += layer
            layer = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == "M5":
            layers += layer
            # layers.append([nn.MaxPool2d(kernel_size=3, stride=1, padding=1)])
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layer += [conv2d, nn.ReLU(True)]
            in_channels = v
    return layers


cfgs = {
    # 'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M5'],
    # 'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M5'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M5'],
    # 'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M5'],
}


def vgg(network="vgg16", **kwargs):
    try:
        cfg = cfgs[network]
    except:
        print("Warning: model {} not in cfgs dict!".format(network))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
