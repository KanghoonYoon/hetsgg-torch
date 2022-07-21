
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

import torchvision.models as models
from hetsgg.layers import FrozenBatchNorm2d
from hetsgg.layers import Conv2d
from hetsgg.layers import DFConv2d
from hetsgg.modeling.make_layers import group_norm
from hetsgg.utils.registry import Registry


class VGG16(nn.Module):
    def __init__(self, cfg):
        super(VGG16, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.conv_body = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    def forward(self, x):
        output = []
        output.append(self.conv_body(x))
        return output

