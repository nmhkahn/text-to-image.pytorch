import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv2d(channel_in, channel_out, 3, 1, 1),
            conv2d(channel_out, channel_out, 3, 1, 1)
        )

    def forward(self, x):
        residual = x
        out = self.block(x)

        out += residual
        out = self.relu(out)
        return out


def linear(channel_in, channel_out,
           activation=nn.LeakyReLU,
           normalizer=nn.BatchNorm1d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Linear(channel_in, channel_out, bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.LeakyReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv_transpose2d(channel_in, channel_out,
                     ksize=4, stride=2, padding=1,
                     activation=nn.LeakyReLU,
                     normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.ConvTranspose2d(channel_in, channel_out,
                              ksize, stride, padding,
                              bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def nn_conv2d(channel_in, channel_out,
              ksize=3, stride=1, padding=1,
              scale_factor=2,
              activation=nn.LeakyReLU,
              normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.UpsamplingNearest2d(scale_factor=scale_factor))
    layer.append(nn.Conv2d(channel_in, channel_out,
                           ksize, stride, padding,
                           bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[1].weight)

    return nn.Sequential(*layer)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        if activation.__name__ == "LeakyReLU":
            layer.append(activation(0.2))
        else:
            layer.append(activation())
    return layer
