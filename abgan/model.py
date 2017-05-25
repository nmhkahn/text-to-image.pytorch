import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        net = list()

        net.append(conv_transpose2d(128, 512, 4, 1, 0))
        net.append(nn_conv2d(512, 256, 3, 1, 1))
        net.append(nn_conv2d(256, 128, 3, 1, 1))
        net.append(nn_conv2d(128, 64, 3, 1, 1))
        net.append(nn_conv2d(64, 64, 3, 1, 1))
        net.append(nn_conv2d(64, 64, 3, 1, 1))

        net.append(conv2d(64, 64, 3, 1, 1))
        net.append(conv2d(64, 64, 3, 1, 1))

        net.append(nn.Conv2d(64, 3, 3, 1, 1))
        net.append(nn.Tanh())

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = x.view(x.size(0), -1, 1, 1)
        for layer in self.net:
            out = layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        net = list()

        net.append(nn.Conv2d(3, 64, 4, 2, 1))
        net.append(nn.LeakyReLU(0.2, inplace=True))

        net.append(conv2d(64, 128, 4, 2, 1))
        net.append(conv2d(128, 256, 4, 2, 1))
        net.append(conv2d(256, 512, 4, 2, 1))
        net.append(conv2d(512, 512, 4, 2, 1))

        net.append(nn.Conv2d(512, 1, 4, 1, 0))
        net.append(nn.Sigmoid())

        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = x
        for layer in self.net:
            out = layer(out)
        return out.view(out.size(0))


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1):
    conv = nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=False)
    bn    = nn.BatchNorm2d(channel_out)
    lrelu = nn.LeakyReLU(0.2, inplace=True)
    init.kaiming_normal(conv.weight)

    return nn.Sequential(conv, bn, lrelu)


def conv_transpose2d(channel_in, channel_out,
                     ksize=4, stride=2, padding=1):
    conv = nn.ConvTranspose2d(channel_in, channel_out,
                              ksize, stride, padding,
                              bias=False)
    bn    = nn.BatchNorm2d(channel_out)
    lrelu = nn.LeakyReLU(0.2, inplace=True)
    init.kaiming_normal(conv.weight)

    return nn.Sequential(conv, bn, lrelu)

def nn_conv2d(channel_in, channel_out,
              ksize=3, stride=1, padding=1,
              scale_factor=2):
    up    = nn.UpsamplingNearest2d(scale_factor=scale_factor)
    conv  = nn.Conv2d(channel_in, channel_out, ksize, stride, padding)

    bn    = nn.BatchNorm2d(channel_out)
    lrelu = nn.LeakyReLU(0.2, inplace=True)
    init.kaiming_normal(conv.weight)

    return nn.Sequential(up, conv, bn, lrelu)
