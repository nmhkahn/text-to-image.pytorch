import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256),
        )

        self.net_joint = nn.Sequential(
            conv_transpose2d(100+256, 512, 4, 1, 0),
            nn_conv2d(512, 256, 3, 1, 1),
            nn_conv2d(256, 128, 3, 1, 1),
            nn_conv2d(128, 64, 3, 1, 1),
            nn_conv2d(64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1),
            conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, text):
        text = self.net_text(text)
        out  = torch.cat([z, text], dim=1)

        out = out.view(out.size(0), -1, 1, 1)
        out = self.net_joint(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256)
        )

        self.net_image = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # conv2, conv3, conv4
            conv2d(64, 128, 4, 2, 1),
            conv2d(128, 256, 4, 2, 1),
            conv2d(256, 512, 4, 2, 1)
        )

        self.net_joint = nn.Sequential(
            conv2d(512+256, 512, 3, 1, 1),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x, text):
        x = self.net_image(x)

        text = self.net_text(text)
        # replicate text embedding to fit image dimension
        text = text.view(text.size(0), text.size(1), 1, 1)
        text = text.repeat(1, 1, x.size(2), x.size(3))

        out = torch.cat([x, text], dim=1)
        out = self.net_joint(out)
        return out.view(out.size(0))


def linear(channel_in, channel_out):
    h     = nn.Linear(channel_in, channel_out, bias=False)
    bn    = nn.BatchNorm1d(channel_out)
    lrelu = nn.LeakyReLU(0.2, inplace=True)
    init.kaiming_normal(h.weight)

    return nn.Sequential(h, bn, lrelu)


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
