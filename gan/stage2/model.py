import torch
import torch.nn as nn
from gan.ops import *

class Stage1Generator(nn.Module):
    def __init__(self):
        super(Stage1Generator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256),
        )

        self.net_joint = nn.Sequential(
            conv_transpose2d(100+256, 512, 4, 1, 0),
            nn_conv2d(512, 256, 3, 1, 1),
            nn_conv2d(256, 128, 3, 1, 1),
            nn_conv2d(128, 128, 3, 1, 1),
            nn_conv2d(128, 128, 3, 1, 1),
            conv2d(128, 128, 3, 1, 1),
            conv2d(128, 3, 3, 1, 1, activation=nn.Tanh, normalizer=None)
        )

    def forward(self, z, text):
        text = self.net_text(text)
        out  = torch.cat([z, text], dim=1)

        out = out.view(out.size(0), -1, 1, 1)
        out = self.net_joint(out)
        return out


class Stage2Generator(nn.Module):
    def __init__(self):
        super(Stage2Generator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256),
        )

        self.net_encoder = nn.Sequential(
            conv2d(3, 128, 3, 1, 1),
            conv2d(128, 256, 4, 2, 1),
            conv2d(256, 512, 4, 2, 1),
        )

        self.net_joint = nn.Sequential(
            conv2d(512+256, 512, 3, 1, 1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn_conv2d(512, 256, 3, 1, 1),
            nn_conv2d(256, 128, 3, 1, 1),
            nn_conv2d(128, 128, 3, 1, 1),
            nn_conv2d(128, 128, 3, 1, 1),
            conv2d(128, 64, 3, 1, 1),
            conv2d(64, 3, 3, 1, 1, activation=nn.Tanh, normalizer=None)
        )

    def forward(self, x, text):
        out_text    = self.net_text(text)
        out_encoder = self.net_encoder(x)

        # replicate text embedding to fit image dimension
        out_text = out_text.view(out_text.size(0), out_text.size(1), 1, 1)
        out_text = out_text.repeat(1, 1, out_encoder.size(2), out_encoder.size(3))

        out = torch.cat([out_encoder, out_text], dim=1)
        out = self.net_joint(out)
        return out


class Stage1Discriminator(nn.Module):
    def __init__(self):
        super(Stage1Discriminator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256)
        )

        self.net_image = nn.Sequential(
            conv2d(3, 64, 4, 2, 1, normalizer=None),
            conv2d(64, 128, 4, 2, 1),
            conv2d(128, 256, 4, 2, 1),
            conv2d(256, 512, 4, 2, 1)
        )

        self.net_joint = nn.Sequential(
            conv2d(512+256, 512, 3, 1, 1),
            conv2d(512, 1, 4, 1, 0, activation=nn.Sigmoid, normalizer=None)
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


class Stage2Discriminator(nn.Module):
    def __init__(self):
        super(Stage2Discriminator, self).__init__()

        self.net_text = nn.Sequential(
            linear(4800, 1024),
            linear(1024, 256)
        )

        self.net_image = nn.Sequential(
            conv2d(3, 64, 4, 2, 1, normalizer=None), # 128
            conv2d(64, 128, 4, 2, 1), # 64
            conv2d(128, 256, 4, 2, 1), # 32
            conv2d(256, 512, 4, 2, 1), # 16
            conv2d(512, 1024, 4, 2, 1), # 8
            conv2d(1024, 1024, 4, 2, 1) # 4
        )

        self.net_joint = nn.Sequential(
            conv2d(1024+256, 512, 3, 1, 1),
            conv2d(512, 1, 4, 1, 0, activation=nn.Sigmoid, normalizer=None)
        )

    def forward(self, x, text):
        out = self.net_image(x)
        text = self.net_text(text)

        # replicate text embedding to fit image dimension
        text = text.view(text.size(0), text.size(1), 1, 1)
        text = text.repeat(1, 1, out.size(2), out.size(3))

        out = torch.cat([out, text], dim=1)
        out = self.net_joint(out)
        return out.view(out.size(0))
