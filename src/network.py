import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


class AdaIn(nn.Module):
    def __init__(self):
        super().__init__()

    def get_mean_std(self, x):
        mean = torch.mean(x, axis=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True) + 1e-6
        return mean, std

    def forward(self, c, s):
        assert c.size() == s.size()
        c_mean, c_std = self.get_mean_std(c)
        s_mean, s_std = self.get_mean_std(s)
        c_norm = s_std * ((c - c_mean) / c_std) + s_mean
        return c_norm


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features

        # Split VGG into blocks to extract activations
        self.relu1_1 = vgg[:2]
        self.relu2_1 = vgg[2:7]
        self.relu3_1 = vgg[7:12]
        self.relu4_1 = vgg[12:21]

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, return_all=False):
        y1 = self.relu1_1(x)
        y2 = self.relu2_1(y1)
        y3 = self.relu3_1(y2)
        y4 = self.relu4_1(y3)

        if return_all:
            return [y1, y2, y3, y4]
        else:
            return y4


class ConvBlock(nn.Module):
    """
    Reflection pad -> Conv -> ReLU
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, use_relu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.pad = nn.ReflectionPad2d((padding, padding, padding, padding))
        self.use_relu = use_relu

    def forward(self, x):
        out = self.conv(self.pad(x))
        if self.use_relu:
            out = F.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(512, 256),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(128, 128),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(64, 64),
            ConvBlock(64, 3, use_relu=False),
        )

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    x = torch.rand((4, 3, 256, 256))
    e = VGGEncoder()
    i = AdaIn()
    d = Decoder()
    y = e(x)
    t = i(y, y)
    out = d(t)
    print(y.size())
    print(t.size())
    print(out.size())
