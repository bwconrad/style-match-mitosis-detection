import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19


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


class BFG(nn.Module):
    def __init__(self):
        super().__init__()
        n = 4

        self.pool_layers = nn.ModuleList([])
        for i in range(1, n):
            self.pool_layers.append(nn.MaxPool2d(2**i, 2**i))

    def forward(self, features):
        features = list(reversed(features))

        # Downscale all features to same size
        scaled = [features[0]]
        for f, pool in zip(features[1:], self.pool_layers):
            scaled.append(pool(f))

        # Concat features
        out = torch.cat(scaled, dim=1)

        return out


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


class Decoder(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(in_ch, 256),
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


class UNetDecoder(nn.Module):
    def __init__(self, in_ch=512):
        super().__init__()

        # Decoder blocks
        self.d1 = nn.Sequential(
            ConvBlock(in_ch, 256), nn.Upsample(scale_factor=2, mode="nearest")
        )
        self.d2 = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.d3 = nn.Sequential(
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.d4 = nn.Sequential(ConvBlock(128, 64), ConvBlock(64, 3, use_relu=False))

        # Skip connections
        self.in1 = nn.InstanceNorm2d(256)
        self.ada_in1 = AdaIn()
        self.in2 = nn.InstanceNorm2d(128)
        self.ada_in2 = AdaIn()
        self.in3 = nn.InstanceNorm2d(64)
        self.ada_in3 = AdaIn()

    def forward(self, x, c, s):
        c4, c3, c2, _ = c
        s4, s3, s2, _ = s

        d1 = self.d1(x)

        skip1 = self.ada_in1(self.in1(c2), s2)
        d1 = torch.cat([d1, skip1], dim=1)
        d2 = self.d2(d1)

        skip2 = self.ada_in2(self.in2(c3), s3)
        d2 = torch.cat([d2, skip2], dim=1)
        d3 = self.d3(d2)

        skip3 = self.ada_in3(self.in3(c4), s4)
        d3 = torch.cat([d3, skip3], dim=1)
        d4 = self.d4(d3)

        return d4


class AdaInNetwork(nn.Module):
    def __init__(self, use_bfg=False, use_skip=False):
        super().__init__()
        self.use_bfg = use_bfg
        self.use_skip = use_skip

        # Encoder
        self.encoder = VGGEncoder()

        # Bottleneck
        if use_bfg:
            self.bfg = BFG()
            self.conv = nn.Conv2d(960, 512, kernel_size=1)
        self.ada_in = AdaIn()

        # Decoder
        if use_skip:
            self.decoder = UNetDecoder()
        else:
            self.decoder = Decoder()

    def forward(self, c, s=None, f_s=None, alpha=1.0):
        # Encode content and style images
        f_c = self.encoder(c, return_all=True)
        if s is not None:
            f_s = self.encoder(s, return_all=True)
        else:
            # Use inputted style features
            assert f_s is not None
            if len(f_s[0].size()) == 3:
                for i in range(len(f_s)):
                    # Repeat along batch dimension
                    f_s[i] = f_s[i].unsqueeze(0).repeat(c.size()[0], 1, 1, 1)

        # Bottleneck
        if self.use_bfg:
            f_c_cat = self.bfg(f_c)
            f_s_cat = self.bfg(f_s)
            t = self.ada_in(f_c_cat, f_s_cat)
            t = alpha * t + (1 - alpha) * f_c_cat
            t_in = self.conv(t)
        else:
            t = self.ada_in(f_c[-1], f_s[-1])
            t = alpha * t + (1 - alpha) * f_c[-1]
            t_in = t

        # Decode stylized image
        if self.use_skip:
            g_t = self.decoder(t_in, f_c, f_s)
        else:
            g_t = self.decoder(t_in)

        return g_t, t, f_s
