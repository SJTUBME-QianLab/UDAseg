import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Shared_Encoder import conv1x1, outconv
from util.utils import UpsampleDeterministic, CBAM, BAM, CBAMParallel, CBAMNoActivate


class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate=0):
        super(DenseLayer, self).__init__()
        self.add_module("norm1", nn.InstanceNorm2d(num_input_features)),
        self.add_module("relu1", nn.LeakyReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features, growth_rate, kernel_size=1, stride=1, bias=False
            ),
        ),
        self.add_module("norm2", nn.InstanceNorm2d(growth_rate)),
        self.add_module("relu2", nn.LeakyReLU(inplace=True)),
        self.add_module(
            "conv2",
            nn.Conv2d(
                growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
            ),
        ),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("norm", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlock, self).forward(x)
        return new_features


class DenseBlockCBAM(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockCBAM, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("cbam", CBAM(num_output_features))
        self.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlockCBAM, self).forward(x)
        return new_features


class DenseBlockCBAMNoActivate(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockCBAMNoActivate, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("cbam", CBAMNoActivate(num_output_features))
        # self.add_module('tanh', nn.Tanh())

    def forward(self, x):
        new_features = super(DenseBlockCBAMNoActivate, self).forward(x)
        return new_features


class DenseBlockCBAMParallel(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockCBAMParallel, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("cbam", CBAMParallel(num_output_features))
        self.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlockCBAMParallel, self).forward(x)
        return new_features


class DenseBlockInCBAMParallel(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockInCBAMParallel, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
            if (i + 1) % 2 == 0:
                self.add_module(
                    "cbam%d" % (i + 1),
                    CBAMParallel(num_input_features + (i + 1) * growth_rate),
                )

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlockInCBAMParallel, self).forward(x)
        return new_features


class DenseBlockTwoCBAM(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockTwoCBAM, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)
            if (i + 1) % 2 == 0:
                self.add_module(
                    "cbam%d" % (i + 1), CBAM(num_input_features + (i + 1) * growth_rate)
                )

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlockTwoCBAM, self).forward(x)
        return new_features


class DenseBlockBAM(nn.Sequential):
    def __init__(
        self,
        num_input_features,
        num_output_features,
        growth_rate=32,
        drop_rate=0,
        num_layers=4,
    ):
        super(DenseBlockBAM, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate, growth_rate, drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)

        self.add_module(
            "conv",
            nn.Conv2d(
                growth_rate * 4 + num_input_features,
                num_output_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.add_module("norm", nn.InstanceNorm2d(num_output_features))
        self.add_module("bam", BAM(num_output_features))
        self.add_module("relu", nn.LeakyReLU(inplace=True))

    def forward(self, x):
        new_features = super(DenseBlockBAM, self).forward(x)
        return new_features


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.de_conv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


class Encoder(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAM(num_channels, flt)
        self.down2 = DenseBlockCBAM(flt, flt * 2)
        self.down3 = DenseBlockCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAM(flt * 4, flt * 8)
        self.down1_bam = DenseBlockBAM(num_channels, flt)
        self.down2_bam = DenseBlockBAM(flt, flt * 2)
        self.down3_bam = DenseBlockBAM(flt * 2, flt * 4)
        self.down4_bam = DenseBlockBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAM(flt * 8, flt * 8)
        self.bottom_bam = DenseBlockBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        if self.attention == "CBAM":
            down1_feat = self.down1(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom(pool4_feat)
        else:
            down1_feat = self.down1_bam(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2_bam(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3_bam(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4_bam(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom_bam(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Decoder(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_conv1_bam = DenseBlockBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_conv2_bam = DenseBlockBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_conv3_bam = DenseBlockBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.up_conv4_bam = DenseBlockBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):

        if self.attention == "CBAM":
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        else:
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1_bam(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2_bam(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3_bam(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4_bam(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class DecoderMultiOut(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(DecoderMultiOut, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_conv1_bam = DenseBlockBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_conv2_bam = DenseBlockBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_conv3_bam = DenseBlockBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.up_conv4_bam = DenseBlockBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        inputs = self.conv1x1(inputs)

        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        side6 = UpsampleDeterministic(upscale=8)(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        side7 = UpsampleDeterministic(upscale=4)(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        side8 = UpsampleDeterministic(upscale=2)(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(up4_feat)

        my_list = [out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)

        return out6, out7, out8, out9, out10


class EncoderCBAMParallel(nn.Module):
    def __init__(self, num_channels=1, attention="CBAMP"):
        super(EncoderCBAMParallel, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAMParallel(num_channels, flt)
        self.down2 = DenseBlockCBAMParallel(flt, flt * 2)
        self.down3 = DenseBlockCBAMParallel(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAMParallel(flt * 4, flt * 8)
        self.down1_in_cbam = DenseBlockInCBAMParallel(num_channels, flt)
        self.down2_in_cbam = DenseBlockInCBAMParallel(flt, flt * 2)
        self.down3_in_cbam = DenseBlockInCBAMParallel(flt * 2, flt * 4)
        self.down4_in_cbam = DenseBlockInCBAMParallel(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAMParallel(flt * 8, flt * 8)
        self.bottom_in_cbam = DenseBlockInCBAMParallel(flt * 8, flt * 8)

    def forward(self, inputs):
        if self.attention == "CBAMP":
            down1_feat = self.down1(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom(pool4_feat)
        else:
            down1_feat = self.down1_in_cbam(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2_in_cbam(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3_in_cbam(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4_in_cbam(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom_in_cbam(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class DecoderCBAMParallel(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAMP"):
        super(DecoderCBAMParallel, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAMParallel(flt * 16, flt * 8)
        self.up_conv1_in_cbam = DenseBlockInCBAMParallel(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAMParallel(flt * 8, flt * 4)
        self.up_conv2_in_cbam = DenseBlockInCBAMParallel(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAMParallel(flt * 4, flt * 2)
        self.up_conv3_in_cbam = DenseBlockInCBAMParallel(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAMParallel(flt * 2, flt)
        self.up_conv4_in_cbam = DenseBlockInCBAMParallel(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):

        if self.attention == "CBAMP":
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        else:
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1_in_cbam(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2_in_cbam(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3_in_cbam(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4_in_cbam(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class DenseUnetCBAM(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAM, self).__init__()
        self.enc = Encoder(inchannels, attention="CBAM")
        self.dec = Decoder(512, attention="CBAM")

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class DenseUnetMeta(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetMeta, self).__init__()
        self.enc = Encoder_Dense_Meta(inchannels, attention="CBAM")
        self.dec = Decoder_Dense_Meta(512, attention="CBAM")

    def forward(self, input):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class DenseUnetMetaMulti(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetMetaMulti, self).__init__()
        self.enc = Encoder_Dense_Meta(inchannels, attention="CBAM")
        self.dec = Decoder_Dense_Meta_Multi_Class(512, attention="CBAM")

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class DenseUnetCBAMOutBottom(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAMOutBottom, self).__init__()
        self.enc = Encoder(inchannels, attention="CBAM")
        self.dec = Decoder(512, attention="CBAM")

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out, bottom_feat


class DenseUnetCBAMMultiOut(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAMMultiOut, self).__init__()
        self.enc = Encoder(inchannels, attention="CBAM")
        self.dec = DecoderMultiOut(512, attention="CBAM")

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out6, out7, out8, out9, out10 = self.dec(
            bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat
        )
        return out6, out7, out8, out9, out10


class DenseUnetCBAMMulti(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAMMulti, self).__init__()
        self.enc = Encoder(inchannels, attention="CBAM")
        self.dec = Decoder(512, attention="CBAM")

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out, bottom_feat, down4_feat


class DenseUnetBAM(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetBAM, self).__init__()
        self.enc = Encoder(inchannels, attention="BAM")
        self.dec = Decoder(512, attention="BAM")

    def forward(self, input):
        # print("BAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class EncoderTwo(nn.Module):
    def __init__(self, num_channels=1):
        super(EncoderTwo, self).__init__()
        flt = 64
        self.down1 = DenseBlockTwoCBAM(num_channels, flt)
        self.down2 = DenseBlockTwoCBAM(flt, flt * 2)
        self.down3 = DenseBlockTwoCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockTwoCBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAM(flt * 8, flt * 8)
        self.bottom_bam = DenseBlockBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        pool1_feat = self.down_pool1(down1_feat)
        down2_feat = self.down2(pool1_feat)
        pool2_feat = self.down_pool2(down2_feat)
        down3_feat = self.down3(pool2_feat)
        pool3_feat = self.down_pool3(down3_feat)
        down4_feat = self.down4(pool3_feat)
        pool4_feat = self.down_pool4(down4_feat)
        bottom_feat = self.bottom(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class DecoderTwo(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(DecoderTwo, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockTwoCBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockTwoCBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockTwoCBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockTwoCBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        inputs = self.conv1x1(inputs)

        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        side6 = UpsampleDeterministic(upscale=8)(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        side7 = UpsampleDeterministic(upscale=4)(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        side8 = UpsampleDeterministic(upscale=2)(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(up4_feat)

        my_list = [out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class DenseUnetTwoCBAM(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetTwoCBAM, self).__init__()
        self.enc = EncoderTwo(inchannels)
        self.dec = DecoderTwo(512)

    def forward(self, input):
        # print("CBAM")
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class DenseUnetCBAMParallel(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAMParallel, self).__init__()
        self.enc = EncoderCBAMParallel(inchannels, attention="CBAMP")
        self.dec = DecoderCBAMParallel(512, attention="CBAMP")

    def forward(self, input):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class DenseUnetInCBAMParallel(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetInCBAMParallel, self).__init__()
        self.enc = EncoderCBAMParallel(inchannels, attention="CBAMINP")
        self.dec = DecoderCBAMParallel(512, attention="CBAMINP")

    def forward(self, input):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


class Encoder_Dense(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder_Dense, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAM(num_channels, flt)
        self.down2 = DenseBlockCBAM(flt, flt * 2)
        self.down3 = DenseBlockCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAM(flt * 4, flt * 8)
        self.down1_bam = DenseBlockBAM(num_channels, flt)
        self.down2_bam = DenseBlockBAM(flt, flt * 2)
        self.down3_bam = DenseBlockBAM(flt * 2, flt * 4)
        self.down4_bam = DenseBlockBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAM(flt * 8, flt * 8)
        self.bottom_bam = DenseBlockBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        if self.attention == "CBAM":
            down1_feat = self.down1(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom(pool4_feat)
        else:
            down1_feat = self.down1_bam(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2_bam(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3_bam(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4_bam(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom_bam(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Encoder_Dense_Meta(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder_Dense_Meta, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAM(num_channels, flt)
        self.down2 = DenseBlockCBAM(flt, flt * 2)
        self.down3 = DenseBlockCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        pool1_feat = self.down_pool1(down1_feat)
        down2_feat = self.down2(pool1_feat)
        pool2_feat = self.down_pool2(down2_feat)
        down3_feat = self.down3(pool2_feat)
        pool3_feat = self.down_pool3(down3_feat)
        down4_feat = self.down4(pool3_feat)
        pool4_feat = self.down_pool4(down4_feat)
        bottom_feat = self.bottom(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Encoder_Dense_Tanh(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder_Dense_Tanh, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAM(num_channels, flt)
        self.down2 = DenseBlockCBAM(flt, flt * 2)
        self.down3 = DenseBlockCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAM(flt * 4, flt * 8)
        self.down1_bam = DenseBlockBAM(num_channels, flt)
        self.down2_bam = DenseBlockBAM(flt, flt * 2)
        self.down3_bam = DenseBlockBAM(flt * 2, flt * 4)
        self.down4_bam = DenseBlockBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAMNoActivate(flt * 8, flt * 8)
        self.bottom_bam = DenseBlockBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        if self.attention == "CBAM":
            down1_feat = self.down1(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom(pool4_feat)
        else:
            down1_feat = self.down1_bam(inputs)
            pool1_feat = self.down_pool1(down1_feat)
            down2_feat = self.down2_bam(pool1_feat)
            pool2_feat = self.down_pool2(down2_feat)
            down3_feat = self.down3_bam(pool2_feat)
            pool3_feat = self.down_pool3(down3_feat)
            down4_feat = self.down4_bam(pool3_feat)
            pool4_feat = self.down_pool4(down4_feat)
            bottom_feat = self.bottom_bam(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Encoder_Dense_NoIf(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder_Dense_NoIf, self).__init__()
        flt = 64
        self.attention = attention
        self.down1 = DenseBlockCBAM(num_channels, flt)
        self.down2 = DenseBlockCBAM(flt, flt * 2)
        self.down3 = DenseBlockCBAM(flt * 2, flt * 4)
        self.down4 = DenseBlockCBAM(flt * 4, flt * 8)
        self.down_pool1 = nn.Conv2d(flt, flt, kernel_size=3, stride=2, padding=1)
        self.down_pool2 = nn.Conv2d(
            flt * 2, flt * 2, kernel_size=3, stride=2, padding=1
        )
        self.down_pool3 = nn.Conv2d(
            flt * 4, flt * 4, kernel_size=3, stride=2, padding=1
        )
        self.down_pool4 = nn.Conv2d(
            flt * 8, flt * 8, kernel_size=3, stride=2, padding=1
        )
        self.bottom = DenseBlockCBAM(flt * 8, flt * 8)

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        pool1_feat = self.down_pool1(down1_feat)
        down2_feat = self.down2(pool1_feat)
        pool2_feat = self.down_pool2(down2_feat)
        down3_feat = self.down3(pool2_feat)
        pool3_feat = self.down_pool3(down3_feat)
        down4_feat = self.down4(pool3_feat)
        pool4_feat = self.down_pool4(down4_feat)
        bottom_feat = self.bottom(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Decoder_Dense_NoIf(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder_Dense_NoIf, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        # self.up_conv1_bam = DenseBlockBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        # self.up_conv2_bam = DenseBlockBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        # self.up_conv3_bam = DenseBlockBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        # self.up_conv4_bam = DenseBlockBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        inputs = self.conv1x1(inputs)

        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        side6 = UpsampleDeterministic(upscale=8)(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        side7 = UpsampleDeterministic(upscale=4)(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        side8 = UpsampleDeterministic(upscale=2)(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        out6 = self.out6(side6)
        out7 = self.out7(side7)
        out8 = self.out8(side8)
        out9 = self.out9(up4_feat)

        my_list = [out6, out7, out8, out9]
        out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class Decoder_Dense(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder_Dense, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_conv1_bam = DenseBlockBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_conv2_bam = DenseBlockBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_conv3_bam = DenseBlockBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.up_conv4_bam = DenseBlockBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):

        if self.attention == "CBAM":
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        else:
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1_bam(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2_bam(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3_bam(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4_bam(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class Decoder_Dense_Meta(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder_Dense_Meta, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        inputs = self.conv1x1(inputs)

        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        output = self.final(up4_feat)

        return output


class Decoder_Dense_Meta_Multi_Class(nn.Module):
    def __init__(self, inchannels=512, num_channels=5, attention="CBAM"):
        super(Decoder_Dense_Meta_Multi_Class, self).__init__()
        flt = 64
        self.attention = attention
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.final = nn.Conv2d(flt, num_channels, kernel_size=1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        inputs = self.conv1x1(inputs)

        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)

        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)

        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)

        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        output = self.final(up4_feat)

        return output


class Decoder_Dense_Tanh(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder_Dense_Tanh, self).__init__()
        flt = 64
        self.attention = attention
        self.leaky_relu = nn.LeakyReLU(inplace=False)
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = DenseBlockCBAM(flt * 16, flt * 8)
        self.up_conv1_bam = DenseBlockBAM(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = DenseBlockCBAM(flt * 8, flt * 4)
        self.up_conv2_bam = DenseBlockBAM(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = DenseBlockCBAM(flt * 4, flt * 2)
        self.up_conv3_bam = DenseBlockBAM(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = DenseBlockCBAM(flt * 2, flt)
        self.up_conv4_bam = DenseBlockBAM(flt * 2, flt)
        self.final = nn.Sequential(
            nn.Conv2d(flt, num_channels, kernel_size=1), nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.out6 = outconv(512, 1)
        self.out7 = outconv(256, 1)
        self.out8 = outconv(128, 1)
        self.out9 = outconv(64, 1)

    def forward(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):

        if self.attention == "CBAM":
            inputs = self.conv1x1(self.leaky_relu(inputs))

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        else:
            inputs = self.conv1x1(inputs)

            up1_feat = self.up_cat1(inputs, down4_feat)
            up1_feat = self.up_conv1_bam(up1_feat)
            side6 = UpsampleDeterministic(upscale=8)(up1_feat)

            up2_feat = self.up_cat2(up1_feat, down3_feat)
            up2_feat = self.up_conv2_bam(up2_feat)
            side7 = UpsampleDeterministic(upscale=4)(up2_feat)

            up3_feat = self.up_cat3(up2_feat, down2_feat)
            up3_feat = self.up_conv3_bam(up3_feat)
            side8 = UpsampleDeterministic(upscale=2)(up3_feat)

            up4_feat = self.up_cat4(up3_feat, down1_feat)
            up4_feat = self.up_conv4_bam(up4_feat)

            out6 = self.out6(side6)
            out7 = self.out7(side7)
            out8 = self.out8(side8)
            out9 = self.out9(up4_feat)

            my_list = [out6, out7, out8, out9]
            out10 = torch.mean(torch.stack(my_list), dim=0)

        return out10


class DenseUnetCBAMNoIf(nn.Module):
    def __init__(self, inchannels=1):
        super(DenseUnetCBAMNoIf, self).__init__()
        self.enc = Encoder_Dense_NoIf(inchannels, attention="CBAM")
        self.dec = Decoder_Dense_NoIf(512, attention="CBAM")

    def forward(self, input):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = self.enc(input)
        out = self.dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
        return out


if __name__ == "__main__":
    a = torch.ones(1, 1, 256, 256)
    model = Decoder_Dense_Meta()

    # enc = Encoder_Dense()
    # dec = Decoder_Dense_EMAU()
    # bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat = enc(a)
    # b = dec(bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat)
    print(model.state_dict().items())
