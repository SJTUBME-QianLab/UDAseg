import torch
import torch.nn as nn

from util.utils import UpsampleDeterministic, CBAM


class Encode_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Encode_Block, self).__init__()

        self.conv1 = Res_Block(in_feat, out_feat)
        self.conv2 = Res_Block_identity(out_feat, out_feat)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Decode_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Decode_Block, self).__init__()

        self.conv1 = Res_Block(in_feat, out_feat)
        self.conv2 = Res_Block_identity(out_feat, out_feat)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.de_conv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs, down_outputs):
        outputs = self.de_conv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Res_Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Res_Block, self).__init__()
        self.conv_input = conv1x1(inplanes, planes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.in1 = nn.InstanceNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.in3 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.cbam = CBAM(planes)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = self.conv_input(x)

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class Res_Block_identity(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Res_Block_identity, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.in1 = nn.InstanceNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.in3 = nn.InstanceNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.cbam = CBAM(planes)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)

        out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.conv(x)
        return x


class Share_Encoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Share_Encoder, self).__init__()
        flt = 64
        self.down1 = Encode_Block(num_channels, flt)
        self.down2 = Encode_Block(flt, flt * 2)
        self.down3 = Encode_Block(flt * 2, flt * 4)
        self.down4 = Encode_Block(flt * 4, flt * 8)
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
        self.bottom = Encode_Block(flt * 8, flt * 8)

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


class Decoder(nn.Module):
    def __init__(self, inchannels=512, num_channels=1):
        super(Decoder, self).__init__()
        flt = 64
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = Decode_Block(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = Decode_Block(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = Decode_Block(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = Decode_Block(flt * 2, flt)
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


class DecoderMulti(nn.Module):
    def __init__(self, inchannels=512, num_channels=1):
        super(DecoderMulti, self).__init__()
        flt = 64
        self.conv1x1 = conv1x1(inchannels, flt * 16)
        self.up_cat1 = UpConcat(flt * 16, flt * 8)
        self.up_conv1 = Decode_Block(flt * 16, flt * 8)
        self.up_cat2 = UpConcat(flt * 8, flt * 4)
        self.up_conv2 = Decode_Block(flt * 8, flt * 4)
        self.up_cat3 = UpConcat(flt * 4, flt * 2)
        self.up_conv3 = Decode_Block(flt * 4, flt * 2)
        self.up_cat4 = UpConcat(flt * 2, flt)
        self.up_conv4 = Decode_Block(flt * 2, flt)
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


class Shared_Res_unet_cbam_insn_nopool_output_feature_awc(nn.Module):
    def __init__(self):
        super(Shared_Res_unet_cbam_insn_nopool_output_feature_awc, self).__init__()
        self.encoder = Share_Encoder()
        self.decoder = Decoder()

    def forward(self, inputs):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats = self.encoder(
            inputs
        )

        out6, out7, out8, out9, out10 = self.decoder(
            bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
        )

        return out6, out7, out8, out9, out10


if __name__ == "__main__":
    a = torch.ones(1, 1024, 8, 8)
    model = Decoder()
    b = list(model(a))
    for i in range(len(b)):
        print(b[i].size())
