import functools
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

channel_dim = 1


class Private_Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Private_Encoder, self).__init__()
        self.input_channels = input_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(
                self.input_channels, 64, kernel_size=3, stride=1, padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        self.model = []
        self.model += [self.cnn]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        output = self.model(x)
        return output


class Private_Decoder(nn.Module):
    def __init__(self, input_channels):
        super(Private_Decoder, self).__init__()
        self.input_channels = input_channels

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(
                self.input_channels, 512, kernel_size=2, stride=2
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Tanh(),
        )
        self.model = []
        self.model += [self.cnn]
        self.model = nn.Sequential(*self.model)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        output = self.model(x)
        return output


class Private_Decoder_DAdd(nn.Module):
    def __init__(self, input_channels):
        super(Private_Decoder_DAdd, self).__init__()
        self.input_channels = input_channels

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(
                self.input_channels, 512, kernel_size=2, stride=2
            ),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Tanh(),
        )
        self.model = []
        self.model += [self.cnn]
        self.model = nn.Sequential(*self.model)

    def forward(self, x1, x2):
        x = x1 + x2
        output = self.model(x)
        return output


class Private_Decoder_Add(nn.Module):
    def __init__(self, input_channels):
        super(Private_Decoder_Add, self).__init__()
        self.input_channels = input_channels
        self.conv1x1_1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=1, stride=1, bias=False
        )
        self.conv1x1_2 = nn.Conv2d(
            input_channels, input_channels, kernel_size=1, stride=1, bias=False
        )
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(
                self.input_channels, 256, kernel_size=2, stride=2
            ),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            nn.Tanh(),
        )
        self.model = []
        self.model += [self.cnn]
        self.model = nn.Sequential(*self.model)

    def forward(self, x1, x2):
        x1 = self.conv1x1_1(x1)
        x2 = self.conv1x1_2(x2)
        x = x1 + x2
        output = self.model(x)
        return output


def criterion_gan(y_pred, y_label):
    if y_label == True:
        y_label = 1
    elif y_label == False:
        y_label = 0
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def criterion_entry(y_pred, y_label):
    if y_label == True:
        y_label = 1
    elif y_label == False:
        y_label = 0
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(
        self,
        use_lsgan=None,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan is not None:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


class GANLossWeight(nn.Module):
    def __init__(
        self,
        use_lsgan=None,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=torch.FloatTensor,
    ):
        super(GANLossWeight, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan is not None:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = (self.real_label_var is None) or (
                self.real_label_var.numel() != input.numel()
            )
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = (self.fake_label_var is None) or (
                self.fake_label_var.numel() != input.numel()
            )
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real, weight):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # print(target_tensor.size())
        # print(input.size())
        return self.loss(input * weight.cuda(), target_tensor.cuda() * weight.cuda())


class NLayerDiscriminator_Patch(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_Patch, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        # print(input.shape)
        return self.model(input)


class NLayerDiscriminator32(nn.Module):
    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=3,
        norm_layer=nn.InstanceNorm2d,
        use_sigmoid=False,
    ):
        super(NLayerDiscriminator32, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=3,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=3,
        norm_layer=nn.InstanceNorm2d,
        use_sigmoid=False,
    ):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminatorMask(nn.Module):
    def __init__(
        self,
        input_nc=1,
        ndf=64,
        n_layers=3,
        norm_layer=nn.InstanceNorm2d,
        use_sigmoid=False,
    ):
        super(NLayerDiscriminatorMask, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2

        self.conv_l1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )
        self.conv_l2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.conv_r1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.conv_r2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        # combine two paths
        x = x_l + x_r
        return x


class NetC(nn.Module):
    def __init__(self, ndf=64):
        super(NetC, self).__init__()
        self.ndf = ndf
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, self.ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            GlobalConvBlock(self.ndf, self.ndf * 2, (13, 13)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(self.ndf * 1, self.ndf * 2, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (ndf*2) x 32 x 32
            GlobalConvBlock(self.ndf * 2, self.ndf * 4, (11, 11)),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (ndf*4) x 16 x 16
            GlobalConvBlock(self.ndf * 4, self.ndf * 8, (9, 9)),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            GlobalConvBlock(self.ndf * 8, self.ndf * 16, (7, 7)),
            nn.InstanceNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (ndf*16) x 4 x 4
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(self.ndf * 8, self.ndf * 8, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
        batchsize = input.size()[0]
        out1 = self.convblock1(input)
        # out1 = self.convblock1_1(out1)
        out2 = self.convblock2(out1)
        # out2 = self.convblock2_1(out2)
        out3 = self.convblock3(out2)
        # out3 = self.convblock3_1(out3)
        out4 = self.convblock4(out3)
        # out4 = self.convblock4_1(out4)
        out5 = self.convblock5(out4)
        # out5 = self.convblock5_1(out5)
        out6 = self.convblock6(out5)
        # out6 = self.convblock6_1(out6) + out6
        output = torch.cat(
            (
                input.view(batchsize, -1),
                1 * out1.view(batchsize, -1),
                2 * out2.view(batchsize, -1),
                2 * out3.view(batchsize, -1),
                2 * out4.view(batchsize, -1),
                2 * out5.view(batchsize, -1),
                4 * out6.view(batchsize, -1),
            ),
            1,
        )
        return output


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


class Conv_Block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv_Block, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
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
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = self.conv_input(x)

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn(out)

        out += residual
        out = self.relu(out)

        return out


class Res_Block_identity(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Res_Block_identity, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.conv3 = conv1x1(planes, planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn(out)

        out += residual
        out = self.relu(out)

        return out


class Share_Encoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Share_Encoder, self).__init__()
        flt = 64
        self.down1 = Encode_Block(num_channels, flt)
        self.down2 = Encode_Block(flt, flt * 2)
        self.down3 = Encode_Block(flt * 2, flt * 4)
        self.down4 = Encode_Block(flt * 4, flt * 8)
        self.down_pool = nn.MaxPool2d(kernel_size=2)
        # self.down_pool = nn.AvgPool2d(kernel_size=2)
        self.bottom = Encode_Block(flt * 8, flt * 16)

    def forward(self, inputs):
        down1_feat = self.down1(inputs)
        pool1_feat = self.down_pool(down1_feat)
        down2_feat = self.down2(pool1_feat)
        pool2_feat = self.down_pool(down2_feat)
        down3_feat = self.down3(pool2_feat)
        pool3_feat = self.down_pool(down3_feat)
        down4_feat = self.down4(pool3_feat)
        pool4_feat = self.down_pool(down4_feat)
        bottom_feat = self.bottom(pool4_feat)

        return bottom_feat, down4_feat, down3_feat, down2_feat, down1_feat


class Decoder(nn.Module):
    def __init__(self, num_channels=1):
        super(Decoder, self).__init__()
        flt = 64
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

    def forward(self, inputs, down4_feat, down3_feat, down2_feat, down1_feat):
        up1_feat = self.up_cat1(inputs, down4_feat)
        up1_feat = self.up_conv1(up1_feat)
        up2_feat = self.up_cat2(up1_feat, down3_feat)
        up2_feat = self.up_conv2(up2_feat)
        up3_feat = self.up_cat3(up2_feat, down2_feat)
        up3_feat = self.up_conv3(up3_feat)
        up4_feat = self.up_cat4(up3_feat, down1_feat)
        up4_feat = self.up_conv4(up4_feat)

        outputs = self.final(up4_feat)
        return outputs


class FCDiscriminator(nn.Module):
    def __init__(self, num_classes=1, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x


class Shared_Res_unet(nn.Module):
    def __init__(self, mode):
        super(Shared_Res_unet, self).__init__()
        self.mode = mode
        self.encoder = Share_Encoder()
        self.decoder_1 = Decoder()
        self.decoder_2 = Decoder()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inputs):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats = self.encoder(
            inputs
        )

        if self.mode == 1:
            outputs = self.decoder_1(
                bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            )
        elif self.mode == 2:
            outputs = self.decoder_2(
                bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
            )
        else:
            raise ValueError("Unkown mode!")

        return outputs


class Model(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super(Model, self).__init__()
        self.encoder: torch.nn.Module = encoder
        self.decoder: torch.nn.Module = decoder

    def forward(self, x):
        bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats = self.encoder(x)
        return self.decoder(
            bottom_feat, down4_feat, down3_feat, down2_feat, down1_feats
        )


class GlobalConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(GlobalConvBlock, self).__init__()
        pad0 = (kernel_size[0] - 1) / 2
        pad1 = (kernel_size[1] - 1) / 2

        self.conv_l1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )
        self.conv_l2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.conv_r1 = nn.Conv2d(
            in_dim, out_dim, kernel_size=(1, kernel_size[1]), padding=(0, pad1)
        )
        self.conv_r2 = nn.Conv2d(
            out_dim, out_dim, kernel_size=(kernel_size[0], 1), padding=(pad0, 0)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        # combine two paths
        x = x_l + x_r
        return x


class NetC(nn.Module):
    def __init__(self, ngpu=0):
        super(NetC, self).__init__()
        ndf = 64
        channel_dim = 1
        self.ngpu = ngpu
        self.convblock1 = nn.Sequential(
            # input is (channel_dim) x 128 x 128
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf) x 64 x 64
        )
        self.convblock1_1 = nn.Sequential(
            # state size. (ndf) x 64 x 64
            GlobalConvBlock(ndf, ndf * 2, (13, 13)),
            # nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 64 x 64
        )
        self.convblock2 = nn.Sequential(
            # state size. (ndf * 2) x 64 x 64
            nn.Conv2d(ndf * 1, ndf * 2, 5, 2, 2, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*2) x 32 x 32
        )
        self.convblock2_1 = nn.Sequential(
            # input is (ndf*2) x 32 x 32
            GlobalConvBlock(ndf * 2, ndf * 4, (11, 11)),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 32 x 32
        )
        self.convblock3 = nn.Sequential(
            # state size. (ndf * 4) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*4) x 16 x 16
        )
        self.convblock3_1 = nn.Sequential(
            # input is (ndf*4) x 16 x 16
            GlobalConvBlock(ndf * 4, ndf * 8, (9, 9)),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf * 8) x 16 x 16
        )
        self.convblock4 = nn.Sequential(
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*8) x 8 x 8
        )
        self.convblock4_1 = nn.Sequential(
            # input is (ndf*8) x 8 x 8
            GlobalConvBlock(ndf * 8, ndf * 16, (7, 7)),
            # nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 8 x 8
        )
        self.convblock5 = nn.Sequential(
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*16) x 4 x 4
        )
        self.convblock5_1 = nn.Sequential(
            # input is (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, ndf * 32, 3, 1, 1, bias=False),
            # nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 4 x 4
        )
        self.convblock6 = nn.Sequential(
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.2),
            # state size. (ndf*32) x 2 x 2
        )
        # self._initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, input):
        # if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu is 1:
        batchsize = input.size()[0]
        out1 = self.convblock1(input)
        # out1 = self.convblock1_1(out1)
        out2 = self.convblock2(out1)
        # out2 = self.convblock2_1(out2)
        out3 = self.convblock3(out2)
        # out3 = self.convblock3_1(out3)
        out4 = self.convblock4(out3)
        # out4 = self.convblock4_1(out4)
        out5 = self.convblock5(out4)
        # out5 = self.convblock5_1(out5)
        out6 = self.convblock6(out5)
        # out6 = self.convblock6_1(out6) + out6
        output = torch.cat(
            (
                input.view(batchsize, -1),
                1 * out1.view(batchsize, -1),
                2 * out2.view(batchsize, -1),
                2 * out3.view(batchsize, -1),
                2 * out4.view(batchsize, -1),
                2 * out5.view(batchsize, -1),
                4 * out6.view(batchsize, -1),
            ),
            1,
        )
        # else:
        # print('For now we only support one GPU')

        return output


if __name__ == "__main__":
    # model = Shared_Res_unet(mode=1)
    model = Private_Encoder(64).cuda()
    # dis = NLayerDiscriminator().cuda()
    # model = NetC()
    # a = torch.ones(1, 1, 256, 256).cuda() * 0.7
    b = torch.ones(1, 64, 256, 256).cuda()
    # d = torch.ones(1, 1, 256, 256).cuda() * 0.5
    c = model(b)
    print(c.size())
    # b = decoder(c,c)
    # print(b.size())
    # x_mean = torch.mean(d, dim=1, keepdim=True)
    # print(x_mean.size())
    # print(x_mean)
