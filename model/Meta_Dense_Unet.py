import torch
from torch import nn

from model.DenseUnet import DenseBlockCBAM, UpConcat
from model.Shared_Encoder import conv1x1


class Encoder_Dense(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Encoder_Dense, self).__init__()
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


class Decoder_Dense(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Decoder_Dense, self).__init__()
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


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + "." + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + "." + k)]

    k_param_fn(model)
    return model


def get_updated_network(old, new, lr):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad

        else:
            updated_theta[k] = state_dicts[k]

    return new


def get_updated_network_new(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad

        else:
            updated_theta[k] = state_dicts[k]

    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)

    return new


class Meta_Decoder_Dense(nn.Module):
    def __init__(self, inchannels=512, num_channels=1, attention="CBAM"):
        super(Meta_Decoder_Dense, self).__init__()

        self.backbone = Decoder_Dense(
            inchannels=inchannels, num_channels=num_channels, attention=attention
        ).cuda()
        self.updated_net = Decoder_Dense(
            inchannels=inchannels, num_channels=num_channels, attention=attention
        ).cuda()

    def inner_net(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        return self.backbone(inputs, down4_feat, down3_feat, down2_feat, down1_feat)

    def outer_net(
        self, inputs, down4_feat=None, down3_feat=None, down2_feat=None, down1_feat=None
    ):
        return self.updated_net(inputs, down4_feat, down3_feat, down2_feat, down1_feat)


class Meta_Encoder_Dense(nn.Module):
    def __init__(self, num_channels=1, attention="CBAM"):
        super(Meta_Encoder_Dense, self).__init__()

        self.backbone = Encoder_Dense(num_channels, attention).cuda()
        self.updated_net = Encoder_Dense(num_channels, attention).cuda()

    def inner_net(self, inputs):
        return self.backbone(inputs)

    def outer_net(self, inputs):
        return self.updated_net(inputs)
