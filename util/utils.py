import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.autograd import Variable


class dice_loss_multi_class(nn.Module):
    def __init__(self):
        super(dice_loss_multi_class, self).__init__()
        self.loss_lambda = [1, 2, 2, 2, 2]

    def forward(self, logits, gt):
        dice = 0
        eps = 1e-7
        softmaxpred = F.softmax(logits, dim=1)

        for i in range(5):
            inse = torch.sum(softmaxpred[:, i, :, :] * gt[:, i, :, :])
            l = torch.sum(softmaxpred[:, i, :, :] * softmaxpred[:, i, :, :])
            r = torch.sum(gt[:, i, :, :])
            dice += (2.0 * (inse + eps) / (l + r + eps)) * self.loss_lambda[i] / 9.0

        return 1 - 1.0 * dice / 5.0


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        target = torch.argmax(target, 1).long()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(
            predict.size(0), target.size(0)
        )
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(
            predict.size(2), target.size(1)
        )
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(
            predict.size(3), target.size(3)
        )

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average
        )
        return loss


def get_time():
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    return cur_time


def corr_loss(y_t1, y_t2):
    t1_mean = torch.mean(y_t1)
    t2_mean = torch.mean(y_t2)
    y_t1_m = y_t1 - t1_mean
    y_t2_m = y_t2 - t2_mean
    insertion = torch.sum(y_t1_m * y_t2_m)
    insertion_sub = torch.sqrt(torch.sum(y_t1_m * y_t1_m) * torch.sum(y_t2_m * y_t2_m))
    return -(insertion / insertion_sub)


def corr_loss_weight(y_t1, y_t2, weight):
    t1_mean = torch.mean(y_t1)
    t2_mean = torch.mean(y_t2)
    y_t1_m = y_t1 - t1_mean
    y_t2_m = y_t2 - t2_mean
    insertion = torch.sum(y_t1_m * y_t2_m * weight)
    insertion_sub = torch.sqrt(
        torch.sum(y_t1_m * y_t1_m * weight) * torch.sum(y_t2_m * y_t2_m * weight)
    )
    return -(insertion / insertion_sub)


class DICELoss_LV(nn.Module):
    def __init__(self):
        super(DICELoss_LV, self).__init__()

    def forward(self, output, mask):
        intersection = output * mask
        intersection = torch.sum(intersection)

        den1 = output * output
        den1 = torch.sum(den1)

        den2 = mask * mask
        den2 = torch.sum(den2)

        eps = 1e-8
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        dice_eso = dice

        loss = 1 - dice_eso
        return loss


def convertToMultiChannel(data):
    data_copy = data
    data = torch.cat((data, data_copy), 1)
    data = torch.cat((data, data_copy), 1)
    return data


def seed_torch(seed=2019):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def upsample_deterministic(x, upscale):
    """
    x: 4-dim tensor. shape is (batch,channel,h,w)
    output: 4-dim tensor. shape is (batch,channel,self. upscale*h,self. upscale*w)
    """
    return (
        x[:, :, :, None, :, None]
        .expand(-1, -1, -1, upscale, -1, upscale)
        .reshape(x.size(0), x.size(1), x.size(2) * upscale, x.size(3) * upscale)
    )


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        """
        Upsampling in pytorch is not deterministic (at least for the version of 1.0.1)
        see https://github.com/pytorch/pytorch/issues/12207
        """
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        """
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,upscale*h,upscale*w)
        """
        return upsample_deterministic(x, self.upscale)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
    lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
    for opt in opts:
        opt.param_groups[0]["lr"] = lr
        if len(opt.param_groups) > 1:
            opt.param_groups[1]["lr"] = lr * 10


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def save_models(model_dict, prefix="./"):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key + ".pth"))


def load_models(model_dict, prefix="./"):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(os.path.join(prefix, key + ".pth")))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.LeakyReLU()
        # self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttentionNoActivate(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionNoActivate, self).__init__()

        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x


class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


class CBAMNoActivate(nn.Module):
    def __init__(self, in_planes):
        super(CBAMNoActivate, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttentionNoActivate()

    def forward(self, x):
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out


def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.logical_and(pred, label).sum()
    return (2 * float(inter_sum) + 1e-5) / (pred_sum + label_sum + 1e-5)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module("flatten", Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                "gate_c_fc_%d" % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            # self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module(
                "gate_c_relu_%d" % (i + 1), nn.LeakyReLU(inplace=True)
            )
        self.gate_c.add_module(
            "gate_c_fc_final", nn.Linear(gate_channels[-2], gate_channels[-1])
        )

    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d(in_tensor, in_tensor.size(2), stride=in_tensor.size(2))
        return self.gate_c(avg_pool).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)


class SpatialGate(nn.Module):
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            "gate_s_bn_reduce0", nn.InstanceNorm2d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module("gate_s_relu_reduce0", nn.LeakyReLU(inplace=True))
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                "gate_s_bn_di_%d" % i,
                nn.InstanceNorm2d(gate_channel // reduction_ratio),
            )
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.LeakyReLU(inplace=True))
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_tensor):
        att = 1 + self.sigmoid(
            self.channel_att(in_tensor) * self.spatial_att(in_tensor)
        )
        return att * in_tensor


class csSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse + U_sse


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=1, bias=False
        )
        self.Conv_Excitation = nn.Conv2d(
            in_channels // 2, in_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2]
        z = self.relu(z)
        z = self.Conv_Excitation(z)  # shape: [bs, c]
        z = self.sigmoid(z)
        return U * z.expand_as(U)


class CBAMParallel(nn.Module):
    def __init__(self, in_planes):
        super(CBAMParallel, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, U):
        U_sse = self.sa(U)
        U_cse = self.ca(U)
        return U_cse + U_sse


def save_arg(args, name):
    arg_dict = vars(args)
    with open("logs/{}".format(name), "a") as f:
        yaml.dump(arg_dict, f)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate_D(optimizer, learning_rate, i_iter, max_iter, power=0.9):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]["lr"] = lr


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(
        1, 1, pred1.size(2), pred1.size(3)
    ) / (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(
        1, 1, pred1.size(2), pred1.size(3)
    )
    return output


def weightmap_copy(pred1, pred2):
    sum_up = torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3))
    sum_down = (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(
        1, 1, pred1.size(2), pred1.size(3)
    )
    # print(sum_up)
    # print(sum_down)
    print(pred1 * pred2)
    print(torch.sum((pred1 * pred2), 1))
    print(torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1))
    output = 1.0 - sum_up / sum_down
    return output


def HoMM3_loss(xs, xt):
    # xs = xs - torch.mean(xs, dim=0)
    # xt = xt - torch.mean(xt, dim=0)

    xs = xs.unsqueeze(-1)
    xs = xs.unsqueeze(-1)
    xt = xt.unsqueeze(-1)
    xt = xt.unsqueeze(-1)

    xs_1 = xs.permute(0, 2, 1, 3)
    xs_2 = xs.permute(0, 2, 3, 1)
    xt_1 = xt.permute(0, 2, 1, 3)
    xt_2 = xt.permute(0, 2, 3, 1)

    HR_Xs = xs * xs_1 * xs_2  # dim: b*L*L*L
    HR_Xs = torch.mean(HR_Xs, dim=0)  # dim: L*L*L
    HR_Xt = xt * xt_1 * xt_2
    HR_Xt = torch.mean(HR_Xt, dim=0)

    return torch.mean((HR_Xs - HR_Xt) * (HR_Xs - HR_Xt))


def HoMM3_loss_weight(xs, xt, weight):
    # xs = xs - torch.mean(xs, dim=0)
    # xt = xt - torch.mean(xt, dim=0)

    xs = xs.unsqueeze(-1)
    xs = xs.unsqueeze(-1)
    xt = xt.unsqueeze(-1)
    xt = xt.unsqueeze(-1)

    xs_1 = xs.permute(0, 2, 1, 3)
    xs_2 = xs.permute(0, 2, 3, 1)
    xt_1 = xt.permute(0, 2, 1, 3)
    xt_2 = xt.permute(0, 2, 3, 1)

    HR_Xs = xs * xs_1 * xs_2 * weight  # dim: b*L*L*L
    HR_Xs = torch.mean(HR_Xs, dim=0)  # dim: L*L*L
    HR_Xt = xt * xt_1 * xt_2 * weight
    HR_Xt = torch.mean(HR_Xt, dim=0)

    return torch.mean((HR_Xs - HR_Xt) * (HR_Xs - HR_Xt))


def KHoMM(xs, xt, order=3, num=30000):
    # xs = xs - torch.mean(xs, dim=0)
    # xt = xt - torch.mean(xt, dim=0)

    dim = xs.size()[1]
    index = torch.Tensor(num, dim).uniform_(0, dim - 1)
    index = index[:, :order].long()  # [300,3]

    xs = xs.transpose(0, 1)
    xs = xs[index]  ##dim=[num,order,batchsize]
    xt = xt.transpose(0, 1)
    xt = xt[index]

    Ho_Xs = torch.prod(xs, dim=1).transpose(0, 1)
    Ho_Xt = torch.prod(xt, dim=1).transpose(0, 1)

    KHoMM = KernelHoMM(Ho_Xs, Ho_Xt, sigma=0.00001)
    return KHoMM


def Cal_pairwise_dist(X, Y):
    # print(X.size())
    norm = lambda x: torch.sum(x * x, 1)
    # print(X.unsqueeze(2).size())
    dist = (norm(X.unsqueeze(2) - Y.transpose(0, 1))).transpose(0, 1)
    # print(dist.size())
    return dist


def KernelHoMM(Ho_Xs, Ho_Xt, sigma):
    dist_ss = Cal_pairwise_dist(Ho_Xs, Ho_Xs)
    dist_tt = Cal_pairwise_dist(Ho_Xt, Ho_Xt)
    dist_st = Cal_pairwise_dist(Ho_Xs, Ho_Xt)

    loss = (
        torch.mean(torch.exp(-sigma * dist_ss))
        + torch.mean(torch.exp(-sigma * dist_tt))
        - 2 * torch.mean(torch.exp(-sigma * dist_st))
    )

    return loss if loss > 0 else 0
