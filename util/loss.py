import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.size_average = size_average

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )

        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )

        if weight is not None:
            # print(loss.size())
            # print(weight.size())
            loss = alpha * loss + beta * loss * weight

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y, weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]):
        bs = x.size(0)
        while x.size()[3] > 1024:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # print(x_vgg[0].size())
        # print(len(x_vgg))
        loss = 0
        for i in range(len(x_vgg)):
            loss += weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGLoss_for_trans(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss_for_trans, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(
        self,
        trans_img,
        struct_img,
        texture_img,
        weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0],
    ):
        while trans_img.size()[3] > 1024:
            trans_img, struct_img, texture_img = (
                self.downsample(trans_img),
                self.downsample(struct_img),
                self.downsample(texture_img),
            )
        trans_vgg, struct_vgg, texture_vgg = (
            self.vgg(trans_img),
            self.vgg(struct_img),
            self.vgg(texture_img),
        )
        loss = 0
        for i in range(len(trans_vgg)):
            if i < 3:
                x_feat_mean = (
                    trans_vgg[i]
                    .view(trans_vgg[i].size(0), trans_vgg[i].size(1), -1)
                    .mean(2)
                )
                y_feat_mean = (
                    texture_vgg[i]
                    .view(texture_vgg[i].size(0), texture_vgg[i].size(1), -1)
                    .mean(2)
                )
                loss += self.criterion(x_feat_mean, y_feat_mean.detach())
            else:
                loss += weights[i] * self.criterion(
                    trans_vgg[i], struct_vgg[i].detach()
                )
        return loss


class VGGLoss_for_trans_simple(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss_for_trans_simple, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.L1Loss()
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(
        self,
        trans_img,
        struct_img,
        texture_img,
        weights=[1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0],
    ):
        while trans_img.size()[3] > 1024:
            trans_img, struct_img, texture_img = (
                self.downsample(trans_img),
                self.downsample(struct_img),
                self.downsample(texture_img),
            )
        trans_vgg, struct_vgg, texture_vgg = (
            self.vgg(trans_img),
            self.vgg(struct_img),
            self.vgg(texture_img),
        )
        loss = 0
        for i in range(len(trans_vgg)):
            if i < 3:
                x_feat_mean = (
                    trans_vgg[i]
                    .view(trans_vgg[i].size(0), trans_vgg[i].size(1), -1)
                    .mean(2)
                )
                y_feat_mean = (
                    texture_vgg[i]
                    .view(texture_vgg[i].size(0), texture_vgg[i].size(1), -1)
                    .mean(2)
                )
                loss += self.criterion(x_feat_mean, y_feat_mean.detach())
            else:
                loss += weights[i] * self.criterion(
                    trans_vgg[i], struct_vgg[i].detach()
                )
        return loss


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(
        log_p, target, ignore_index=255, weight=weight, size_average=False
    )
    if size_average:
        loss /= mask.data.sum()
    return loss


def myL1Loss(source, target):
    return torch.mean(torch.abs(source - target))


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


def discrepancy_slice_wasserstein(p1, p2):
    s = p1.shape
    p1 = p1.view(s[1], s[2] * s[3])
    p2 = p2.view(s[1], s[2] * s[3])
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], 128).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    # print(p1.size())
    # print(p2.size())
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist


if __name__ == "__main__":
    a = torch.zeros(1, 32, 256, 256)
    b = torch.ones(1, 32, 256, 256) * 0.5
    c = torch.ones(1, 3, 256, 256).cuda()
    # VGGLoss = VGGLoss_for_trans()
    print(discrepancy_slice_wasserstein(a, b))
