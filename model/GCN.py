# The based unit of graph convolutional networks.

import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from model.non_local import NLBlockND, NLBlockNDMultiClass


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class ConvTemporalGraphical(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous(), A


class GraphConvolutionLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.get_adjacency = NLBlockND(in_features)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN_Module(Module):
    def __init__(self, in_features, weighted):
        super(GCN_Module, self).__init__()
        self.in_features = in_features
        self.get_adjacency = NLBlockND(in_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.weighted = weighted
        if self.weighted:
            self.weight1 = nn.Parameter(torch.randn(512, 16 * 16))
            self.weight2 = nn.Parameter(torch.randn(512, 16 * 16))

    def forward(self, input):
        n, c, w, h = input.size()
        adj = self.get_adjacency(input)
        input = input.view(n, 1, c, w * h)
        if self.weighted:
            input = self.relu(torch.einsum("nkcv,cw->nkcw", (input, self.weight1)))
            input = self.relu(torch.einsum("nkcv,kvw->nkcw", (input, adj)))
            input = self.relu(torch.einsum("nkcv,cw->nkcw", (input, self.weight2)))
        input = torch.einsum("nkcv,kvw->nkw", (input, adj))
        input = input.view(n, 1, w, h)
        return self.sigmoid(input)


class GraphConvolutionLayersMultiClass(Module):
    def __init__(self, in_features):
        super(GraphConvolutionLayersMultiClass, self).__init__()
        self.in_features = in_features
        self.get_adjacency = NLBlockNDMultiClass(in_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.decrease_featrue = nn.Conv2d(512, in_features, kernel_size=1)

    def forward(self, input):
        input = self.decrease_featrue(input)
        n, c, w, h = input.size()
        adj = self.get_adjacency(input)
        input = input.view(n, 5, int(c / 5), w * h)
        input = self.relu(torch.einsum("nkcv,fvw->nkcw", (input, adj)))
        input = torch.einsum("nkcv,fvw->nkw", (input, adj))
        input = input.view(n, 5, w, h)
        return self.softmax(input)


if __name__ == "__main__":
    b = torch.rand(1, 512, 16, 16)
    model = GraphConvolutionLayersMultiClass(500)
    c = model(b)
    print(c.size())
