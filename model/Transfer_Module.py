import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class Private_Encoder(nn.Module):
    def __init__(self, input_channels, code_size=8):
        super(Private_Encoder, self).__init__()
        self.input_channels = input_channels
        self.code_size = code_size

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        self.model = []
        self.model += [self.cnn]
        self.model += [nn.AdaptiveAvgPool2d((1, 1))]
        self.model += [nn.Conv2d(256, self.code_size, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        bs = x.size(0)
        output = self.model(x).view(bs, -1)
        return output


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm="none",
        activation="relu",
        pad_type="zero",
        bias=True,
    ):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == "zero":
            self.pad = nn.ZeroPad2d(padding)
        # else:
        # assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == "bn":
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == "in":
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == "ln":
            self.norm = LayerNorm(norm_dim)
        elif norm == "adain":
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == "none":
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "none":
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(
            input_dim,
            output_dim,
            kernel_size,
            stride,
            dilation=dilation,
            bias=self.use_bias,
        )

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class ResBlock(nn.Module):
    def __init__(self, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type
            )
        ]
        model += [
            Conv2dBlock(
                dim, dim, 3, 1, 1, norm=norm, activation="none", pad_type=pad_type
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm="in", activation="relu", pad_type="zero"):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Private_Decoder(nn.Module):
    def __init__(self, shared_code_channel, private_code_size=8):
        super(Private_Decoder, self).__init__()
        num_att = 256
        self.shared_code_channel = shared_code_channel
        self.private_code_size = private_code_size

        self.main = []
        self.upsample = nn.Sequential(
            # input: 1/8 * 1/8
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            Conv2dBlock(
                256, 128, 3, 1, 1, norm="ln", activation="relu", pad_type="zero"
            ),
            # 1/4 * 1/4
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            Conv2dBlock(
                128, 64, 3, 1, 1, norm="ln", activation="relu", pad_type="zero"
            ),
            # 1/2 * 1/2
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            Conv2dBlock(64, 32, 3, 1, 1, norm="ln", activation="relu", pad_type="zero"),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(True),
            Conv2dBlock(32, 16, 3, 1, 1, norm="ln", activation="relu", pad_type="zero"),
            # 1 * 1
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh(),
        )

        self.main += [
            Conv2dBlock(
                shared_code_channel + num_att + 1,
                256,
                3,
                stride=1,
                padding=1,
                norm="ln",
                activation="relu",
                pad_type="reflect",
                bias=False,
            )
        ]
        self.main += [ResBlocks(3, 256, "ln", "relu", pad_type="zero")]
        self.main += [self.upsample]

        self.main = nn.Sequential(*self.main)
        self.mlp_att = nn.Sequential(
            nn.Linear(private_code_size, private_code_size),
            nn.ReLU(),
            nn.Linear(private_code_size, private_code_size),
            nn.ReLU(),
            nn.Linear(private_code_size, private_code_size),
            nn.ReLU(),
            nn.Linear(private_code_size, num_att),
        )

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, : m.num_features]
                std = torch.exp(adain_params[:, m.num_features : 2 * m.num_features])
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def forward(self, shared_code, private_code, d):
        d = Variable(torch.FloatTensor(shared_code.shape[0], 1).fill_(d)).cuda()

        d = d.unsqueeze(1)

        d_img = d.view(d.size(0), d.size(1), 1, 1).expand(
            d.size(0), d.size(1), shared_code.size(2), shared_code.size(3)
        )

        att_params = self.mlp_att(private_code)

        att_img = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(
            att_params.size(0),
            att_params.size(1),
            shared_code.size(2),
            shared_code.size(3),
        )

        code = torch.cat([shared_code, att_img, d_img], 1)

        output = self.main(code)
        return output


if __name__ == "__main__":

    encoder = Private_Encoder(64).cuda()
    decoder = Private_Decoder(64).cuda()
    a = torch.ones(1, 64, 16, 16).cuda()
    b = torch.ones(1, 64, 256, 256).cuda()
    c = encoder(b)
    print(c.size())
    b = decoder(a, c, 0)
    print(b.size())
