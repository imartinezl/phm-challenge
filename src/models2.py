# %%

import torch
import torch.nn as nn

# %%


class MLP(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        hidden_layers=3,
        dropout=0.0,
        output_size=None,
    ):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(hidden_layers):
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# %%
import torch.nn.functional as F

class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels=3, out_classes=11, channels=[20,20], kernels=[8,8], strides=[1,1]):
        super().__init__()

        channels.insert(0, in_channels)
        self.layers = nn.Sequential(*[ConvBlock(channels[i], channels[i+1], kernels[i], strides[i]) for i in range(len(kernels))])
        self.final = nn.Linear(channels[-1], out_classes)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return self.final(x.mean(dim=-1))
        
# %%

class CustomModel(nn.Module):
    def __init__(self, params):

        super(CustomModel, self).__init__()
        self.model = FCNBaseline(**params)


    def forward(self, x_dat, x_ref):

        # y_dat = self.fcn(torch.swapaxes(x_dat, 2, 1))

        x_tmp = torch.swapaxes(x_dat, 2, 1)

        return self.model(x_tmp)


# %%

