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
        return conv1d_same_padding(
            input, self.weight, self.bias, self.stride, self.dilation, self.groups
        )


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = ((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding // 2,
        dilation=dilation,
        groups=groups,
    )


# %%
class ConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


# %%
class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=11,
        channels=[20, 20],
        kernels=[8, 8],
        strides=[1, 1],
    ):
        super().__init__()

        channels.insert(0, in_channels)
        self.layers = nn.Sequential(
            *[
                ConvBlock(channels[i], channels[i + 1], kernels[i], strides[i])
                for i in range(len(kernels))
            ]
        )
        self.final = nn.Linear(channels[-1], out_channels)

    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


# %%
from typing import cast, Union, List


class InceptionModel(nn.Module):
    """A PyTorch implementation of the InceptionTime model.
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    num_blocks:
        The number of inception blocks to use. One inception block consists
        of 3 convolutional layers, (optionally) a bottleneck and (optionally) a residual
        connector
    in_channels:
        The number of input channels (i.e. input.shape[-1])
    out_channels:
        The number of "hidden channels" to use. Can be a list (for each block) or an
        int, in which case the same value will be applied to each block
    bottleneck_channels:
        The number of channels to use for the bottleneck. Can be list or int. If 0, no
        bottleneck is applied
    kernel_sizes:
        The size of the kernels to use for each inception block. Within each block, each
        of the 3 convolutional layers will have kernel size
        `[kernel_size // (2 ** i) for i in range(3)]`
    num_pred_classes:
        The number of output classes
    """

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: Union[List[int], int],
        bottleneck_channels: Union[List[int], int],
        kernel_sizes: Union[List[int], int],
        use_residuals: Union[List[bool], bool, str] = "default",
        num_pred_classes: int = 1,
    ) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            "num_blocks": num_blocks,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "bottleneck_channels": bottleneck_channels,
            "kernel_sizes": kernel_sizes,
            "use_residuals": use_residuals,
            "num_pred_classes": num_pred_classes,
        }

        channels = [in_channels] + cast(
            List[int], self._expand_to_blocks(out_channels, num_blocks)
        )
        bottleneck_channels = cast(
            List[int], self._expand_to_blocks(bottleneck_channels, num_blocks)
        )
        kernel_sizes = cast(List[int], self._expand_to_blocks(kernel_sizes, num_blocks))
        if use_residuals == "default":
            use_residuals = [True if i % 3 == 2 else False for i in range(num_blocks)]
        use_residuals = cast(
            List[bool],
            self._expand_to_blocks(
                cast(Union[bool, List[bool]], use_residuals), num_blocks
            ),
        )

        self.blocks = nn.Sequential(
            *[
                InceptionBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    residual=use_residuals[i],
                    bottleneck_channels=bottleneck_channels[i],
                    kernel_size=kernel_sizes[i],
                )
                for i in range(num_blocks)
            ]
        )

        # a global average pooling (i.e. mean of the time dimension) is why
        # in_features=channels[-1]
        self.linear = nn.Linear(in_features=channels[-1], out_features=num_pred_classes)

    @staticmethod
    def _expand_to_blocks(
        value: Union[int, bool, List[int], List[bool]], num_blocks: int
    ) -> Union[List[int], List[bool]]:
        if isinstance(value, list):
            assert len(value) == num_blocks, (
                f"Length of inputs lists must be the same as num blocks, "
                f"expected length {num_blocks}, got {len(value)}"
            )
        else:
            value = [value] * num_blocks
        return value

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.blocks(x).mean(dim=-1)  # the mean is the global average pooling
        return self.linear(x)


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool,
        stride: int = 1,
        bottleneck_channels: int = 32,
        kernel_size: int = 41,
    ) -> None:
        assert kernel_size > 3, "Kernel size must be strictly greater than 3"
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(
            *[
                Conv1dSamePadding(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernel_size_s[i],
                    stride=stride,
                    bias=False,
                )
                for i in range(len(kernel_size_s))
            ]
        )

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(
                *[
                    Conv1dSamePadding(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


# %%


class CustomModel(nn.Module):
    def __init__(self, params):

        super(CustomModel, self).__init__()
        self.fcn = FCNBaseline(**params)

    def forward(self, x_dat, x_ref):
        x_input = x_dat
        y = self.fcn(torch.swapaxes(x_input, 2, 1))
        return y, x_dat, x_ref
        # return y, None, None


class CustomModelRef(nn.Module):
    def __init__(self, params):

        super(CustomModelRef, self).__init__()
        self.fcn = FCNBaseline(**params)

    def forward(self, x_dat, x_ref):
        x_input = x_dat
        x_input = torch.cat([x_dat, x_ref.mean(dim=1)], dim=2)
        y = self.fcn(torch.swapaxes(x_input, 2, 1))
        return y, x_dat, x_ref
        # return y, None, None


# %%

import difw


class CPABAverage(nn.Module):
    def __init__(
        self,
        tess_size,
        zero_boundary,
        n_recurrence,
        outsize,
        N,
        in_channels,
        out_channels,
        channels,
        kernels,
        strides,
    ):

        super(CPABAverage, self).__init__()
        self.T = difw.Cpab(tess_size, "pytorch", "cpu", zero_boundary, "svd")
        self.n_recurrence = n_recurrence
        self.outsize = outsize
        self.N = N

        self.localization = FCNBaseline(
            in_channels, out_channels, channels, kernels, strides
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(out_channels, 16), nn.ReLU(True), nn.Linear(16, self.T.params.d)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

    def stn(self, x, z=None):
        xp = x if z is None else torch.cat([x, z], dim=2)
        xs = self.localization(torch.swapaxes(xp, 2, 1))
        theta = self.fc_loc(xs)

        x_align = self.T.transform_data_ss(x, theta, outsize=self.outsize, N=self.N)
        return x_align, theta

    def forward(self, x, z=None):
        thetas = []
        for i in range(self.n_recurrence):
            x, theta = self.stn(x, z)
            thetas.append(theta)
        return x, thetas


# %%


class CustomModelAlign(nn.Module):
    def __init__(
        self,
        params_alignment_ref,
        params_alignment_dat,
        params_classification,
    ):

        super(CustomModelAlign, self).__init__()

        self.align_ref = CPABAverage(**params_alignment_ref)
        self.align_dat = CPABAverage(**params_alignment_dat)
        self.fcn = FCNBaseline(**params_classification)

    def forward(self, x_dat, x_ref):
        # AVERAGE XREF
        x_ref_align, thetas_ref = [], []
        for x_ref_i in x_ref:
            x_ref_i_align, thetas_ref_i = self.align_ref(x_ref_i)
            x_ref_align.append(x_ref_i_align)
            thetas_ref.append(thetas_ref_i)
        x_ref_align = torch.stack(x_ref_align)
        x_ref_avg = torch.mean(x_ref_align, dim=1)

        # ALIGN XDAT
        # x_dat_align, thetas_dat = self.align_dat(x_dat)
        x_dat_align, thetas_dat = self.align_dat(x_dat, x_ref_avg)

        x_input = torch.cat([x_dat_align, x_ref_avg], dim=2)
        y = self.fcn(torch.swapaxes(x_input, 2, 1))
        return y, x_dat_align, x_ref_align


# %%


class CustomModelAlignMLP(nn.Module):
    def __init__(
        self,
        params_alignment_ref,
        params_alignment_dat,
        params_classification,
    ):

        super(CustomModelAlignMLP, self).__init__()

        self.align_ref = CPABAverage(**params_alignment_ref)
        self.align_dat = CPABAverage(**params_alignment_dat)
        self.mlp = MLP(**params_classification)

    def forward(self, x_dat, x_ref):
        # AVERAGE XREF
        x_ref_align, thetas_ref = [], []
        for x_ref_i in x_ref:
            x_ref_i_align, thetas_ref_i = self.align_ref(x_ref_i)
            x_ref_align.append(x_ref_i_align)
            thetas_ref.append(thetas_ref_i)
        x_ref_align = torch.stack(x_ref_align)
        x_ref_avg = torch.mean(x_ref_align, dim=1)

        # ALIGN XDAT
        # x_dat_align, thetas_dat = self.align_dat(x_dat)
        x_dat_align, thetas_dat = self.align_dat(x_dat, x_ref_avg)

        x_input = torch.cat([x_dat_align, x_ref_avg], dim=2)
        y = self.mlp(torch.swapaxes(x_input, 2, 1))
        return y, x_dat_align, x_ref_align


# %%


class BaselineModel(nn.Module):
    def __init__(self, params):

        super(BaselineModel, self).__init__()

        self.mlp = MLP(**params)

    def forward(self, x_dat, x_ref):

        x_input = x_dat
        y = self.mlp(torch.swapaxes(x_input, 2, 1)).mean(dim=1)
        return y, x_dat, x_ref
        # return y, None, None


# %%

class CustomModelNoAlign(nn.Module):
    def __init__(
        self,
        params_classification,
    ):

        super(CustomModelNoAlign, self).__init__()

        self.fcn = FCNBaseline(**params_classification)

    def forward(self, x_dat, x_ref):
        # AVERAGE XREF
        x_ref_avg = torch.mean(x_ref, dim=1)
        x_input = torch.cat([x_dat, x_ref_avg], dim=2)
        y = self.fcn(torch.swapaxes(x_input, 2, 1))
        return y, None, None
        return y, x_dat, x_ref
