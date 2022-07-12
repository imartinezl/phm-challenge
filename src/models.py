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


class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """

    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# %%
class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """

    def __init__(
        self,
        ni,
        no,
        kernel,
        stride,
        pad,
        dropout=None,
        activ=lambda: nn.ReLU(inplace=True),
    ):

        super().__init__()
        assert dropout is None or (0.0 < dropout < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# %%
class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


# %%


class CNN(nn.Module):
    def __init__(self, n_channels, n_features, dropout):

        super(CNN, self).__init__()

        self.n_channels = n_channels
        self.n_features = n_features

        self.net = nn.Sequential(
            SepConv1d(n_channels, 32, 8, 2, 3, dropout=dropout),
            SepConv1d(32, 64, 8, 4, 2, dropout=dropout),
            SepConv1d(64, 128, 8, 4, 2, dropout=dropout),
            SepConv1d(128, 256, 8, 4, 2),
            Flatten(),
        )

        self.fc_input_dim = self.get_conv_to_fc_dim()

    def get_conv_to_fc_dim(self):
        rand_tensor = torch.rand([1, self.n_channels, self.n_features])
        return self.net(rand_tensor).numel()

    def forward(self, x):
        return self.net(x)


# %%


class CustomModel(nn.Module):
    def __init__(self, cnn_params, fcn_params):

        super(CustomModel, self).__init__()

        self.cnn = CNN(**cnn_params)

        fcn_params["input_size"] = self.cnn.fc_input_dim
        self.fcn = MLP(**fcn_params)

    def forward(self, x_dat, x_ref):

        # y_dat = self.fcn(torch.swapaxes(x_dat, 2, 1))

        x_tmp = torch.swapaxes(x_dat, 2, 1)

        y_tmp = self.cnn(x_tmp)

        y_tmp = self.fcn(y_tmp)

        return y_tmp


# %%


class BaselineModel(nn.Module):
    def __init__(self):

        super(BaselineModel, self).__init__()

        self.k = nn.Parameter(torch.Tensor([0]), requires_grad=True)

    def forward(self, x_dat, x_ref):

        return torch.randn(x_dat.shape[0], 11) * self.k


# %%
