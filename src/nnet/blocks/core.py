import torch.nn as nn
from .utils import activation_func


class ConvLayer(nn.Module):
    """
    Exemple de classe qui définit une couche convolutive
    """
    def __init__(self, input_chan, output_chan, kernel_size=3, dropout=0,
                 norm=None,
                 activation='relu'):
        super(ConvLayer, self).__init__()
        bias = norm is None or norm is 'none'
        output_chan = int(output_chan)
        layers = [nn.Conv2d(in_channels=int(input_chan),
                            out_channels=output_chan,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            stride=1,
                            bias=bias)]

        if norm == 'instance':
            layers.append(nn.InstanceNorm2d(output_chan, True))

        elif norm == 'batch':
            layers.append(nn.BatchNorm2d(output_chan))

        elif norm == 'group':
            layers.append(nn.GroupNorm(output_chan // 16, output_chan))

        layers.append(activation_func(activation))

        if dropout:
            if activation == 'selu':
                layers.append(nn.AlphaDropout(dropout))
            else:
                layers.append(nn.Dropout2d(dropout))

        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)

