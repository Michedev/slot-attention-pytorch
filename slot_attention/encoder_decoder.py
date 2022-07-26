from typing import List, Iterable, Tuple, Union, Sequence

from torch import nn

from slot_attention.shared.sequential_cnn import make_sequential_from_config
from slot_attention.positional_embedding import PositionalEmbedding
from slot_attention.nn_utils import calc_output_shape_conv, ActivationFunctionEnum

class Encoder(nn.Module):

    def __init__(self, width: int,
                 height: int,
                 channels: Sequence[int] = (32, 32, 32, 32),
                 kernels: Union[Sequence[int], int] = (5, 5, 5, 5),
                 strides: Union[Sequence[int], int] = (1, 1, 1, 1),
                 paddings: Union[Sequence[int], int] = (2, 2, 2, 2),
                 input_channels: int = 3,
                 batchnorms: Union[bool, Sequence[bool]] = False):
        super().__init__()
        assert len(kernels) == len(strides) == len(paddings) == len(channels)
        self.conv_bone = make_sequential_from_config(input_channels, channels, kernels,
                                                     batchnorms, False, paddings, strides, 'relu',
                                                     try_inplace_activation=False)
        output_channels = channels[-1]
        output_width, output_height = calc_output_shape_conv(width, height, kernels, paddings, strides)
        self.pos_embedding = PositionalEmbedding(output_width, output_height, output_channels)
        self.lnorm = nn.GroupNorm(1, output_channels, affine=True, eps=0.001)
        self.conv_1x1 = [nn.Conv1d(output_channels, output_channels, kernel_size=1), nn.ReLU(inplace=True),
                         nn.Conv1d(output_channels, output_channels, kernel_size=1)]
        self.conv_1x1 = nn.Sequential(*self.conv_1x1)

    def forward(self, x):
        conv_output = self.conv_bone(x)
        conv_output = self.pos_embedding(conv_output)
        conv_output = conv_output.flatten(2, 3)  # bs x c x (w * h)
        conv_output = self.lnorm(conv_output)
        return self.conv_1x1(conv_output)


class Decoder(nn.Module):

    def __init__(self,
                 input_channels: int,
                 width: int,
                 height: int,
                 channels: Sequence[int] = (32, 32, 32, 4),
                 kernels: Sequence[int] = (5, 5, 5, 3),
                 strides: Sequence[int] = (1, 1, 1, 1),
                 paddings: Sequence[int] = (2, 2, 2, 1),
                 output_paddings: Sequence[int] = (0, 0, 0, 0),
                 conv_transposes: Union[bool, Sequence[bool]] = False,
                 activations: Union[Sequence[ActivationFunctionEnum], ActivationFunctionEnum] = 'relu'):
        super().__init__()
        self.conv_bone = []
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        if conv_transposes: assert len(channels) == len(output_paddings)
        self.pos_embedding = PositionalEmbedding(width, height, input_channels)
        self.width = width
        self.height = height

        self.conv_bone = make_sequential_from_config(input_channels, channels, kernels, False, False,
                                                     paddings, strides, activations, output_paddings, conv_transposes)

    def forward(self, slot):
        slot = self.pos_embedding(slot)
        output = self.conv_bone(slot)
        img, mask = output[:, :3], output[:, -1:]
        return img, mask
