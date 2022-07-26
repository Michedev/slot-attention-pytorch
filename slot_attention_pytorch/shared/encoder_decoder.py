from dataclasses import dataclass
from typing import List, Literal, Union, Dict

import torch
from slot_attention_pytorch.nn_utils import get_activation_module
from slot_attention_pytorch.shared.sequential_cnn import make_sequential_from_config
from torch import nn as nn


class EncoderParams(Dict):
    activation: Literal['relu', 'leakyrelu', 'elu']
    channels: List[int]
    batchnorms: List[bool]
    bn_affines: List[bool]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    mlp_hidden_size: int
    mlp_output_size: int


class DecoderParams(Dict):
    activation: Literal['relu', 'leakyrelu', 'elu']
    channels: List[int]
    batchnorms: List[bool]
    bn_affines: List[bool]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]

@dataclass(eq=False)
class EncoderNet(nn.Module):
    width: int
    height: int

    input_channels: int
    activations: Literal['relu', 'leakyrelu', 'elu']
    channels: List[int]
    batchnorms: List[bool]
    bn_affines: List[bool]
    kernels: List[int]
    strides: List[int]
    paddings: List[int]
    mlp_hidden_size: int
    mlp_output_size: int

    def __post_init__(self):
        super().__init__()
        self.convs, params = make_sequential_from_config(self.input_channels, self.channels, self.kernels,
                                                 self.batchnorms, self.bn_affines, self.paddings,
                                                 self.strides, self.activations, return_params=True)
        width = self.width
        height = self.height
        for kernel, stride, padding in zip(params['kernels'], params['strides'], params['paddings']):
            width = (width + 2 * padding - kernel) // stride + 1
            height = (height + 2 * padding - kernel) // stride + 1

        self.mlp = nn.Sequential(
            nn.Linear(self.channels[-1] * width * height, self.mlp_hidden_size),
            get_activation_module(self.activations, try_inplace=True),
            nn.Linear(self.mlp_hidden_size, self.mlp_output_size)
        )

    def forward(self, x):
        x = self.convs(x).flatten(1)
        x = self.mlp(x)
        return x


@dataclass(eq=False)
class BroadcastDecoderNet(nn.Module):
    w_broadcast: int
    h_broadcast: int
    input_channels: int
    activations: List[Union[Literal['relu', 'leakyrelu', 'elu'], None]]
    channels: List[int]
    paddings: Union[List[int], int]
    kernels: List[int]
    batchnorms: List[bool] = False
    bn_affines: List[bool] = False
    strides: List[int] = 1

    def __post_init__(self):
        super().__init__()
        self.parse_w_h()

        self.convs = []
        self.convs = make_sequential_from_config(self.input_channels, self.channels, self.kernels,
                                                 self.batchnorms, self.bn_affines, self.paddings,
                                                 self.strides, self.activations)

        ys = torch.linspace(-1, 1, self.h_broadcast)
        xs = torch.linspace(-1, 1, self.w_broadcast)
        ys, xs = torch.meshgrid(ys, xs)
        coord_map = torch.stack((ys, xs)).unsqueeze(0)
        self.register_buffer('coord_map_const', coord_map)

    def parse_w_h(self):
        if not isinstance(self.w_broadcast, int):
            self.w_broadcast = eval(self.w_broadcast)
            assert isinstance(self.w_broadcast, int)
        if not isinstance(self.h_broadcast, int):
            self.h_broadcast = eval(self.h_broadcast)
            assert isinstance(self.h_broadcast, int)

    def forward(self, z):
        batch_size = z.shape[0]
        z_tiled = z.unsqueeze(-1).unsqueeze(-1).expand(batch_size, z.shape[1], self.h_broadcast, self.w_broadcast)
        coord_map = self.coord_map_const.expand(batch_size, 2, self.h_broadcast, self.w_broadcast)
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.convs(inp)
        return result
