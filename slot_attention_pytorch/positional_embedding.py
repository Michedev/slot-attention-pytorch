import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, width: int, height: int, channels: int) -> None:
        super().__init__()
        east = torch.linspace(0, 1, height).repeat(width)
        west = torch.linspace(1, 0, height).repeat(width)
        south = torch.linspace(0, 1, width).repeat(height)
        north = torch.linspace(1, 0, width).repeat(height)
        east = east.reshape(width, height)
        west = west.reshape(width, height)
        south = south.reshape(height, width).T
        north = north.reshape(height, width).T
        linear_pos_embedding = torch.stack([north, south, west, east], dim=0)  # 4 x w x h
        linear_pos_embedding.unsqueeze_(0)  # for batch size
        self.channels_map = nn.Conv2d(4, channels, kernel_size=1)
        self.register_buffer('linear_position_embedding', linear_pos_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs_linear_position_embedding = self.linear_position_embedding.expand(x.size(0), 4, x.size(2), x.size(3))
        x += self.channels_map(bs_linear_position_embedding)
        return x
