from typing import List, Union, Literal

import torch

ActivationFunctionEnum = Union[Literal['relu', 'leakyrelu', 'elu', 'glu'], None]

@torch.no_grad()
def init_trunc_normal(model, mean=0, std: float = 1):
    for name, tensor in model.named_parameters():
        if 'bias' in name:
            tensor.zero_()
        elif 'weight' in name:
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)

@torch.no_grad()
def init_xavier_(model):
    # print("Xavier init")
    for name, tensor in model.named_parameters():
        if name.endswith('.bias'):
            tensor.zero_()
        elif len(tensor.shape) == 1:
            pass  # silent
            # print(f"    skipped tensor '{name}' because shape="
            #       f"{tuple(tensor.shape)} and it does not end with '.bias'")
        else:
            torch.nn.init.xavier_uniform_(tensor)


def norm_gradient(model, p_norm):
    return sum(param.grad.norm(p_norm) for param in model.parameters())


def get_activation_module(act_f_name, try_inplace=True):
    if act_f_name == 'leakyrelu':
        ActF = torch.nn.LeakyReLU()
    elif act_f_name == 'elu':
        ActF = torch.nn.ELU()
    elif act_f_name == 'relu':
        ActF = torch.nn.ReLU(inplace=try_inplace)
    elif act_f_name == 'glu':
        ActF = torch.nn.GLU(dim=1)  # channel dimension in images
    elif act_f_name == 'sigmoid':
        ActF = torch.nn.Sigmoid()
    elif act_f_name == 'tanh':
        ActF = torch.nn.Tanh()
    else:
        raise ValueError(f"act_f_name = {act_f_name} not found")
    return ActF


def calc_output_shape_conv(width: int, height: int, kernels: List[int], paddings: List[int], strides: List[int]):
    for kernel, stride, padding in zip(kernels, strides, paddings):
        width = (width + 2 * padding - kernel) // stride + 1
        height = (height + 2 * padding - kernel) // stride + 1
    return width, height
