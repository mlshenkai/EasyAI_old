# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/7/4 7:25 PM
# @File: common_module
# @Email: mlshenkai@163.com
import torch

from EasyAI.models.basics.base_module import BaseModule
from EasyAI.models.basics.activation_module import Activation
from EasyAI.models.basics.normal_module import Normal
from EasyAI.models.basics.common_types import _size_2_t, _pair
import torch.nn as nn


class CNA(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        is_same_pad: bool = True,
        padding_mode: str = "zeros",
        inplace: bool = True,
        act="silu",
        normal="bn2",
    ):
        """

        Args:
             in_channels: int,
             out_channels: int,
             kernel_size: _size_2_t,
             stride: _size_2_t = 1,
             padding: _size_2_t = 0,
             dilation: _size_2_t = 1,
             groups: int = 1,
             bias: bool = True,
             is_same_pad: bool = True,
             padding_mode: str = 'zeros',
             inplace: bool = True,
             act="silu"
             normal="bn2"
        Returns:
            nn.Module
        """
        if is_same_pad:
            padding = (kernel_size - 1) // 2
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(CNA, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size_,
            stride=stride_,
            padding=padding_,
            dilation=dilation_,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            inplace=inplace,
            act=act,
            normal=normal,
        )
        self.act = Activation(act_type=act, inplace=inplace)
        self.normal = Normal(normal, out_channels)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.outputs = out_channels

    def get_output_size(self):
        return self.outputs

    def forward(self, x):
        return self.act(self.normal(self.conv(x)))

    def fuse_forward(self, x):
        return self.act(self.conv(x))


class DWConv(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        act="silu",
    ):
        """
        depthwise+pointwise
        Args:
            in_channels:
            out_channels:
            kernel_size:
            stride:
            act:
        Returns:
            nn.Module
        """
        super(DWConv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            act=act,
        )
        self.d_conv = CNA(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.p_conv = CNA(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act=act,
        )
        self.output_channels = out_channels

    def forward(self, x):
        x = self.d_conv(x)
        return self.p_conv(x)

    def get_output_size(self):
        return self.output_channels


class Bottleneck(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
        act="silu",
    ):
        super(Bottleneck, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            shortcut=shortcut,
            expansion=expansion,
            depthwise=depthwise,
            act=act,
        )
        hidden_channels = int(out_channels * expansion)
        conv = DWConv if depthwise else CNA
        self.conv1 = CNA(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            act=act,
        )
        self.conv2 = conv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            act=act,
        )
        self.use_add = shortcut and in_channels == out_channels
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y



class ResLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 act="leakyrelu"):
        super(ResLayer, self).__init__()
        min_channels = in_channels // 2
        self.layer1 = CNA(in_channels,min_channels, kernel_size=1, stride=1,act=act)
        self.layer2 = CNA(min_channels, min_channels, kernel_size=3,stride=1,act=act)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return out

class SPPBottleneck(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size_list: list,
                 act="silu"):
        """
        Spatial pyramid pooling layer
        Args:
            in_channels:
            out_channels:
            kernel_size_list:
            act:
        """
        super(SPPBottleneck, self).__init__()
        hidden_channel = in_channels // 2
        self.conv1 = CNA(in_channels,out_channels,kernel_size=1,act=act)
        self.module_list = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=kernel_size,stride=1,padding=kernel_size//2)
                for kernel_size in kernel_size_list
            ]
        )
        conv2_channels = hidden_channel * (len(kernel_size_list) + 1)
        self.conv2 = CNA(conv2_channels, out_channels, kernel_size=1, stride=1,act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x]+[m(x) for m in self.module_list],dim=1)
        x = self.conv2(x)
        return x

class CSPLayer(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_block: int = 1,
                 shortcut: bool = True,
                 expansion: float = 0.5,
                 depthwise: bool = False,
                 act="silu"):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = CNA(in_channels, hidden_channels, kernel_size=1,stride=1,act=act)
        self.conv2 = CNA(in_channels, hidden_channels, kernel_size=1, stride=1,act=act)

