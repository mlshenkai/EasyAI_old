# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/7/4 7:40 PM
# @File: activation_module
# @Email: mlshenkai@163.com
"""
基础激活函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(1.2 * x + 3.0, inplace=self.inplace) / 6.0


class GELU(nn.Module):
    def __init__(self, inplace=True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, act_type, inplace=True):
        super(Activation, self).__init__()
        if isinstance(act_type, str):
            act_type = act_type.lower()
            if act_type == "relu":
                self.act = nn.ReLU(inplace=inplace)
            elif act_type == "relu6":
                self.act = nn.ReLU6(inplace=inplace)
            elif act_type == "sigmoid":
                self.act = nn.Sigmoid()
            elif act_type == "hard_sigmoid":
                self.act = HSigmoid(inplace)
            elif act_type == "hard_swish":
                self.act = HSwish(inplace=inplace)
            elif act_type == "leakyrelu":
                self.act = nn.LeakyReLU(inplace=inplace)
            elif act_type == "gelu":
                self.act = GELU(inplace=inplace)
            elif act_type == "swish":
                self.act = Swish(inplace=inplace)
            else:
                raise NotImplementedError
        elif isinstance(act_type, nn.Module):
            self.act = act_type
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.act(inputs)