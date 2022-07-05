# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/7/4 6:47 PM
# @File: basic_module
# @Email: mlshenkai@163.com
import torch
import torch.nn as nn
from EasyAI.models.basics.activation_module import Activation
from EasyAI.models.basics.normal_module import Normal


class BaseModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()
        self.args = args
        self.kwargs = kwargs
