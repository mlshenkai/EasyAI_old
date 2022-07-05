# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/7/5 11:01 AM
# @File: normal_module
# @Email: mlshenkai@163.com
import torch.nn as nn


class Normal(nn.Module):
    def __init__(
        self,
        normal_type,
        num_features=None,
        normalized_shape=None,
        eps: float = 1e-5,
        momentum: float = 0.1,
        elementwise_affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        normal module
        Args:
            normal_type:
            num_features:
            normalized_shape:
            eps: float = 1e-5:
            momentum: float = 0.1,
            elementwise_affine: bool = True
            track_running_stats=True:
        Returns:
            nn.Module
        """
        super(Normal, self).__init__()
        if isinstance(normal_type, str):
            if normal_type == "bn1":
                self.normal = nn.BatchNorm1d(
                    num_features, eps, momentum, elementwise_affine, track_running_stats
                )
            elif normal_type == "bn2":
                self.normal = nn.BatchNorm2d(
                    num_features, eps, momentum, elementwise_affine, track_running_stats
                )
            elif normal_type == "bn3":
                self.normal = nn.BatchNorm3d(
                    num_features, eps, momentum, elementwise_affine, track_running_stats
                )
            elif normal_type == "ln":
                self.normal = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
            else:
                raise NotImplementedError
        elif isinstance(normal_type, nn.Module):
            self.normal = normal_type
        else:
            raise NotImplementedError
