# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2022/7/5 10:37 AM
# @File: common_types
# @Email: mlshenkai@163.com
from typing import TypeVar, Union, Tuple
from torch._six import container_abcs
from itertools import repeat

T = TypeVar("T")
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
