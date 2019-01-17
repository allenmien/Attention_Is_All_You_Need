# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:12
@Author : Mark
@File   : clone.py
"""
import copy
import torch.nn as nn


def clones(module, N):
    """
    :param module:
    :param N:
    :return:
    """
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
