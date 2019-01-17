# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:22
@Author : Mark
@File   : subsequent_mask.py
"""
import numpy as np
import torch


def subsequent_mask(size):
    """
    :param size:
    :return:
    """
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
