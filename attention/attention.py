# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:00
@Author : Mark
@File   : attention.py
"""
import math
import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query:[30,8,10,64]
    :param key:[30,8,10,64]
    :param value:[30,8,10,64]
    :param mask:[30,1,1,10]:[30,1]中的每一个都是[1,10]的全是1的矩阵
    :param dropout:nn.Dropout(p=0.1)
    :return:
    """
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
