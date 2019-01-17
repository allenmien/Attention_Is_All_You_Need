# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:08
@Author : Mark
@File   : layer_norm.py
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        """
        :param features:512
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        :param x:[30,10,512] 30样本，10个单词，512维的词向量的 representation
        :return:[30,10,512] 每一个样本，每一个单词，在+position_encoding后不再标准化分布，这一步返回一个512维横向标准化之后的 representation
        """
        mean = x.mean(-1, keepdim=True) #[30,10,1] 横向求平均
        std = x.std(-1, keepdim=True) #[30,10,1] 横向求标准差
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
