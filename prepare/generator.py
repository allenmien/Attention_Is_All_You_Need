# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:17
@Author : Mark
@File   : generator.py
"""
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        """
        :param d_model:
        :param vocab:
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        return F.log_softmax(self.proj(x), dim=-1)
