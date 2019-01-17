# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:23
@Author : Mark
@File   : embeddings.py
"""
import math
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """
        :param d_model:
        :param vocab:
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
