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
        :param d_model:512
        :param vocab:11
        :param self.lut : 11*512的词向量矩阵
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x: batch.src = [30,10]-->元素是：0-10之间的矩阵
        :param  : self.lut(x)= [30,10,512] 30样本，10个单词，512维的词向量的 representation
        :return: self.lut(x)*22 = [30,10,512] 30样本，10个单词，512维的词向量的 representation
        """
        return self.lut(x) * math.sqrt(self.d_model)
