# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 13:57
@Author : Mark
@File   : multi_headed_attention.py
"""
import torch.nn as nn

from attention.attention_core import attention
from utils.clone import clones


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h:
        :param self.h=8
        :param self.d_k=64
        :param nbatches=
        :param d_model:
        :param dropout:
        """
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query:标准化之后的[30,10,512]
        :param key:标准化之后的[30,10,512]
        :param value:标准化之后的[30,10,512]
        :param mask:[30,1,10]--->全是1的矩阵
        :param nbatches=30
        :return:
        """
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # [30,1,1,10]:[30,1]中的每一个都是[1,10]的全是1的矩阵
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # input:512---> output:64
        query = self.linears[0](query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # [30,8,10,64]:每一个样本是[8,10,64],64是query特征的纬度
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # original method
        # query, key, value = \
        #     [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #      for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
