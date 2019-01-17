# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:02
@Author : Mark
@File   : position_wise_feed_forward.py
"""
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model:
        :param d_ff:
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
