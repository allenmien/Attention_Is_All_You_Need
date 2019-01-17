# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:07
@Author : Mark
@File   : sublayer_connection.py
"""
import torch.nn as nn
from layer_connections.layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """
        :param size:
        :param dropout:
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        :param x:
        :param sublayer:
        :return:
        """
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
