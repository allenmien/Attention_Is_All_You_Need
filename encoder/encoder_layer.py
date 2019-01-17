# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:06
@Author : Mark
@File   : encoder_layer.py
"""
import torch.nn as nn

from utils.clone import clones
from layer_connections.sublayer_connection import SublayerConnection


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        :param size: d_model=512
        :param self_attn: MultiHeadedAttention
        :param feed_forward:
        :param dropout:
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        :param x:
        :param mask:
        :return:
        """
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
