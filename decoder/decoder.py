# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:16
@Author : Mark
@File   : decoder.py
"""
import torch.nn as nn

from utils.clone import clones
from layer_connections.layer_norm import LayerNorm


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        """
        :param layer:
        :param N:
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
