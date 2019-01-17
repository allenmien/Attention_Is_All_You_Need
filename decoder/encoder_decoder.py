# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:05
@Author : Mark
@File   : encoder_decoder.py
"""
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        :param encoder: Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        :param decoder:
        :param src_embed: nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        :param tgt_embed:
        :param generator:
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param src:
        :param tgt:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        self.src_embed(src) -->Embeddings-->PositionalEncoding-->(Encoder-->EncoderLayer
        -->SublayerConnection-->LayerNorm-->MultiHeadedAttention-->attention-->PositionwiseFeedForward)*6
        -->
        :param src: src = batch.src = [30,10]-->元素是：0-10之间的矩阵
                    self.src_embed(src) = layer
        :param self.src_embed： nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        :param ：d_model=512, src_vocab=11, position = PositionalEncoding(d_model, dropout),dropout=0.1
        :param Embeddings(d_model, src_vocab):[30,10,512] 30样本，10个单词，512维的词向量的 representation
        :param position:上面的[30,10,512]会成为positional_encoding的输入x
        :param self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))会得到一个
                [30,10,512] embedding+positonal_encoding的整个encoder的输入矩阵

        :param self.src_embed(src), src_mask会作为EncoderLayer forward的输入
        :param ：src_mask [30,1,10]--->全是1的矩阵
        :param : EncoderLayer(d_model, c(attn), c(ff), dropout) = self.src_embed(src)
        :param : 这里的意思是 self.src_embed(src) 作为输入，
        :return:
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        :param memory:[50,10,512] encoder最后一层的输出
        :param src_mask:[30,1,10]--->全是1的矩阵
        :param tgt:[30,9]--->src[30,(0,9)] :这里的意思是一个source[0,10]的序列，target[0,9]和source是想同的，预测target[11]?=source[11]
        :param tgt_mask:[30,9,9]:30中的每一个矩阵为对角线分开，下面为1，上面为0
        :param self.tgt_embed(tgt) [30,9,512] 目标词汇的word representation
        :return:
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
