# -*-coding:utf-8-*-
"""
@Time   : 2019-01-17 14:19
@Author : Mark
@File   : batch.py
"""
from evl.subsequent_mask import subsequent_mask
from torch.autograd import Variable


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        """
        :param src:[30,10]--->random in (0,10)
        :param self.src_mask :[30,1,10]--->全是1的矩阵
        :param trg:[30,9]--->src[30,(0,9)]
        :param self.trg_y:[30,9]--->src[30,(1,10)]
        :param self.trg_mask:[30,9,9]：30中的每一个矩阵为对角线分开，下面为1，上面为0
        :param self.ntokens:270
        :param pad:0
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum() # 30*9*1=270 ([30,9]全是1)

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        :param tgt:
        :param pad:
        :return:
        """
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2) #[30,1,9]---> all 1
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask #[30,9,9]：30中的每一个矩阵为对角线分开，下面为1，上面为0
        # Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)): [1,9,9]
        # 对角线分开，下面为1，上面为0
