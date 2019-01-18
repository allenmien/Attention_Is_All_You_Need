# -*-coding:utf-8-*-
"""
@Time   : 2019-01-18 09:57
@Author : Mark
@File   : data_generate.py
"""
import numpy as np
import torch
from torch.autograd import Variable

from data.batch import Batch


def data_gen(V, batch, nbatches):
    """
    :param V:
    :param batch:
    :param nbatches:
    :param src=Batch.src: [30,10] -->(0,10)
    :param tgt=Barch.tgt: [30,10] -->(0,10)
    :param
    :return:
    """
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
