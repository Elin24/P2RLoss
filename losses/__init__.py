# -*- coding: utf-8 -*-

from .p2rloss import P2RLoss

def build_loss(config):
    factor = config.FACTOR
    lossfunc = config.LOSS

    Loss = {
        'P2R': P2RLoss
    }[lossfunc]

    return Loss(factor=factor), Loss(factor=factor)
