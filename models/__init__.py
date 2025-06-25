# -*- coding: utf-8 -*-

from .vgg16bn import VGG16_BN
def build_model(config):
    model = {
        'vgg16bn': VGG16_BN
    }[config.NAME.lower()]

    return model(config), model(config)
