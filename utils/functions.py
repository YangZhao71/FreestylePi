"""Ported from Mycroft-precise
"""
from math import exp, log, sqrt, pi
import numpy as np
from typing import *

LOSS_BIAS = 0.9  # [0..1] where 1 is inf bias


def weighted_log_loss(yt, yp) -> Any:
    """
    Binary crossentropy with a bias towards false negatives
    yt: Target
    yp: Prediction
    """
    from keras import backend as K

    pos_loss = -(0 + yt) * K.log(0 + yp + K.epsilon())
    neg_loss = -(1 - yt) * K.log(1 - yp + K.epsilon())

    return LOSS_BIAS * K.mean(neg_loss) + (1. - LOSS_BIAS) * K.mean(pos_loss)


def weighted_mse_loss(yt, yp) -> Any:
    from keras import backend as K

    total = K.sum(K.ones_like(yt))
    neg_loss = total * K.sum(K.square(yp * (1 - yt))) / K.sum(1 - yt)
    pos_loss = total * K.sum(K.square(1. - (yp * yt))) / K.sum(yt)

    return LOSS_BIAS * neg_loss + (1. - LOSS_BIAS) * pos_loss


def false_pos(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast(yp * (1 - yt) > 0.5, 'float')) / K.maximum(1.0, K.sum(1 - yt))


def false_neg(yt, yp) -> Any:
    from keras import backend as K
    return K.sum(K.cast((1 - yp) * (0 + yt) > 0.5, 'float')) / K.maximum(1.0, K.sum(0 + yt))


def sigmoid(x):
    """Sigmoid squashing function for scalars"""
    return 1 / (1 + exp(-x))


def asigmoid(x):
    """Inverse sigmoid (logit) for scalars"""
    return -log(1 / x - 1)


def pdf(x, mu, std):
    """Probability density function (normal distribution)"""
    if std == 0:
        return 0
    return (1.0 / (std * sqrt(2 * pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))


def load_keras() -> Any:
    # Remove warning messages in Keras and TensorFlow
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    import keras
    keras.losses.weighted_log_loss = weighted_log_loss
    keras.metrics.false_pos = false_pos
    keras.metrics.false_positives = false_pos
    keras.metrics.false_neg = false_neg
    return keras
