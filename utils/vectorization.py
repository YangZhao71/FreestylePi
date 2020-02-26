"""Ported from Mycroft-precise
"""
import hashlib
import numpy as np
import os
from typing import *

from utils.audio import load_audio, InvalidAudio
from sonopy import mfcc_spec, mel_spec

inhibit_t = 0.4
inhibit_dist_t = 1.0
inhibit_hop_t = 0.1


def vectorize_raw(audio: np.ndarray) -> np.ndarray:
    """Turns audio into feature vectors, without clipping for length"""
    if len(audio) == 0:
        raise InvalidAudio('Cannot vectorize empty audio!')

    sample_rate = 16000

    window_t = 0.1
    window_samples = int(sample_rate * window_t + 0.5)

    hop_t = 0.05
    hop_samples = int(sample_rate * hop_t + 0.5)

    n_filt = 20
    n_fft = 512
    n_mfcc = 13

    return mfcc_spec(
        audio, sample_rate, (window_samples, hop_samples),
        num_filt=n_filt, fft_size=n_fft, num_coeffs=n_mfcc)


def add_deltas(features: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(features)
    for i in range(1, len(features)):
        deltas[i] = features[i] - features[i - 1]

    return np.concatenate([features, deltas], -1)


def vectorize(audio: np.ndarray) -> np.ndarray:
    """
    Args:
        audio: Audio verified to be of `sample_rate`

    Returns:
        array<float>: Vector representation of audio
    """
    if len(audio) > pr.max_samples:
        audio = audio[-pr.max_samples:]
    features = vectorize_raw(audio)
    if len(features) < pr.n_features:
        features = np.concatenate([
            np.zeros((pr.n_features - len(features), features.shape[1])),
            features
        ])
    if len(features) > pr.n_features:
        features = features[-pr.n_features:]

    return features


def vectorize_delta(audio: np.ndarray) -> np.ndarray:
    return add_deltas(vectorize(audio))


def vectorize_inhibit(audio: np.ndarray) -> np.ndarray:
    """
    Returns an array of inputs generated from the
    wake word audio that shouldn't cause an activation
    """

    def samp(x):
        return int(pr.sample_rate * x)

    inputs = []
    for offset in range(samp(inhibit_t), samp(inhibit_dist_t), samp(inhibit_hop_t)):
        if len(audio) - offset < samp(pr.buffer_t / 2.):
            break
        inputs.append(vectorize(audio[:-offset]))
    return np.array(inputs) if inputs else np.empty((0, pr.n_features, pr.feature_size))
