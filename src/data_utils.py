"""Data Utility Functions."""
# pylint: disable=invalid-name
import os
import pickle as pickle

import numpy as np

def scoring_function(x, lin_exp_boundary, doubling_rate):
    """Computes score function values.

        The scoring functions starts linear and evolves into an exponential
        increase.
    """
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x < lin_exp_boundary)
    exp_region = np.logical_and(x >= lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score
