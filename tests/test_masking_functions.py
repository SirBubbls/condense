"""Tests for selection functions."""

import numpy as np
import condense.optimizer.masking_functions as mf
from logging import info


def test_mask_min_value():
    """Testing default selection function."""

    test_array = np.array([[1, 4, 3], [4, 2, 6]])
    mask = mf.mask_min_value(test_array, 0.5)

    assert (mask != test_array).any(), 'check for mask'
    assert mask.shape == test_array.shape, 'check if shapes changed'
    assert mask.dtype == bool, 'check mask datatype'
    assert (mask != mf.mask_min_value(test_array, 0.2)).any(), 'check if p argument affects mask'

    # check values
    info(f'Mask Values: {mask}')
    assert (mask == np.array([[True, False, True], [False, True, False]])).all(), 'correct mask values'
