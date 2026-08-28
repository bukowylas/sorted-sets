"""
The Leapfrog, vectorized: same idea as the_leapfrog.py (binary-search
each element of the smaller array into the larger one), but as ONE
batched np.searchsorted call over the whole small array at once,
instead of a Python loop calling searchsorted once per element.
"""

import numpy as np


def intersect_leapfrog_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Both a and b must be STRICTLY increasing (sorted, no duplicate
    values) 1D arrays of comparable dtype - see the_plodder.py's
    docstring for why this is a declared restriction rather than a bug."""
    if len(a) > len(b):
        big, small = a, b
    else:
        big, small = b, a

    idx = np.searchsorted(big, small)
    idx_clipped = np.clip(idx, 0, len(big) - 1)
    matches = big[idx_clipped] == small
    return np.sort(small[matches])
