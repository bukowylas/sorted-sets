"""
The Plodder: the classic linear merge. Walk both sorted arrays together,
one step at a time, advancing whichever pointer points to the smaller
current value - when they match, that's a shared element.

No cleverness, no shortcuts - just steady, honest progress through both
lists exactly once. O(n + m) comparisons, guaranteed.
"""

import numpy as np


def intersect_plodder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Both a and b must be STRICTLY increasing (sorted, no duplicate
    values) 1D arrays of comparable dtype - a repeated value within
    either array will produce a repeated match in the output, unlike
    numpy.intersect1d, which always de-duplicates its result regardless
    of input. This is a declared scope restriction, not a bug: this
    study's arrays are always built without duplicates."""
    i, j = 0, 0
    result = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return np.array(result, dtype=a.dtype)
