"""
The Leapfrog: for asymmetric sizes, walk the SMALLER array step by step,
and for each of its elements, binary-search directly into the LARGER
array rather than walking it. Skips over long stretches of the big
array in one jump per lookup, at the cost of each jump being a
non-sequential access (a cache-unfriendly leap rather than a
cache-friendly walk).

Picks the smaller array as the "walker" automatically, since the whole
point is to minimize the number of expensive lookups into the big one.
"""

import numpy as np


def intersect_leapfrog(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Both a and b must be STRICTLY increasing (sorted, no duplicate
    values) 1D arrays of comparable dtype - see the_plodder.py's
    docstring for why this is a declared restriction rather than a bug."""
    if len(a) > len(b):
        big, small = a, b
    else:
        big, small = b, a

    result = []
    for value in small:
        idx = np.searchsorted(big, value)
        if idx < len(big) and big[idx] == value:
            result.append(value)
    return np.array(result, dtype=a.dtype)
