"""
The Galloper: exponential (galloping) search. Like the Leapfrog, walks
the smaller array step by step and searches for each value in the
larger one - but instead of a full binary search over the whole
remaining range each time, it first hops forward in doubling strides
(1, 2, 4, 8, ...) from the last found position, until it overshoots the
target, then binary-searches within just that small overshoot bracket.

The idea: since both arrays are sorted and we search left-to-right, the
next target is usually close to where the last one was found - galloping
exploits that locality, touching only a small local window instead of
the whole remaining array, while still guaranteeing worst-case
O(log(gap)) per lookup like a full binary search would.
"""

import numpy as np


def _gallop_search(big: np.ndarray, start: int, value) -> int:
    """Returns the leftmost index in big[start:] where value could be
    inserted to keep big sorted, searching only from `start` onward."""
    n = len(big)
    if start >= n:
        return n

    # Exponential probing: find a bracket [lo, hi) containing the
    # insertion point, doubling the stride each time.
    bound = 1
    lo = start
    while lo + bound < n and big[lo + bound] < value:
        lo = lo + bound
        bound *= 2
    hi = min(lo + bound + 1, n)

    # Binary search within the (small) bracket found above.
    return lo + int(np.searchsorted(big[lo:hi], value))


def intersect_galloper(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Both a and b must be STRICTLY increasing (sorted, no duplicate
    values) 1D arrays of comparable dtype - see the_plodder.py's
    docstring for why this is a declared restriction rather than a bug."""
    if len(a) > len(b):
        big, small = a, b
    else:
        big, small = b, a

    result = []
    pos = 0
    for value in small:
        idx = _gallop_search(big, pos, value)
        if idx < len(big) and big[idx] == value:
            result.append(value)
            pos = idx + 1
        else:
            pos = idx
    return np.array(result, dtype=a.dtype)
