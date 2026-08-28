"""
The Hasher, vectorized: same idea as the_hasher.py (ignore sortedness,
just check membership), but via np.isin's single vectorized call
instead of a Python-level set-and-list-comprehension.
"""

import numpy as np


def intersect_hasher_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Both a and b must have no duplicate values - see the_hasher.py's
    docstring for why this is a declared restriction rather than a bug."""
    mask = np.isin(b, a)
    return np.sort(b[mask])
