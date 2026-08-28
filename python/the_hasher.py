"""
The Hasher: ignores that both arrays are sorted entirely. Dumps one array
into a hash set, then checks membership for every element of the other.
No finesse - just raw hashing power thrown at the problem, on the
principle that a good hash table doesn't care what order you found your
data in.
"""

import numpy as np


def intersect_hasher(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Order-agnostic - correct even if a and b weren't sorted at all,
    though in this study they always will be, for a fair fight. Both a
    and b must have no duplicate values: a repeated value in b that is
    also present in a will appear once per repetition in the output,
    unlike numpy.intersect1d, which always de-duplicates its result.
    This is a declared scope restriction, not a bug."""
    a_set = set(a.tolist())
    result = [x for x in b.tolist() if x in a_set]
    return np.array(sorted(result), dtype=a.dtype)
