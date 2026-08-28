"""
The referee: checks every contender's intersection result against
numpy.intersect1d (ground truth) before any race is allowed to count.
No speed number means anything if the contender is quietly wrong.
"""

import numpy as np

from the_plodder import intersect_plodder
from the_hasher import intersect_hasher
from the_leapfrog import intersect_leapfrog
from the_galloper import intersect_galloper
from the_hasher_vectorized import intersect_hasher_vectorized
from the_leapfrog_vectorized import intersect_leapfrog_vectorized

CONTENDERS = [
    ("the plodder", intersect_plodder),
    ("the hasher", intersect_hasher),
    ("the leapfrog", intersect_leapfrog),
    ("the galloper", intersect_galloper),
    ("the hasher (vectorized)", intersect_hasher_vectorized),
    ("the leapfrog (vectorized)", intersect_leapfrog_vectorized),
]

FAILURES = []


def check(condition, message):
    if not condition:
        FAILURES.append(message)
        print(f"  FAIL: {message}")
    else:
        print(f"  ok: {message}")


def check_case(name, a, b):
    expected = np.intersect1d(a, b)
    for contender_name, fn in CONTENDERS:
        got = fn(a, b)
        check(np.array_equal(got, expected),
              f"{contender_name} on '{name}': got {got.tolist()}, expected {expected.tolist()}")


def main():
    rng = np.random.default_rng(1)

    check_case("both empty", np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    check_case("one empty", np.array([], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64))
    check_case("no overlap", np.array([1, 3, 5], dtype=np.int64), np.array([2, 4, 6], dtype=np.int64))
    check_case("full overlap", np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64))
    check_case("partial overlap", np.array([1, 2, 3, 4, 5], dtype=np.int64), np.array([3, 4, 5, 6, 7], dtype=np.int64))
    check_case("single element each, match", np.array([5], dtype=np.int64), np.array([5], dtype=np.int64))
    check_case("single element each, no match", np.array([5], dtype=np.int64), np.array([9], dtype=np.int64))
    check_case("very asymmetric sizes", np.array([500], dtype=np.int64), np.sort(rng.integers(0, 10_000, 5000)))
    check_case("small array values all past end of big array",
               np.array([9990, 9991, 9992], dtype=np.int64), np.arange(5000, dtype=np.int64))
    check_case("small array values all before start of big array",
               np.array([1, 2, 3], dtype=np.int64), np.arange(5000, 10000, dtype=np.int64))
    check_case("small array matches scattered through big array in increasing order",
               np.array([10, 500, 501, 4999], dtype=np.int64), np.arange(5000, dtype=np.int64))

    for _ in range(20):
        n_a = rng.integers(1, 200)
        n_b = rng.integers(1, 200)
        a = np.sort(np.unique(rng.integers(0, 500, n_a)))
        b = np.sort(np.unique(rng.integers(0, 500, n_b)))
        check_case(f"random (len_a={len(a)}, len_b={len(b)})", a, b)

    print("\n=== Documented restriction: duplicate values are OUT OF SCOPE ===")
    print("(a review pass found every contender disagrees with numpy.intersect1d")
    print(" once duplicates are present - declared as a scope restriction in each")
    print(" contender's docstring, not fixed, since this study's own arrays never")
    print(" contain duplicates. This check documents the restriction rather than")
    print(" hiding it - it is EXPECTED to fail, and failure here is not a FAILURES bug.)")
    a_dup = np.array([1, 2, 2, 2, 3], dtype=np.int64)
    b_dup = np.array([2, 2, 3, 3, 4], dtype=np.int64)
    expected_dup = np.intersect1d(a_dup, b_dup)
    for name, fn in CONTENDERS:
        got = fn(a_dup, b_dup)
        matches = np.array_equal(got, expected_dup)
        print(f"  {name} on duplicate-containing input: got {got.tolist()}, "
              f"numpy.intersect1d gives {expected_dup.tolist()} "
              f"({'matches (unexpected!)' if matches else 'differs, as documented'})")

    print(f"\n{'='*60}")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S) - do not trust any race results yet.")
        for f in FAILURES:
            print(f"  - {f}")
    else:
        print("All contenders agree with numpy.intersect1d. Cleared to race.")


if __name__ == "__main__":
    main()
