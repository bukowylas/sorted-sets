"""
The race: times all four contenders across a grid of size ratios and
overlap densities, using numpy.intersect1d as an additional reference
point. Correctness is already established by referee.py - this script
assumes all four are correct and focuses purely on speed.

Kept light for modest hardware: small-to-moderate array sizes, few
repeats, a compact grid - meant to finish in well under a minute.
"""

import time

import numpy as np

from the_plodder import intersect_plodder
from the_hasher import intersect_hasher
from the_leapfrog import intersect_leapfrog
from the_galloper import intersect_galloper
from the_hasher_vectorized import intersect_hasher_vectorized
from the_leapfrog_vectorized import intersect_leapfrog_vectorized

CONTENDERS = [
    ("plodder", intersect_plodder),
    ("hasher", intersect_hasher),
    ("leapfrog", intersect_leapfrog),
    ("galloper", intersect_galloper),
    ("hasher_vec", intersect_hasher_vectorized),
    ("leapfrog_vec", intersect_leapfrog_vectorized),
]

# (small_size, big_size) pairs - covers near-equal sizes up through
# heavily asymmetric ones, since that asymmetry is exactly where the
# Leapfrog/Galloper are theoretically supposed to shine.
SIZE_PAIRS = [
    (1_000, 1_000),
    (1_000, 10_000),
    (100, 10_000),
    (10, 10_000),
]

OVERLAP_DENSITIES = [0.0, 0.1, 0.5, 0.9]  # fraction of the smaller array's values also present in the larger one

N_REPEATS = 5


def make_arrays(small_size, big_size, overlap_density, rng):
    """Builds a sorted 'big' array of distinct values, and a sorted
    'small' array where `overlap_density` fraction of its values are
    guaranteed to also appear in 'big', and the rest are guaranteed not
    to."""
    universe = np.arange(big_size * 3)  # generous value range to avoid collisions
    big = np.sort(rng.choice(universe, size=big_size, replace=False))

    n_shared = int(round(small_size * overlap_density))
    n_distinct = small_size - n_shared

    shared_part = rng.choice(big, size=min(n_shared, len(big)), replace=False) if n_shared > 0 else np.array([], dtype=big.dtype)

    non_big_universe = np.setdiff1d(universe, big, assume_unique=False)
    distinct_part = rng.choice(non_big_universe, size=min(n_distinct, len(non_big_universe)), replace=False) if n_distinct > 0 else np.array([], dtype=big.dtype)

    small = np.sort(np.concatenate([shared_part, distinct_part]))
    return small.astype(np.int64), big.astype(np.int64)


def time_call(fn, a, b, repeats=N_REPEATS):
    fn(a, b)  # discard warm-up call
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(a, b)
        timings.append(time.perf_counter() - start)
    return np.median(timings)


def main():
    rng = np.random.default_rng(42)
    results = []

    total_cells = len(SIZE_PAIRS) * len(OVERLAP_DENSITIES)
    done = 0
    start_time = time.perf_counter()

    for small_size, big_size in SIZE_PAIRS:
        for density in OVERLAP_DENSITIES:
            small, big = make_arrays(small_size, big_size, density, rng)

            cell_times = {}
            for name, fn in CONTENDERS:
                cell_times[name] = time_call(fn, small, big)
            cell_times["intersect1d"] = time_call(lambda a, b: np.intersect1d(a, b), small, big)

            winner = min(cell_times, key=cell_times.get)
            results.append({
                "small_size": small_size, "big_size": big_size, "density": density,
                "times": cell_times, "winner": winner,
            })

            done += 1
            elapsed = time.perf_counter() - start_time
            times_str = "  ".join(f"{name}={t*1e6:8.1f}us" for name, t in cell_times.items())
            print(f"[{done}/{total_cells}] small={small_size:>5} big={big_size:>6} density={density:.1f}  "
                  f"{times_str}  -> {winner}  [{elapsed:.1f}s elapsed]", flush=True)

    print(f"\nTotal wall-clock time: {time.perf_counter() - start_time:.1f}s")

    print("\n=== Summary: winner by (size pair, overlap density) ===")
    print(f"{'size pair':>18} | " + " | ".join(f"d={d:.1f}" for d in OVERLAP_DENSITIES))
    print("-" * (20 + 10 * len(OVERLAP_DENSITIES)))
    by_key = {(r["small_size"], r["big_size"], r["density"]): r for r in results}
    for small_size, big_size in SIZE_PAIRS:
        row = []
        for density in OVERLAP_DENSITIES:
            r = by_key[(small_size, big_size, density)]
            row.append(r["winner"])
        print(f"{f'{small_size}v{big_size}':>18} | " + " | ".join(f"{w:>7}" for w in row))

    return results


if __name__ == "__main__":
    main()
