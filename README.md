# The Sorted Sets Study: intersecting two sorted arrays

Currently NumPy's `intersect1d` does not assume its input is sorted — it sorts (or hashes) internally regardless, even when the caller already has sorted data. This study searches for the best method for this specific case: given two arrays that are already sorted, is there a faster way to find their intersection than NumPy's general-purpose function? Four algorithmic ideas were built and raced against each other and against `intersect1d`, and one of them — batched binary search of the smaller array into the larger one — wins across nearly the entire tested range, beating `intersect1d` by 3-20x in most cases.

## Background

This study started from a simple question: for two sorted arrays, is the textbook-optimal linear merge actually the fastest approach in practice, or does something else win once real hardware (cache behavior, vectorization, interpreter overhead) enters the picture? It's a search for the best method, undertaken out of curiosity about the algorithmic question itself.

## The contenders

Each candidate got a name during development, kept here because they're a clearer handle than formal algorithm names for telling the story of what happened:

- **The Plodder** — the classic linear merge. Walk both sorted arrays together, one step at a time, advancing whichever pointer is behind.
- **The Hasher** — ignores that the inputs are sorted entirely. Dumps one array into a hash set, checks membership for the other.
- **The Leapfrog** — binary-searches every element of the smaller array directly into the larger one, skipping the linear walk.
- **The Galloper** — like the Leapfrog, but starts each search with small doubling hops from where the last search left off, exploiting the fact that consecutive targets tend to be close together in sorted data.

## Leveling the playing field

The first race pitted all four candidates, run as plain Python loops, against `numpy.intersect1d`. The result: `intersect1d` won almost everywhere, and the Plodder — supposedly a reasonable O(n+m) algorithm — was the slowest contender by a wide margin, taking thousands of microseconds where `intersect1d` took hundreds.

This result was accurate but not yet a fair comparison of algorithms — it compared "a Python `for`/`while` loop, one element at a time" against "a single call into NumPy's compiled C internals," which is a language-implementation difference, not an algorithmic one. Any algorithm run as a Python loop would lose that race regardless of its merits. Leveling the field meant vectorizing what could be vectorized and re-running the comparison.

The Leapfrog and the Hasher both vectorize naturally: the Leapfrog becomes a single batched `np.searchsorted` call over the whole small array at once, and the Hasher becomes a single `np.isin` call. The Plodder and the Galloper have no natural single-call NumPy form — their logic is inherently sequential, since each step depends on exactly where the previous one left off. This is a real finding in itself: an algorithm's fit with array-oriented vectorization can matter as much in practice as its theoretical complexity.

## Results

Winner by array-size pair and overlap density, vectorized candidates only:

| size pair | density=0.0 | density=0.1 | density=0.5 | density=0.9 |
|---|---|---|---|---|
| 1,000 vs 1,000 | Leapfrog | Hasher | Leapfrog | Leapfrog |
| 1,000 vs 10,000 | Leapfrog | Leapfrog | Leapfrog | Leapfrog |
| 100 vs 10,000 | Leapfrog | Leapfrog | Leapfrog | Leapfrog |
| 10 vs 10,000 | Leapfrog | Leapfrog | Leapfrog | Leapfrog |

**The vectorized Leapfrog wins nearly the entire grid**, and not narrowly — it beats `intersect1d` by roughly 3-20x across most of the tested space, with the advantage growing as the size ratio becomes more asymmetric. The asymmetric-size rows are solid: the Leapfrog wins there in every repeated run of this benchmark. Near equal array sizes, the picture is closer: the vectorized Hasher and vectorized Leapfrog typically differ by only 1.2-2x, well within normal run-to-run measurement noise, so the winner in that region can flip between repeated runs of the same benchmark. The table above reflects one run; the equal-size row is a close, occasionally-flipping race rather than a settled boundary.

The core idea — batch every lookup into a single `np.searchsorted` call rather than looping — is a small change with a large effect: for already-sorted input, it outperforms NumPy's general-purpose intersection function.

## Scope

This study covers the uniform-population, single-machine, Python-only comparison described above. It does not include a Rust or Cython translation (unlike the companion Reshuffling Study), and does not map the exact threshold at which the Hasher overtakes the Leapfrog in the near-equal-size region — both are natural next steps, not yet done here.

Every candidate assumes its input arrays are strictly increasing — sorted with no duplicate values. `numpy.intersect1d` always de-duplicates its result regardless of input; none of the candidates here do, so a duplicate value in either input array produces a result that disagrees with `numpy.intersect1d`. This is declared in each candidate's docstring and demonstrated directly in `referee.py` rather than silently left as an untested case, since this study's own benchmark arrays never contain duplicates.

## Repository layout

- `python/` — all four candidates (loop-based and vectorized where applicable), the correctness referee, and the race benchmark.
