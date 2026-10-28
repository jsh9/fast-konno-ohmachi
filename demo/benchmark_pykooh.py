"""
Time fast_konno_ohmachi against pykooh on the Tohoku record in this folder.

    pip install fast-konno-ohmachi pykooh scipy
    python demo/benchmark_pykooh.py               # every CPU the process may use
    taskset -c 0 python demo/benchmark_pykooh.py  # one CPU (Linux)

Each case reports the best of several runs (builds included, so memory the
process touches for the first time does not count against the first library
timed), and fast cases keep running for about a second. pykooh's numba kernel is compiled before any timing starts. fast_konno_ohmachi runs with n_cores=None, which uses
every CPU the process is allowed to run on, so `taskset -c 0` makes both
libraries single-threaded (NumPy's BLAS included).

pykooh's simplified=True keeps the same 10**(+-3/b) range as method="approx",
and simplified=False sums over every frequency like method="exact". pykooh's
CachedSmoother stores the exact window as a dense matrix and has no batch
method, so 64 spectra are 64 calls. pykooh returns NaN where the window holds
less than 0.3 of total weight; the difference column skips those outputs and
counts them.
"""

import os
import time
from pathlib import Path

import numpy as np
import pykooh
import scipy.io as sio

import fast_konno_ohmachi as ko

BANDWIDTH = 40.0
N_SPECTRA = 64


def best_time(fn, repeat, min_seconds=1.0, max_runs=500):
    """Best of at least `repeat` runs, continuing until about `min_seconds`
    have passed, so millisecond cases are not decided by a few noisy runs."""
    best = float("inf")
    result = None
    runs = 0
    spent = 0.0
    while runs < repeat or (spent < min_seconds and runs < max_runs):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        best = min(best, elapsed)
        spent += elapsed
        runs += 1
    return best, result


def difference(reference, other):
    """Largest absolute difference over outputs both give, as a fraction of the
    reference's peak, and how many outputs ``other`` left as NaN."""
    reference = np.asarray(reference)
    other = np.asarray(other)
    finite = np.isfinite(other)
    gap = np.max(np.abs(other[finite] - reference[finite])) / np.max(np.abs(reference))
    return float(gap), int(np.count_nonzero(~finite))


def seconds(value):
    return f"{value * 1e3:.1f} ms" if value < 1 else f"{value:.2f} s"


def main():
    data = sio.loadmat(Path(__file__).resolve().parent / "spectrum.mat")
    freq = np.ravel(data["freq"]).astype(np.float64)
    spec = np.ravel(data["spec"]).astype(np.float64)
    decades = np.log10(freq[-1] / freq[0])
    log_out = np.geomspace(freq[0], freq[-1], int(np.rint(100 * decades)) + 1)
    rng = np.random.default_rng(0)
    batch = spec * rng.uniform(0.5, 2.0, (N_SPECTRA, 1))

    # Compile pykooh's numba kernel for both code paths before timing.
    for simplified in (False, True):
        pykooh.smooth(freq[:10], freq[:10], spec[:10], BANDWIDTH, simplified=simplified)

    rows = []

    def compare(label, pk_fn, ko_fn, repeat, check=True):
        pk_time, pk_out = best_time(pk_fn, repeat) if pk_fn else (None, None)
        ko_time, ko_out = best_time(ko_fn, repeat)
        diff = difference(ko_out, pk_out) if (check and pk_fn) else None
        rows.append((label, pk_time, ko_time, diff))

    for name, outputs in (("input grid", freq), (f"{log_out.size} log-spaced", log_out)):
        n_out = f"{outputs.size} outputs" if outputs is freq else f"{name} outputs"
        compare(
            f"`smooth`, exact window, {n_out}",
            lambda o=outputs: pykooh.smooth(o, freq, spec, BANDWIDTH, simplified=False),
            lambda o=outputs: ko.smooth(spec, freq, BANDWIDTH, freq_out=o, method="exact", n_cores=None),
            repeat=3,
        )
        compare(
            f"`smooth`, window cut at 10\\*\\*(±3/b), {n_out}",
            lambda o=outputs: pykooh.smooth(o, freq, spec, BANDWIDTH, simplified=True),
            lambda o=outputs: ko.smooth(spec, freq, BANDWIDTH, freq_out=o, method="approx", n_cores=None),
            repeat=5,
        )

        # Cached weights, exact window on both sides.
        pk_build, smoother = best_time(lambda o=outputs: pykooh.CachedSmoother(freq, o, BANDWIDTH), 2)
        ko_build, weights = best_time(
            lambda o=outputs: ko.PrecomputedWeights(freq, BANDWIDTH, freq_out=o, method="exact"), 2
        )
        rows.append((f"cached exact weights, build, {n_out}", pk_build, ko_build, None))
        compare(
            f"cached exact weights, 1 spectrum, {n_out}",
            lambda s=smoother: s(spec),
            lambda w=weights: w.apply(spec, n_cores=None),
            repeat=5,
        )
        compare(
            f"cached exact weights, {N_SPECTRA} spectra, {n_out}",
            lambda s=smoother: np.vstack([s(row) for row in batch]),
            lambda w=weights: w.apply_many(batch, n_cores=None),
            repeat=3,
        )
        del smoother, weights

        # fast_konno_ohmachi's cut window has no pykooh counterpart for caching.
        ko_build, weights = best_time(lambda o=outputs: ko.PrecomputedWeights(freq, BANDWIDTH, freq_out=o), 3)
        rows.append((f"cached cut weights, build, {n_out}", None, ko_build, None))
        compare(
            f"cached cut weights, {N_SPECTRA} spectra, {n_out}",
            None,
            lambda w=weights: w.apply_many(batch, n_cores=None),
            repeat=3,
        )
        del weights

    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(
        f"{freq.size}-point record, bandwidth {BANDWIDTH:g}, {cpus} CPU(s), "
        f"pykooh {pykooh.__version__}, NumPy {np.__version__}\n"
    )
    print("| Case | pykooh | fast-konno-ohmachi | Speedup | Largest difference |")
    print("|---|---|---|---|---|")
    for label, pk_time, ko_time, diff in rows:
        pk_text = seconds(pk_time) if pk_time is not None else "n/a"
        speedup = f"{pk_time / ko_time:.1f}x" if pk_time is not None else ""
        diff_text = ""
        if diff is not None:
            gap, nan = diff
            diff_text = f"{gap:.1e}" + (f" ({nan} NaN in pykooh)" if nan else "")
        print(f"| {label} | {pk_text} | {seconds(ko_time)} | {speedup} | {diff_text} |")


if __name__ == "__main__":
    main()
