# _Fast_ Konno-Ohmachi

Konno-Ohmachi spectral smoothing for Python. The smoother is written in Rust and published as wheels for Linux, macOS, and Windows.

## Background

Konno-Ohmachi smoothing, from Konno & Ohmachi (1998) [[abstract](http://bssa.geoscienceworld.org/content/88/1/228.short), [PDF](http://www.eq.db.shibaura-it.ac.jp/papers/Konno&Ohmachi1998.pdf)], applies a window whose width is constant in log-frequency:

```text
w = (sin(b log10(f / fc)) / (b log10(f / fc)))^4
```

Low frequencies get a narrower window than high frequencies. Seismologists use that so the low-frequency part of a spectrum (below about 10 Hz) is not smeared out.

The plot below compares that window with a median filter on the Fourier amplitude spectrum of a ground acceleration recorded during the [2011 Tohoku-Oki earthquake](https://en.wikipedia.org/wiki/2011_T%C5%8Dhoku_earthquake_and_tsunami). Above about 5 Hz the two are similar. At lower frequencies the median filter flattens the spectrum.

![](demo.png)

## Installation

```bash
pip install fast-konno-ohmachi
```

Wheels are built for CPython 3.10 and newer (abi3), on Linux x86_64 and arm64, macOS arm64 and x86_64, and Windows x64 and arm64. Installing from a source checkout needs a Rust toolchain.

## Usage

```python
import numpy as np
import fast_konno_ohmachi as ko

freq = np.linspace(0.1, 20.0, 2000)
signal = np.sin(freq)

smoothed = ko.smooth(signal, freq, bandwidth=40)
```

`bandwidth` is the Konno-Ohmachi `b`. Lower values smooth more. The default is 40, and any positive finite number is allowed.

`freq` above is `freq_in`: the frequency of each input sample, in hertz. It has to be finite, positive, and strictly increasing. `freq_out` chooses where the smooth is written and defaults to `freq_in`. A shorter `freq_out` does less work.

```python
coarse = np.linspace(freq[0], freq[-1], 400)
smoothed = ko.smooth(signal, freq, bandwidth=40, freq_out=coarse)
```

`method="approx"` (the default) keeps frequencies with `|b log10(f / fc)| <= 3` and evaluates the window with a polynomial. If an output frequency has no input sample in that range, that one output uses the full exact window. `method="exact"` sums the published window over every positive frequency.

`n_cores` is how many cores to use. The default for `smooth` is 1. `None` uses every logical CPU available to the process, which under a container CPU quota can be fewer than the machine has, and a larger value raises `ValueError`. The workers are ordinary threads; they are not pinned to cores.

```python
smoothed = ko.smooth(signal, freq, bandwidth=40, n_cores=None)
```

When many spectra share a frequency grid, build the weights once and reuse them:

```python
weights = ko.PrecomputedWeights(freq, bandwidth=40)
first = weights.apply(signal, n_cores=1)

batch = np.vstack([signal, signal * 0.5])
many = weights.apply_many(batch)  # n_cores defaults to all logical CPUs
```

`PrecomputedWeights` is not callable. `dtype="float32"` (or `np.float32`) stores the cached weights in float32; the arrays returned by `apply` and `apply_many` stay float64. The attributes `freq_in`, `freq_out`, `bandwidth`, `method` and `dtype` read back the constructor arguments, and the object can be pickled or copied, which rebuilds the weights from them.

With `method="exact"`, `PrecomputedWeights` stores one weight per (input, output) pair: `len(freq_in) * len(freq_out) * 8` bytes, or 4 with `dtype="float32"`. When `freq_out` is `freq_in` (the default) the window is symmetric, so each pair is stored once, `n * (n - 1) / 2` weights for `n` frequencies: 0.9 GB for the 15000-point spectrum in `demo/`. Weights that do not fit raise `MemoryError`, and more than 2**32 - 1 of them raise `ValueError`. For a one-off exact smooth use `smooth(method="exact")`, which stores no weights.

On a log-uniform grid, with `method="approx"` and `freq_out` left as `freq_in`, both paths use a short convolution instead of a weight per output frequency.

`demo/Demo_konno_ohmachi_smooth.py` smooths the spectrum plotted above. Running it needs `scipy` and `matplotlib`.

## Benchmark against pykooh

[pykooh](https://github.com/arkottke/pykooh) 0.5.1 is another Konno-Ohmachi implementation, compiled with numba. The tables below time both libraries on the 15000-point Tohoku spectrum in `demo/` (a linear grid from 0.0033 to 50 Hz) with bandwidth 40. pykooh's `simplified=True` keeps the same `10**(±3/b)` range as `method="approx"` (the "cut window" rows), `simplified=False` sums over every frequency like `method="exact"`, and its `CachedSmoother` plays the role of `PrecomputedWeights(method="exact")`. pykooh has no cached form of the cut window.

Each time is the best of several runs (fast cases repeat for about a second), on a 4-vCPU x86_64 Linux VM with AVX-512, Python 3.13 and NumPy 2.5.4. The 1-CPU columns pin the whole process to one CPU with `taskset -c 0`, which also limits NumPy's BLAS; they are the per-core comparison. `fast-konno-ohmachi` runs with `n_cores=None` throughout.

Output at the 15000 input frequencies:

| Case | pykooh, 1 CPU | fast-konno-ohmachi, 1 CPU | pykooh, 4 CPUs | fast-konno-ohmachi, 4 CPUs |
|---|---|---|---|---|
| `smooth`, exact window | 7.61 s | 3.49 s | 7.08 s | 0.82 s |
| `smooth`, cut window | 1.29 s | 243 ms | 1.26 s | 63 ms |
| Exact weights: build | 37.9 s | 2.16 s | 36.3 s | 2.03 s |
| Exact weights: apply to 1 spectrum | 154 ms | 109 ms | 40 ms | 34 ms |
| Exact weights: apply to 64 spectra | 9.92 s | 3.96 s | 2.99 s | 1.16 s |
| Cut weights: build | n/a | 344 ms | n/a | 364 ms |
| Cut weights: apply to 64 spectra | n/a | 579 ms | n/a | 141 ms |

Output at 419 log-spaced frequencies, 100 per decade:

| Case | pykooh, 1 CPU | fast-konno-ohmachi, 1 CPU | pykooh, 4 CPUs | fast-konno-ohmachi, 4 CPUs |
|---|---|---|---|---|
| `smooth`, exact window | 201 ms | 84 ms | 201 ms | 27 ms |
| `smooth`, cut window | 13.4 ms | 5.2 ms | 13.5 ms | 4.0 ms |
| Exact weights: build | 928 ms | 99 ms | 971 ms | 96 ms |
| Exact weights: apply to 1 spectrum | 3.9 ms | 3.5 ms | 1.5 ms | 1.0 ms |
| Exact weights: apply to 64 spectra | 293 ms | 96 ms | 111 ms | 26 ms |
| Cut weights: build | n/a | 5.8 ms | n/a | 5.8 ms |
| Cut weights: apply to 64 spectra | n/a | 8.4 ms | n/a | 2.2 ms |

- **One-shot smoothing** is 2.2 to 5.3 times faster on one CPU and 3.4 to 20 times faster on four, because pykooh's numba kernel runs on one thread.
- **Building exact weights** is 9 to 18 times faster. On one grid (`freq_out` left as `freq_in`) the window is symmetric, so `PrecomputedWeights` computes and stores each pair once: 0.9 GB instead of pykooh's 1.8 GB dense matrix.
- **Applying exact weights to one spectrum** is 1.1 to 1.6 times faster. Both libraries are limited by memory bandwidth here. On one grid `fast-konno-ohmachi` reads half the bytes; on the log-spaced outputs both read 50 MB, and the margin comes from streaming four weight rows at once from huge pages.
- **Applying exact weights to 64 spectra** is 2.5 to 4.3 times faster. `apply_many` reads each weight row once per block of spectra, while pykooh's `CachedSmoother`, which has no batch method, reads the whole matrix once per spectrum.
- **Results agree** to within 2e-14 of the peak with the exact window and 3e-12 with the cut window. At 42 of the log-spaced outputs (43 with the cut window), all below 0.016 Hz, pykooh returns NaN because the window weights there sum to less than 0.3; `fast-konno-ohmachi` returns a value, falling back to the exact window where the cut window holds no sample.

`demo/benchmark_pykooh.py` reproduces these tables; it needs `pykooh` and `scipy`.

## Version 0.2

0.2 replaces the pure-Python implementation. `fast_konno_ohmachi`, `faster_konno_ohmachi`, and `slow_konno_ohmachi` are gone, along with `smooth_coeff`, `progress_bar`, and the even-integer limit on the bandwidth.

```python
# 0.1
smoothed = fast_konno_ohmachi(signal, freq, smooth_coeff=40, progress_bar=False)

# 0.2
smoothed = ko.smooth(signal, freq, bandwidth=40)
```
