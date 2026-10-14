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

`n_cores` is how many cores to use. The default for `smooth` is 1. `None` uses every logical CPU, and a larger value is an error. The workers are ordinary threads; they are not pinned to cores.

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

With `method="exact"`, `PrecomputedWeights` stores one weight per (input, output) pair: `len(freq_in) * len(freq_out) * 8` bytes, or 4 with `dtype="float32"`. The 15000-point spectrum in `demo/` needs 1.8 GB. A matrix that does not fit raises `MemoryError`, and one with more than 2**32 - 1 weights raises `ValueError`. For a one-off exact smooth use `smooth(method="exact")`, which stores no weights.

On a log-uniform grid, with `method="approx"` and `freq_out` left as `freq_in`, both paths use a short convolution instead of a weight per output frequency.

`demo/Demo_konno_ohmachi_smooth.py` smooths the spectrum plotted above. Running it needs `scipy` and `matplotlib`.

## Version 0.2

0.2 replaces the pure-Python implementation. `fast_konno_ohmachi`, `faster_konno_ohmachi`, and `slow_konno_ohmachi` are gone, along with `smooth_coeff`, `progress_bar`, and the even-integer limit on the bandwidth.

```python
# 0.1
smoothed = fast_konno_ohmachi(signal, freq, smooth_coeff=40, progress_bar=False)

# 0.2
smoothed = ko.smooth(signal, freq, bandwidth=40)
```
