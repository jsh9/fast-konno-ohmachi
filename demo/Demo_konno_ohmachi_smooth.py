"""
Smooth a sample Fourier spectrum with fast_konno_ohmachi.

The raw signal is the Fourier amplitude spectrum of a ground acceleration
waveform recorded during the 2011/3/11 Magnitude-9.0 Tohoku-Oki earthquake.
A median filter is plotted alongside so the extra smoothing at low frequency
is visible.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.signal

import fast_konno_ohmachi as ko


def main():
    data = sio.loadmat(Path(__file__).resolve().parent / "spectrum.mat")
    freq = np.ravel(data["freq"])
    spec = np.ravel(data["spec"])

    started = time.perf_counter()
    smoothed = ko.smooth(spec, freq, bandwidth=40, n_cores=1)
    print(f"Elapsed time (smooth, 1 core): {time.perf_counter() - started:.3f} sec.")

    started = time.perf_counter()
    parallel = ko.smooth(spec, freq, bandwidth=40, n_cores=None)
    print(f"Elapsed time (smooth, all cores): {time.perf_counter() - started:.3f} sec.")
    if not np.allclose(smoothed, parallel):
        raise AssertionError("n_cores changed the smoothed spectrum")

    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    if not np.allclose(weights.apply(spec), smoothed):
        raise AssertionError("PrecomputedWeights disagreed with smooth")

    median = scipy.signal.medfilt(spec, kernel_size=201)

    plt.figure(figsize=(4, 3), dpi=150, edgecolor="k", facecolor="w")
    ax = plt.axes()
    plt.semilogx(freq, spec, c=[0.5] * 3, lw=0.5, label="Raw signal")
    plt.semilogx(freq, smoothed, "r", lw=1.5, label="Konno-Ohmachi")
    plt.semilogx(freq, median, "b", lw=1.0, label="Median filter")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Fourier Amplitude Spectra")
    plt.legend(loc="best", fontsize=10.5)
    plt.grid(color=[0.75] * 3, ls=":")
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.3)
    plt.show()


if __name__ == "__main__":
    main()
