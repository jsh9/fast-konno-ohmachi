# _Fast_ Konno-Ohmachi smoothing
#### This is a Python module that performs Konno-Ohmachi spectral smoothing very fast (reduces running time by ~60%).

## Description
Konno-Ohmachi is a smoothing algorithm proposed by Konno & Ohmachi (1998) [[Abstract](http://bssa.geoscienceworld.org/content/88/1/228.short), [PDF](http://www.eq.db.shibaura-it.ac.jp/papers/Konno&Ohmachi1998.pdf)], which achieves a "uniform-span" smoothing to frequency spectra in logarithmic scale.

For lower frequencies, the Konno-Ohmachi smoothing window is narrower (i.e., less smoothing), and for higher frequencies, the window is wider (i.e., more smoothing). "Conventional" smoothing filters use same smoothing window widths at all locations. This feature of the Konno-Ohmachi filter is preferred in engineering seismology, where the variations of amplitudes in lower frequencies (< 10 Hz) are more important than in higher frequencies.

The figure below shows the result of Konno-Ohmachi filter versus a "conventional" median value filter [[Wiki](https://en.wikipedia.org/wiki/Median_filter)]. The two filters yield similar results for frequency > 5 Hz, but for lower frequencies, the median filter over-smoothes the raw spectrum, which is undesirable.

![](demo.png)
###### (The raw signal used in this example is the Fourier amplitude spectrum of a ground acceleration waveform recorded during 2011/3/11 Magnitude-9.0 Tohoku-Oki Earthquake.)

## Computation speed
Konno-Ohmachi filter is time-consuming due to varying window widths. This module stores pre-calculated smoothing window values, which, compared to other ordinary Konno-Ohmachi smoothers, reduces the calculation time by ~60% (hence the "fast" in the module name).

## Subroutines

This module has three subroutines:

+ `fast_konno_ohmachi()`: the "fast" method, using single CPU core;
+ `faster_konno_ohmachi()`: the "faster" method, using multiple CPU cores;
+ `slow_konno_ohmachi()`: the "slow" method, which does not use pre-calculated smoothing window.

## How to use this module
Put `konno_ohmachi.py` in your Python search path. Then,

```python
import konno_ohmachi as ko
smoothed = ko.fast_konno_ohmachi(spectrum, freq, smooth_coeff=40, verbose=True)
```

or (the "faster" function, using multiple CPU cores):

```python
smoothed = ko.faster_konno_ohmachi(spectrum, freq, smooth_coeff=40, n_cores=4)
```

See `Demo_konno_ohmachi_smooth.py` for more detailed examples.

## Notes on parallel computing

1. When using `faster_konno_ohmachi()`, the user should to protect the main script with `if __name__ == '__main__'` (see the demo script). This is **mandatory** for Windows, and **highly recommended** for Mac/Linux.
2. The `faster_konno_ohmachi()` function uses multiple CPU cores, but it is not necessarily faster than `fast_konno_ohmachi()`, because the data I/O between the CPU cores takes extra time ("computation overhead"). Below is a benchmarking of the running time for input signals with different length:

| Length of  signal (x1000) | 1    | 3    | 5    | 7    | 9    | 11   | 13   | 15   | 17   | 19   | 21   | 23   | 25   | 27   | 29   | 31   |
| ------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Time of "fast" (sec)      | 0.1  | 0.4  | 0.9  | 1.5  | 2.2  | 3.3  | 4.4  | 5.4  | 6.8  | 8.1  | 9.6  | 11.2 | 13.1 | 14.9 | 16.7 | 18.8 |
| Time of "faster" (sec)    | 2.4  | 2.4  | 2.8  | 3.0  | 3.3  | 3.9  | 4.1  | 4.6  | 4.9  | 5.5  | 5.8  | 6.7  | 7.5  | 8.2  | 9.0  | 9.8  |


Or as shown in this figure:

![](./benchmark.png)

## Dependencies

`konno_ohmachi.py` only dependes on Numpy 1.11.0+; works for both Python 2.7 and Python 3+.

In order to run `Demo_konno_ohmachi_smooth.py`, you also need Scipy 0.17.1+, and Matplotlib 1.5.1+.


## Limitations
`fast_konno_ohmachi` only supports **even integers** between [2,100] as eligible smoothing coefficient (i.e., "b"). Out-of-range values, odd integers, and/or decimal numbers will be constraint within [2,100] and rounded to an even integer.

This is merely to reduce the file size of the source code (because smoothing windows are pre-calculated and hard-coded into the source). In practice, b values outside of [2,100] are very rarely used.

## License
Copyright (c) 2013-2019, Jian Shi. See LICENSE.txt file for details.
