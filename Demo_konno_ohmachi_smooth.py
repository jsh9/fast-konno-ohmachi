# -*- coding: utf-8 -*-
'''
This Python script uses fast_konno_ohmachi to smooth a sample Fourier spectrum.
To demonstrate the benefit of Konno-Ohmachi smoothing, we also smooth the
raw signal with a median filter, and plot both results.

The raw signal used in this example is the Fourier amplitude spectrum of a
ground acceleration waveform recorded during 2011/3/11 Magnitude-9.0 Tohoku-Oki
Earthquake.

(c) 2013-2019, Jian Shi
'''

import time
import numpy as np
import scipy.signal
import scipy.io as sio
import matplotlib.pyplot as plt

import konno_ohmachi as ko

d = sio.loadmat('./spectrum.mat')  # load data from MAT file
freq = d['freq']  # frequency array; np.shape(freq) should be (15000L, 1L)
spec = d['spec']  # Fourier amplitude spectrum; np.shape(spec) = (15000L, 1L)
freq = np.ndarray.flatten(freq)  # flatten array such that shape = (15000L,)
spec = np.ndarray.flatten(spec)  # flatten array such that shape = (15000L,)

spec = np.repeat(spec,4)
freq = np.linspace(np.min(freq), np.max(freq), len(spec))

N = 2
spec = spec[::N]
freq = freq[::N]

if __name__ == '__main__':
    __spec__ = None  # this is to circumvent a bug with using Spyder

    t1 = time.time()
    y0 = ko.fast_konno_ohmachi(spec,freq,progress_bar=True)
    t2 = time.time()
    print('\nElapsed time (fast-konno-ohmachi): %.1f sec.' % (t2 - t1))

    t3 = time.time()
    y1 = ko.faster_konno_ohmachi(spec,freq,n_cores=None)
    t4 = time.time()
    print('\nElapsed time (faster-konno-ohmachi): %.1f sec.' % (t4 - t3))

    assert(np.allclose(y0, y1))  # make sure y0 and y1 are almost identical

    y2 = scipy.signal.medfilt(np.ndarray.flatten(spec),kernel_size=201)

    fig = plt.figure(figsize=(4,3),dpi=150,edgecolor='k',facecolor='w')
    ax = plt.axes()
    plt.semilogx(freq,spec,c=[0.5]*3,lw=0.5,label='Raw siginal')
    plt.semilogx(freq,y1,'r',lw=1.5,label='Konno-Ohmachi')
    plt.semilogx(freq,y2,'b',lw=1.0,label='Median filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Fourier Amplitude Spectra')
    plt.legend(loc='best',fontsize=10.5)
    plt.grid(color=[0.75]*3,ls=':')
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.3)
    plt.show()
