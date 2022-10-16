"""
[Description]
    This module performs Konno-Ohmachi smoothing very fast.

    It has pre-calculated smoothing window values built-in, thus
    circumventing the need to re-calculate "w" every time. Compared to
    ordinary Konno-Ohmachi smoothing algorithms, this module is ~50% faster.

[References]
    The formula of Konno-Ohmachi smoothing window is here:
        http://www.geopsy.org/wiki/index.php/Smoothing_details

    Original paper:
        K. Konno & T. Ohmachi (1998) "Ground-motion characteristics estimated
        from spectral ratio between horizontal and vertical components of
        microtremor." Bulletin of the Seismological Society of America.
        Vol.88, No.1, 228-241.

        Abstract:
            http://www.bssaonline.org/content/88/1/228.short
        K. Konno's own PDF copy:
            http://www.eq.db.shibaura-it.ac.jp/papers/Konno&Ohmachi1998.pdf
"""

import sys
import itertools
import multiprocessing as mp
import numpy as np

from fast_konno_ohmachi._precalculated import PRE_CALCULATED_SMOOTHING_WINDOWS


def fast_konno_ohmachi(raw_signal, freq_array, smooth_coeff=40, progress_bar=True):
    """
    Perform fast Konno-Ohmachi filtering. The speedup is achieved by
    using pre-calculated smoothing windows.

    Parameters
    ----------
    raw_signal : array-like
        Signal to be smoothed in frequency domain. 1-dimensional. (For optimum
        speed, use 1D numpy array as input.)
    freq_array : array-like
        Frequency array corresponding to the signal. It must have the same
        length as "raw_signal". Unit: Hz. (For optimum speed, use 1D numpy
        array as input.)
    smooth_coeff : int
        A parameter determining the degree of smoothing. The lower this
        parameter, the more the signal is smoothed. Only even integers between
        2 and 100 are allowed. Out-of-range values are constraint to [2,100].
        Non-integer values are rounded. Odd integers are implicitly converted
        to even integers.
    progress_bar : bool
        Whether to print a progress bar.

    Returns
    -------
    numpy.ndarray
        The smoothed signal (1D numpy array).
    """
    x = np.array(raw_signal).flatten()
    f = np.array(freq_array).flatten()
    b = smooth_coeff

    if round(b) != b:
        b = round(b)   # round non integers

    if np.remainder(b, 2) == 1:  # if b is odd
        b = b-1  # make it even

    if b < 2:
        b = 2  # "cup" b value

    if b > 100:
        b = 100  # "cap" b value

    if len(x) != len(f):
        msg = 'Length of input signal and frequency array must be the same.'
        raise ValueError(msg)

    L = len(x)
    y = np.zeros(L)  # pre-allocation of smoothed signal

    # Extract (b/2)-th row from A as "reference array" because A's 1st row
    # corresponds to b = 2, and A's 2nd row corresponds to b = 4, etc.
    # Note: "int(b/2.0)-1" has "-1" because row index in Python starts from 0
    ref_array = PRE_CALCULATED_SMOOTHING_WINDOWS[int(b / 2.0) - 1, :]

    ref_z = np.arange(0.5, 2.001, 0.001)  # equivalent to 0.5:0.001:2 in MATLAB

    if progress_bar:
        progress_bar_width = 40  # unit: characters
        print('\n|------------    Progress    ------------|')  # reference bar
        sys.stdout.write('|')

    # =======  Moving window smoothing: fc from f[1] to f[-2]  ========
    for i in range(L):
        if progress_bar and (np.remainder(i, L//progress_bar_width) == 0):
            sys.stdout.write('|')  # prints "|" without spaces or new lines

        if (i == 0) or (i == L-1):
            continue  # skip first and last indices for now

        fc = f[i]  # central frequency
        w = np.zeros(L)  # pre-allocation of smoothing window "w"

        z = f / fc  # "z" = dimensionless frequency, normalized by fc
        z = z[np.where(z >= 0.5)]  # only keep elements between 0.5 and 2.0,
        z = z[np.where(z <= 2.0)]  # because outsize [0.5, 2.0], w is almost 0

        w0 = np.interp(z, ref_z, ref_array)  # 1D interpolation
              # Note: In practice, w is almost 0 when z (normalized frequency)
              #       is outside [0.5, 2.0].  Thus only the non-zero part of
              #       "w", i.e., "w0", is calculated via interpolation.
              #       Then, "w" is reconstructed from "w0" by padding zeros.

        idx = np.argmax(w0) # the index where w0 has maximum value
        shift = i+1 - idx # i+1 = "true index" (starting from 1 rather than 0)
        w0 = np.lib.pad(w0, (shift, 0), mode='constant', constant_values=(0))
                  # shift w0 to the right by "shift", and pad zeros in front
        if len(w0) >= len(w):  # if length of w0 already exceeds w
            w = w0[0:len(w)]  # trim w0 down to the same length as w
        else:  # otherwise, put w0 into w
            w[0:len(w0)] = w0

        y[i] = np.dot(w, x) / np.sum(w)  # apply smoothing filter to "x"

    y[0] = y[1]  # calculate first and last indices
    y[-1] = y[-2]

    if progress_bar:
        sys.stdout.write('|\n')

    return y


def faster_konno_ohmachi(raw_signal, freq_array, smooth_coeff=40, n_cores=None):
    """
    Performs faster Konno-Ohmachi smoothing. It is faster than
    ``fast_konno_ohmachi()`` because it uses multi-processing.

    Parameters
    ----------
    raw_signal : array-like
        Signal to be smoothed in frequency domain. 1-dimensional. (For optimum
        speed, use 1D numpy array as input.)
    freq_array : array-like
        Frequency array corresponding to the signal. It must have the same
        length as "raw_signal". Unit: Hz. (For optimum speed, use 1D numpy
        array as input.)
    smooth_coeff : int
        A parameter determining the degree of smoothing. The lower this
        parameter, the more the signal is smoothed. Only even integers between
        2 and 100 are allowed. Out-of-range values are constraint to [2,100].
        Non-integer values are rounded. Odd integers are implicitly converted
        to even integers.
    n_cores : int or None
        Number of CPU cores to use in parallel computing. If ``None``, all cores
        are used.

    Returns
    -------
    numpy.ndarray
        The smoothed signal (1D numpy array).
    """
    x = np.array(raw_signal).flatten()
    f = np.array(freq_array).flatten()
    b = smooth_coeff

    if round(b) != b:
        b = round(b)   # round non integers
    if np.remainder(b, 2) == 1:  # if b is odd
        b = b - 1  # make it even
    if b < 2:
        b = 2  # "cup" b value
    if b > 100:
        b = 100  # "cap" b value

    if len(x) != len(f):
        msg = 'Length of input signal and frequency array must be the same.'
        raise ValueError(msg)

    L = len(x)

    # Extract (b/2)-th row from A as "reference array" because A's 1st row
    # corresponds to b = 2, and A's 2nd row corresponds to b = 4, etc.
    # Note: "int(b/2.0)-1" has "-1" because row index in Python starts from 0
    ref_array = PRE_CALCULATED_SMOOTHING_WINDOWS[int(b / 2.0) - 1, :]

    ref_z = np.arange(0.5, 2.001, 0.001)  # equivalent to 0.5:0.001:2 in MATLAB

    # =======  Moving window smoothing: fc from f[1] to f[-2]  ========
    p = mp.Pool(n_cores)
    y = p.map(
        _loop_body,
        itertools.product(range(L), [f], [L], [ref_z], [ref_array], [x]),
    )

    y[0] = y[1]  # calculate first and last indices
    y[-1] = y[-2]

    return y


def _loop_body(para):
    """
    The loop body to be passed to the paralle pool.

    A subroutine for faster_konno_ohmachi().

    Notes:
    1. Due to the limitation of the multiprocessing module of Python, it cannot
       be put within faster_konno_ohmachi() as a local function.)
    2. Python 2.7 does not have Pool.starmap method, thus the loop body can only
       take one parameter, and then unpack.
    """
    i, f, L, ref_z, ref_array, x = para  # unpack

    fc = f[i]  # central frequency
    w = np.zeros(L)  # pre-allocation of smoothing window "w"

    z = f / fc  # "z" = dimensionless frequency, normalized by fc
    z = z[np.where(z >= 0.5)]  # only keep elements between 0.5 and 2.0,
    z = z[np.where(z <= 2.0)]  # because outsize [0.5, 2.0], w is almost 0

    w0 = np.interp(z,ref_z,ref_array)  # 1D interpolation
          # Note: In practice, w is almost 0 when z (normalized frequency)
          #       is outside [0.5, 2.0].  Thus only the non-zero part of
          #       "w", i.e., "w0", is calculated via interpolation.
          #       Then, "w" is reconstructed from "w0" by padding zeros.

    idx = np.argmax(w0) # the index where w0 has maximum value
    shift = i+1 - idx # i+1 = "true index" (starting from 1 rather than 0)
    w0 = np.lib.pad(w0, (shift, 0) ,mode='constant', constant_values=(0))
              # shift w0 to the right by "shift", and pad zeros in front
    if len(w0) >= len(w):  # if length of w0 already exceeds w
        w = w0[0:len(w)]  # trim w0 down to the same length as w
    else:  # otherwise, put w0 into w
        w[0:len(w0)] = w0

    y_i = np.dot(w, x) / np.sum(w)  # apply smoothing filter to "x"

    return y_i


def slow_konno_ohmachi(raw_signal, freq_array, smooth_coeff=40, progress_bar=True):
    """
    Perform regular Konno-Ohmachi smoothing, in which the smoothing windows are
    calculated on the fly. (That's why it is very slow.)

    There is no need to actually use this function except to test the
    correctness of ``fast_konno_ohmachi()``.

    Parameters
    ----------
    raw_signal : array-like
        Signal to be smoothed in frequency domain. 1-dimensional. (For optimum
        speed, use 1D numpy array as input.)
    freq_array : array-like
        Frequency array corresponding to the signal. It must have the same
        length as "raw_signal". Unit: Hz. (For optimum speed, use 1D numpy
        array as input.)
    smooth_coeff : float
        A parameter determining the degree of smoothing. The lower this
        parameter, the more the signal is smoothed.
    progress_bar : bool
        Whether to print a progress bar.

    Returns
    -------
    numpy.ndarray
        The smoothed signal (1D numpy array).
    """
    x = np.array(raw_signal).flatten()
    f = np.array(freq_array).flatten()
    f_shifted = f / (1 + 1e-4)  # shift slightly to avoid numerical errors
    b = float(smooth_coeff)

    if len(x) != len(f):
        msg = 'Length of input signal and frequency array must be the same.'
        raise ValueError(msg)

    L = len(x)
    y = np.zeros(L)  # pre-allocation of smoothed signal

    if progress_bar:
        progress_bar_width = 40  # unit: characters
        print('\n|------------    Progress    ------------|')  # reference bar
        sys.stdout.write('|')

    # =======  Moving window smoothing: fc from f[1] to f[-2]  ========
    for i in range(L):
        if progress_bar and (np.remainder(i, L/progress_bar_width) == 0):
            sys.stdout.write('|')  # prints "|" without spaces or new lines

        if (i == 0) or (i == L-1):
            continue  # skip first and last indices for now

        fc = f[i]  # central frequency
        w = np.zeros(L)  # pre-allocation of smoothing window "w"

        z = f_shifted / fc  # "z" = dimensionless frequency, normalized by fc
        w = (np.sin(b * np.log10(z)) / b / np.log10(z)) ** 4.0
        w[np.isnan(w)] = 0  # replace NaN's with 0

        y[i] = np.dot(w,x) / np.sum(w)  # apply smoothing filter to "x"

    y[0] = y[1]  # calculate first and last indices
    y[-1] = y[-2]

    if progress_bar:
        sys.stdout.write('|\n')

    return y
