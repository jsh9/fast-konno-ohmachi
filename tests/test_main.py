import pytest
from pathlib import Path
import numpy as np
import scipy.io as sio

import fast_konno_ohmachi as fko


def _load_mat_file():
    data = sio.loadmat((Path.cwd() / Path('demo/spectrum.mat')).as_posix())
    freq = data['freq']
    spec = data['spec']
    return spec, freq


def _assert_close(expected, actual, thres):
    assert len(expected) == len(actual)

    area_actual = np.trapz(actual)
    area_expected = np.trapz(expected)

    assert np.abs(area_actual - area_expected) / area_expected <= 0.005

    max_expected = max(expected)
    for i in range(len(expected)):
        relative_diff = abs(actual[i] - expected[i]) / max_expected
        if relative_diff >= thres:
            raise ValueError(f'{relative_diff} >= {thres} at i = {i}')


test_cases = [
    (80, 0.07),
    (60, 0.06),
    (40, 0.03),
    (20, 0.02),
    (10, 0.01),
]


@pytest.mark.parametrize('smooth_coeff, thres', test_cases)
def test_fast_konno_ohmachi(smooth_coeff, thres):
    spec, freq = _load_mat_file()
    smoothed_fast = fko.fast_konno_ohmachi(
        spec, freq, smooth_coeff=smooth_coeff, progress_bar=False,
    )
    smoothed_slow = fko.slow_konno_ohmachi(
        spec, freq, smooth_coeff=smooth_coeff, progress_bar=False,
    )
    _assert_close(expected=smoothed_slow, actual=smoothed_fast, thres=thres)


test_cases_2 = [
    (80, 0.14),
    (60, 0.11),
    (40, 0.06),
    (20, 0.04),
    (10, 0.03),
]


@pytest.mark.parametrize('smooth_coeff, thres', test_cases_2)
def test_fast_konno_ohmachi__downsampled(smooth_coeff, thres):
    spec, freq = _load_mat_file()

    spec = np.array(spec).flatten()[::2]
    freq = np.array(freq).flatten()[::2]

    smoothed_fast = fko.fast_konno_ohmachi(
        spec, freq, smooth_coeff=smooth_coeff, progress_bar=False,
    )
    smoothed_slow = fko.slow_konno_ohmachi(
        spec, freq, smooth_coeff=smooth_coeff, progress_bar=False,
    )
    _assert_close(expected=smoothed_slow, actual=smoothed_fast, thres=thres)



def test_faster_konno_ohmachi():
    spec, freq = _load_mat_file()
    smoothed_fast = fko.fast_konno_ohmachi(
        spec, freq, smooth_coeff=40, progress_bar=False,
    )
    smoothed_faster = fko.faster_konno_ohmachi(
        spec, freq, smooth_coeff=40, n_cores=None,
    )
    assert list(smoothed_faster) == list(smoothed_fast)
