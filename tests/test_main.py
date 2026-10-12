from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

import fast_konno_ohmachi as ko


SPECTRUM = Path(__file__).resolve().parents[1] / "demo" / "spectrum.mat"

# Pointwise error of method="approx" against method="exact", as a fraction of
# the peak of the exact smooth. These are the thresholds the previous
# interpolating implementation was held to on the native Tohoku record.
NATIVE_THRESHOLDS = [
    (80, 0.07),
    (60, 0.06),
    (40, 0.03),
    (20, 0.02),
    (10, 0.01),
]


def _load_spectrum():
    data = sio.loadmat(SPECTRUM.as_posix())
    freq = np.ravel(data["freq"]).astype(np.float64)
    spec = np.ravel(data["spec"]).astype(np.float64)
    return spec, freq


def _assert_close(expected, actual, thres):
    expected = np.asarray(expected, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    assert expected.shape == actual.shape

    area_actual = np.trapezoid(actual)
    area_expected = np.trapezoid(expected)
    assert abs(area_actual - area_expected) / abs(area_expected) <= 0.005

    relative = np.abs(actual - expected) / np.max(expected)
    worst = int(np.argmax(relative))
    assert relative[worst] < thres, f"{relative[worst]} >= {thres} at i = {worst}"


def _reference(signal, freq_in, freq_out, bandwidth, method):
    """Published window, summed in NumPy.

    ``approx`` keeps samples with ``|b log10(f / fc)| <= 3 + 1e-9``, the same
    cutoff the convolution kernel uses. An output with no sample in that
    range uses the full exact window.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    freq_in = np.asarray(freq_in, dtype=np.float64).ravel()
    freq_out = np.asarray(freq_out, dtype=np.float64).ravel()
    log_in = np.log10(freq_in)
    out = np.empty(freq_out.shape, dtype=np.float64)
    for i, fc in enumerate(freq_out):
        x = bandwidth * (log_in - np.log10(fc))
        if method == "approx":
            keep = np.abs(x) <= 3.0 + 1e-9
            if not np.any(keep):
                keep = np.ones(freq_in.shape, dtype=bool)
        else:
            keep = np.ones(freq_in.shape, dtype=bool)
        weights = np.zeros_like(x)
        weights[keep] = 1.0
        nonzero = keep & (np.abs(x) >= 1e-12)
        sinc = np.sin(x[nonzero]) / x[nonzero]
        weights[nonzero] = sinc ** 4
        out[i] = np.dot(weights, signal) / np.sum(weights)
    return out


@pytest.mark.parametrize("bandwidth, thres", NATIVE_THRESHOLDS)
def test_approx_matches_exact_on_native_record(bandwidth, thres):
    spec, freq = _load_spectrum()
    exact = ko.smooth(spec, freq, bandwidth, method="exact", n_cores=None)
    approx = ko.smooth(spec, freq, bandwidth=bandwidth, method="approx", n_cores=1)
    _assert_close(expected=exact, actual=approx, thres=thres)


def test_precomputed_weights_match_smooth_for_two_spectra():
    spec, freq = _load_spectrum()
    other = np.sqrt(spec)
    weights = ko.PrecomputedWeights(freq, bandwidth=40, method="approx")
    for signal in (spec, other):
        direct = ko.smooth(signal, freq, bandwidth=40, method="approx", n_cores=2)
        cached = weights.apply(signal, n_cores=2)
        np.testing.assert_allclose(cached, direct, rtol=1e-9, atol=1e-12)


def test_precomputed_weights_follow_freq_out():
    spec, freq = _load_spectrum()
    freq_out = freq[::50]
    direct = ko.smooth(spec, freq, bandwidth=25.5, freq_out=freq_out)
    weights = ko.PrecomputedWeights(freq, bandwidth=25.5, freq_out=freq_out)
    cached = weights.apply(spec)
    assert cached.shape == freq_out.shape
    assert cached.dtype == np.float64
    np.testing.assert_allclose(cached, direct, rtol=1e-9, atol=1e-12)


def test_float32_weights_stay_close():
    spec, freq = _load_spectrum()
    direct = ko.smooth(spec, freq, bandwidth=40)
    weights = ko.PrecomputedWeights(freq, bandwidth=40, dtype="float32")
    cached = weights.apply(spec)
    np.testing.assert_allclose(cached, direct, rtol=1e-5, atol=1e-8)


def test_empty_approx_window_uses_the_exact_sum():
    freq = np.arange(1.0, 11.0)
    freq_out = np.array([1.5, 2.0, 5.5])
    constant = ko.smooth(np.ones(freq.size), freq, bandwidth=100, freq_out=freq_out)
    assert np.isfinite(constant).all()
    np.testing.assert_allclose(constant, 1.0)

    signal = np.linspace(0.2, 1.7, freq.size)
    approx = ko.smooth(signal, freq, bandwidth=100, freq_out=freq_out, method="approx")
    exact = ko.smooth(signal, freq, bandwidth=100, freq_out=freq_out, method="exact")
    assert np.isfinite(approx).all()
    np.testing.assert_allclose(approx[[0, 2]], exact[[0, 2]], rtol=1e-12, atol=1e-12)
    reference = _reference(signal, freq, freq_out, 100, "approx")
    np.testing.assert_allclose(approx, reference, rtol=1e-8, atol=1e-10)
    weights = ko.PrecomputedWeights(freq, bandwidth=100, freq_out=freq_out, method="approx")
    np.testing.assert_allclose(weights.apply(signal), approx, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("bandwidth", [20, 30, 60, 100])
def test_window_edge_is_consistent_on_integer_tap_grids(bandwidth):
    freq = np.logspace(0, 2, 201)
    signal = np.sin(np.log(freq)) ** 2 + 0.3
    direct = ko.smooth(signal, freq, bandwidth=bandwidth)
    weights = ko.PrecomputedWeights(freq, bandwidth=bandwidth, freq_out=freq[:-1])
    cached = weights.apply(signal)
    np.testing.assert_allclose(direct[:-1], cached, rtol=1e-12, atol=1e-12)
    reference = _reference(signal, freq, freq, bandwidth, "approx")
    np.testing.assert_allclose(direct, reference, rtol=1e-8, atol=1e-8)


def test_log_grid_smooth_matches_precomputed_weights():
    freq = np.logspace(-2, 2, 500)
    signal = np.sin(np.log(freq)) ** 2 + 0.3
    direct = ko.smooth(signal, freq, bandwidth=30, method="approx")
    weights = ko.PrecomputedWeights(freq, bandwidth=30, method="approx")
    cached = weights.apply(signal, n_cores=2)
    np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=1e-12)

    reference = _reference(signal, freq, freq, 30, "approx")
    np.testing.assert_allclose(direct, reference, rtol=1e-6, atol=1e-8)


def test_exact_matches_published_sum_on_a_short_grid():
    freq = np.linspace(0.2, 12.0, 90)
    signal = np.cos(freq) ** 2 + 0.1
    for method in ("exact", "approx"):
        direct = ko.smooth(signal, freq, bandwidth=15.5, method=method, n_cores=2)
        weights = ko.PrecomputedWeights(freq, bandwidth=15.5, method=method)
        cached = weights.apply(signal, n_cores=1)
        reference = _reference(signal, freq, freq, 15.5, method)
        np.testing.assert_allclose(direct, reference, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(cached, direct, rtol=1e-9, atol=1e-12)


def test_n_cores_do_not_change_the_result():
    spec, freq = _load_spectrum()
    one = ko.smooth(spec, freq, bandwidth=40, n_cores=1)
    two = ko.smooth(spec, freq, bandwidth=40, n_cores=2)
    np.testing.assert_array_equal(two, one)

    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    np.testing.assert_array_equal(weights.apply(spec, n_cores=2), weights.apply(spec, n_cores=1))


def test_apply_many_matches_row_by_row():
    spec, freq = _load_spectrum()
    rows = np.vstack([spec, np.sqrt(spec), spec[::-1], np.full_like(spec, spec.mean())])
    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    batch = weights.apply_many(rows, n_cores=2)
    assert batch.shape == rows.shape
    for row, smoothed in zip(rows, batch):
        np.testing.assert_allclose(smoothed, weights.apply(row, n_cores=1), rtol=1e-12, atol=1e-12)

    default_batch = weights.apply_many(rows)
    np.testing.assert_array_equal(default_batch, batch)


def test_endpoints_are_computed():
    freq = np.arange(1, 30, dtype=np.float64)
    signal = np.zeros_like(freq)
    signal[0] = 1.0
    smoothed = ko.smooth(signal, freq, bandwidth=40, method="exact")
    assert smoothed[0] > smoothed[1]


def test_constant_signal_is_unchanged():
    freq = np.arange(1, 40, dtype=np.float64)
    smoothed = ko.smooth(np.full(freq.shape, 3.5), freq, bandwidth=40)
    np.testing.assert_allclose(smoothed, 3.5, rtol=0, atol=1e-9)


def test_column_vectors_are_accepted():
    spec, freq = _load_spectrum()
    column_spec = spec.reshape(-1, 1)
    column_freq = freq.reshape(-1, 1)
    flat = ko.smooth(spec, freq, bandwidth=40)
    column = ko.smooth(column_spec, column_freq, bandwidth=40)
    np.testing.assert_array_equal(column, flat)


@pytest.mark.parametrize(
    "freq",
    [
        np.array([0.1, 0.0, 0.3]),
        np.array([0.2, -1.0, 0.4]),
        np.array([0.1, np.nan, 0.3]),
        np.array([0.1, np.inf, 0.3]),
        np.array([0.1, 0.4, 0.2]),
        np.array([0.1, 0.2, 0.2]),
        np.array([]),
    ],
)
def test_bad_frequencies_raise(freq):
    signal = np.ones_like(freq)
    with pytest.raises(ValueError):
        ko.smooth(signal, freq, bandwidth=40)
    with pytest.raises(ValueError):
        ko.PrecomputedWeights(freq, bandwidth=40)


@pytest.mark.parametrize("bandwidth", [0, -5, np.inf, np.nan])
def test_bad_bandwidth_raises(bandwidth):
    freq = np.arange(1, 6, dtype=np.float64)
    signal = np.ones_like(freq)
    with pytest.raises(ValueError):
        ko.smooth(signal, freq, bandwidth=bandwidth)


def test_n_cores_above_the_machine_raises():
    freq = np.arange(1, 8, dtype=np.float64)
    signal = np.ones_like(freq)
    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    rows = np.vstack([signal, signal])
    for call in (
        lambda n_cores: ko.smooth(signal, freq, n_cores=n_cores),
        lambda n_cores: weights.apply(signal, n_cores=n_cores),
        lambda n_cores: weights.apply_many(rows, n_cores=n_cores),
    ):
        with pytest.raises(ValueError, match="n_cores"):
            call(10**6)


def test_bad_method_dtype_cores_and_shapes_raise():
    freq = np.arange(1, 8, dtype=np.float64)
    signal = np.ones_like(freq)
    with pytest.raises(ValueError):
        ko.smooth(signal, freq, method="standard")
    with pytest.raises(ValueError):
        ko.PrecomputedWeights(freq, dtype="float16")
    with pytest.raises(ValueError):
        ko.smooth(signal, freq, n_cores=0)
    with pytest.raises(ValueError):
        ko.smooth(signal[:-1], freq)
    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    with pytest.raises(ValueError):
        weights.apply(signal[:-1])
    with pytest.raises(ValueError):
        weights.apply_many(signal)
    with pytest.raises(ValueError):
        weights.apply_many(np.ones((2, freq.size - 1)))
    with pytest.raises(TypeError):
        weights(signal)
