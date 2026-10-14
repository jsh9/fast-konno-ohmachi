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


def _accepts_n_cores(n_cores):
    try:
        ko.smooth([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_cores=n_cores)
    except ValueError:
        return False
    return True


# An explicit n_cores above the logical CPU count is an error, so two workers
# only run where the host has them. n_cores=None works everywhere.
TWO_CORES = pytest.param(
    2, marks=pytest.mark.skipif(not _accepts_n_cores(2), reason="needs two logical CPUs")
)


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
        direct = ko.smooth(signal, freq, bandwidth=40, method="approx", n_cores=None)
        cached = weights.apply(signal, n_cores=None)
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


CACHE_DTYPES = [("float64", 1e-12), ("float32", 1e-6)]


@pytest.mark.parametrize("dtype, rtol", CACHE_DTYPES)
def test_tiny_bandwidth_on_a_log_grid_averages_every_sample(dtype, rtol):
    freq = np.array([1.0, 10.0, 100.0])
    signal = np.array([1.0, 2.0, 3.0])
    direct = ko.smooth(signal, freq, bandwidth=1e-20)
    exact = ko.smooth(signal, freq, bandwidth=1e-20, method="exact")
    assert np.isfinite(direct).all()
    np.testing.assert_allclose(direct, 2.0, rtol=1e-12)
    np.testing.assert_allclose(direct, exact, rtol=1e-12)

    weights = ko.PrecomputedWeights(freq, bandwidth=1e-20, dtype=dtype)
    np.testing.assert_allclose(weights.apply(signal), direct, rtol=rtol)
    batch = weights.apply_many(np.vstack([signal, signal[::-1]]))
    np.testing.assert_allclose(batch, np.vstack([direct, direct]), rtol=rtol)


@pytest.mark.parametrize("bandwidth", [1e-20, 0.05, 1.0, 5.0])
def test_window_wider_than_the_grid_matches_pointwise(bandwidth):
    # Same log-uniform grid in and out, so smooth takes the convolution path.
    freq = np.logspace(0, 1, 50)
    signal = np.sin(np.log(freq)) ** 2 + 0.3
    direct = ko.smooth(signal, freq, bandwidth=bandwidth)
    reference = _reference(signal, freq, freq, bandwidth, "approx")
    np.testing.assert_allclose(direct, reference, rtol=1e-8, atol=1e-10)

    pointwise = ko.PrecomputedWeights(freq, bandwidth=bandwidth, freq_out=freq[:-1])
    np.testing.assert_allclose(direct[:-1], pointwise.apply(signal), rtol=1e-12, atol=1e-12)
    for dtype, rtol in CACHE_DTYPES:
        cached = ko.PrecomputedWeights(freq, bandwidth=bandwidth, dtype=dtype)
        np.testing.assert_allclose(cached.apply(signal), direct, rtol=rtol)


def _two_sample_window(bandwidth, method):
    x = bandwidth * np.log10(2.0)
    if method == "approx" and abs(x) > 3.0:
        return 0.0
    return (np.sin(x) / x) ** 4


@pytest.mark.parametrize("scale", [1.0, 1e-20])
@pytest.mark.parametrize("bandwidth, method", [(40, "exact"), (2, "exact"), (2, "approx")])
def test_two_close_small_frequencies_keep_their_window(scale, bandwidth, method):
    freq = np.array([1.0, 2.0]) * scale
    signal = np.array([0.0, 1.0])
    neighbor = _two_sample_window(bandwidth, method)
    expected = np.array([neighbor, 1.0]) / (1.0 + neighbor)

    direct = ko.smooth(signal, freq, bandwidth=bandwidth, method=method)
    np.testing.assert_allclose(direct, expected, rtol=1e-9, atol=1e-15)
    for dtype, rtol in CACHE_DTYPES:
        weights = ko.PrecomputedWeights(freq, bandwidth=bandwidth, method=method, dtype=dtype)
        np.testing.assert_allclose(weights.apply(signal), expected, rtol=max(rtol, 1e-9), atol=1e-7)
        batch = weights.apply_many(np.vstack([signal, signal[::-1]]))
        np.testing.assert_allclose(
            batch, np.vstack([expected, expected[::-1]]), rtol=max(rtol, 1e-9), atol=1e-7
        )


@pytest.mark.parametrize("method", ["approx", "exact"])
def test_smoothing_is_invariant_to_frequency_scale(method):
    rng = np.random.default_rng(7)
    freq = np.sort(rng.uniform(0.5, 20.0, 60))
    freq_out = np.sort(rng.uniform(0.6, 19.0, 25))
    signal = rng.uniform(0.1, 2.0, freq.size)
    base = ko.smooth(signal, freq, bandwidth=8, freq_out=freq_out, method=method)
    base_same = ko.smooth(signal, freq, bandwidth=8, method=method)
    for scale in [1e-20, 1e-6, 1e3, 1e20]:
        scaled = ko.smooth(signal, freq * scale, bandwidth=8, freq_out=freq_out * scale, method=method)
        np.testing.assert_allclose(scaled, base, rtol=1e-9, atol=1e-12)
        same = ko.smooth(signal, freq * scale, bandwidth=8, method=method)
        np.testing.assert_allclose(same, base_same, rtol=1e-9, atol=1e-12)
        for dtype, rtol in CACHE_DTYPES:
            weights = ko.PrecomputedWeights(
                freq * scale, bandwidth=8, freq_out=freq_out * scale, method=method, dtype=dtype
            )
            np.testing.assert_allclose(weights.apply(signal), base, rtol=max(rtol, 1e-9))


def test_log_grid_is_invariant_to_frequency_scale():
    freq = np.logspace(0, 2, 120)
    signal = np.sin(np.log(freq)) ** 2 + 0.3
    base = ko.smooth(signal, freq, bandwidth=20)
    for scale in [1e-20, 1e20]:
        direct = ko.smooth(signal, freq * scale, bandwidth=20)
        np.testing.assert_allclose(direct, base, rtol=1e-9)
        pointwise = ko.PrecomputedWeights(freq * scale, bandwidth=20, freq_out=freq[:-1] * scale)
        np.testing.assert_allclose(pointwise.apply(signal), direct[:-1], rtol=1e-9)


@pytest.mark.parametrize("scale", [1.0, 1e-20])
def test_single_sample_and_constant_signal_survive_small_frequencies(scale):
    for method in ("approx", "exact"):
        single = ko.smooth([5.0], [scale], method=method)
        np.testing.assert_array_equal(single, [5.0])
        freq = np.array([1.0, 1.5, 4.0, 9.0]) * scale
        constant = ko.smooth(np.full(freq.size, 2.5), freq, bandwidth=2, method=method)
        np.testing.assert_allclose(constant, 2.5, rtol=1e-12)


def test_log_grid_smooth_matches_precomputed_weights():
    freq = np.logspace(-2, 2, 500)
    signal = np.sin(np.log(freq)) ** 2 + 0.3
    direct = ko.smooth(signal, freq, bandwidth=30, method="approx")
    weights = ko.PrecomputedWeights(freq, bandwidth=30, method="approx")
    cached = weights.apply(signal, n_cores=None)
    np.testing.assert_allclose(cached, direct, rtol=1e-12, atol=1e-12)

    reference = _reference(signal, freq, freq, 30, "approx")
    np.testing.assert_allclose(direct, reference, rtol=1e-6, atol=1e-8)


def test_exact_matches_published_sum_on_a_short_grid():
    freq = np.linspace(0.2, 12.0, 90)
    signal = np.cos(freq) ** 2 + 0.1
    for method in ("exact", "approx"):
        direct = ko.smooth(signal, freq, bandwidth=15.5, method=method, n_cores=None)
        weights = ko.PrecomputedWeights(freq, bandwidth=15.5, method=method)
        cached = weights.apply(signal, n_cores=1)
        reference = _reference(signal, freq, freq, 15.5, method)
        np.testing.assert_allclose(direct, reference, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(cached, direct, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("n_cores", [TWO_CORES, None])
def test_n_cores_do_not_change_the_result(n_cores):
    spec, freq = _load_spectrum()
    one = ko.smooth(spec, freq, bandwidth=40, n_cores=1)
    many = ko.smooth(spec, freq, bandwidth=40, n_cores=n_cores)
    np.testing.assert_array_equal(many, one)

    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    np.testing.assert_array_equal(
        weights.apply(spec, n_cores=n_cores), weights.apply(spec, n_cores=1)
    )


@pytest.mark.parametrize("n_cores", [1, TWO_CORES, None])
def test_apply_many_matches_row_by_row(n_cores):
    spec, freq = _load_spectrum()
    rows = np.vstack([spec, np.sqrt(spec), spec[::-1], np.full_like(spec, spec.mean())])
    weights = ko.PrecomputedWeights(freq, bandwidth=40)
    batch = weights.apply_many(rows, n_cores=n_cores)
    assert batch.shape == rows.shape
    for row, smoothed in zip(rows, batch):
        np.testing.assert_array_equal(smoothed, weights.apply(row, n_cores=1))

    default_batch = weights.apply_many(rows)
    np.testing.assert_array_equal(default_batch, batch)


@pytest.mark.parametrize("n_cores", [TWO_CORES, None])
@pytest.mark.parametrize("bandwidth", [0.5, 7.0, 40.0])
def test_log_grid_convolution_is_identical_for_any_worker_count(bandwidth, n_cores):
    # At bandwidth 0.5 the capped kernel spans the whole grid.
    freq = np.logspace(-1, 1, 2001)
    signal = np.sin(3 * np.log(freq)) + 1.5
    serial = ko.smooth(signal, freq, bandwidth, n_cores=1)
    np.testing.assert_array_equal(ko.smooth(signal, freq, bandwidth, n_cores=n_cores), serial)

    rows = np.vstack([signal, signal[::-1], 2 * signal])
    for dtype, rtol in CACHE_DTYPES:
        weights = ko.PrecomputedWeights(freq, bandwidth, dtype=dtype)
        cached = weights.apply(signal, n_cores=1)
        np.testing.assert_allclose(cached, serial, rtol=rtol)
        np.testing.assert_array_equal(weights.apply(signal, n_cores=n_cores), cached)

        batch = weights.apply_many(rows, n_cores=n_cores)
        for row, smoothed in zip(rows, batch):
            np.testing.assert_array_equal(smoothed, weights.apply(row, n_cores=1))
        single = weights.apply_many(signal[None, :], n_cores=n_cores)
        np.testing.assert_array_equal(single[0], cached)
        empty = weights.apply_many(np.empty((0, freq.size)), n_cores=n_cores)
        assert empty.shape == (0, freq.size)


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


def _unaligned(values):
    """Copy ``values`` into a C-contiguous float64 array that is not aligned."""
    values = np.asarray(values, dtype=np.float64)
    buffer = bytearray(values.nbytes + 8)
    base = np.frombuffer(buffer, dtype=np.uint8).ctypes.data
    offset = next(k for k in range(1, 8) if (base + k) % values.itemsize)
    out = np.ndarray(values.shape, dtype=np.float64, buffer=buffer, offset=offset)
    out[...] = values
    assert out.flags.c_contiguous
    assert not out.flags.aligned
    return out


def test_unaligned_inputs_match_aligned_copies():
    rng = np.random.default_rng(3)
    freq = np.sort(rng.uniform(0.5, 20.0, 40))
    freq_out = np.linspace(1.0, 18.0, 15)
    signal = rng.uniform(0.1, 2.0, freq.size)
    expected = ko.smooth(signal, freq, bandwidth=10, freq_out=freq_out)

    for name in ("signal", "freq_in", "freq_out"):
        args = {"signal": signal, "freq_in": freq, "freq_out": freq_out}
        args[name] = _unaligned(args[name])
        before = args[name].copy()
        result = ko.smooth(args["signal"], args["freq_in"], 10, freq_out=args["freq_out"])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(args[name], before)

    weights = ko.PrecomputedWeights(_unaligned(freq), 10, freq_out=_unaligned(freq_out))
    reference = ko.PrecomputedWeights(freq, 10, freq_out=freq_out)
    np.testing.assert_array_equal(weights.apply(_unaligned(signal)), reference.apply(signal))

    rows = np.vstack([signal, signal[::-1], np.sqrt(signal)])
    unaligned_rows = _unaligned(rows)
    batch = weights.apply_many(unaligned_rows)
    assert batch.shape == (3, freq_out.size)
    np.testing.assert_array_equal(batch, reference.apply_many(rows))
    np.testing.assert_array_equal(unaligned_rows, rows)
    for row, smoothed in zip(rows, batch):
        np.testing.assert_array_equal(smoothed, reference.apply(row))


def test_unusual_array_layouts_match_plain_copies():
    rng = np.random.default_rng(5)
    freq = np.sort(rng.uniform(0.5, 20.0, 41))
    signal = rng.uniform(0.1, 2.0, freq.size)
    weights = ko.PrecomputedWeights(freq, bandwidth=12)
    expected = weights.apply(signal)

    strided = np.empty(2 * freq.size)
    strided[::2] = signal
    read_only = signal.copy()
    read_only.flags.writeable = False
    variants = (strided[::2], read_only, signal.astype(">f8"), signal[::-1], signal.astype(np.float32))
    for variant in variants:
        plain = np.array(variant, dtype=np.float64)
        assert plain.flags.c_contiguous and plain.flags.aligned
        np.testing.assert_array_equal(weights.apply(variant), weights.apply(plain))
        np.testing.assert_array_equal(ko.smooth(variant, freq, 12), ko.smooth(plain, freq, 12))
    np.testing.assert_array_equal(weights.apply(strided[::2]), expected)

    rows = np.vstack([signal, signal[::-1], signal * 2])
    fortran = np.asfortranarray(rows)
    assert not fortran.flags.c_contiguous
    np.testing.assert_array_equal(weights.apply_many(fortran), weights.apply_many(rows))

    empty = weights.apply_many(np.empty((0, freq.size)))
    assert empty.shape == (0, freq.size)
    single = weights.apply_many(signal[None, :])
    np.testing.assert_array_equal(single[0], expected)


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


def test_exact_weights_too_large_raise_before_allocating():
    # 70000 * 70000 weights exceed what the class can index, so this must fail
    # before any of them is allocated.
    freq = np.linspace(1.0, 2.0, 70_000)
    with pytest.raises(ValueError, match=r'smooth\(method="exact"\)'):
        ko.PrecomputedWeights(freq, bandwidth=40, method="exact")
    with pytest.raises(ValueError, match=r'smooth\(method="exact"\)'):
        ko.PrecomputedWeights(freq, bandwidth=40, method="exact", dtype="float32")


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
