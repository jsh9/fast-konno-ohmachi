//! Konno-Ohmachi spectral smoothing.

use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::kernel::{
    apply_many, apply_plan, build_plan, parse_dtype, parse_method, resolve_cores, smooth_signal,
    Plan,
};

mod kernel;

fn value_error(message: impl ToString) -> PyErr {
    PyValueError::new_err(message.to_string())
}

fn as_f64_1d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (obj,))?;
    let flat = array.call_method0("ravel")?;
    let contiguous = numpy.call_method1("ascontiguousarray", (flat, numpy.getattr("float64")?))?;
    let typed = contiguous.cast::<PyArray1<f64>>()?;
    typed
        .to_vec()
        .map_err(|err| value_error(format!("could not read a numeric array: {err}")))
}

fn as_f64_2d(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f64>, usize, usize)> {
    let numpy = py.import("numpy")?;
    let array = numpy.call_method1("asarray", (obj,))?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(value_error(
            "signals must be a 2-D array of shape (n_records, n_freq)",
        ));
    }
    let contiguous = numpy.call_method1("ascontiguousarray", (array, numpy.getattr("float64")?))?;
    let typed = contiguous.cast::<PyArray2<f64>>()?;
    let dims = typed.dims();
    let data = typed
        .to_vec()
        .map_err(|err| value_error(format!("could not read signals: {err}")))?;
    Ok((data, dims[0], dims[1]))
}

fn vector_out(py: Python<'_>, data: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    PyArray1::from_vec(py, data)
}

fn matrix_out(
    py: Python<'_>,
    data: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), data)
        .map_err(|err| value_error(format!("could not shape the result: {err}")))?;
    Ok(array.to_pyarray(py))
}

fn output_frequencies(freq_in: &[f64], freq_out: Option<Vec<f64>>) -> Vec<f64> {
    freq_out.unwrap_or_else(|| freq_in.to_vec())
}

/// Smooth one spectrum with the Konno-Ohmachi window.
///
/// Parameters
/// ----------
/// signal : array-like
///     Spectrum samples, one per ``freq_in`` value.
/// freq_in : array-like
///     Frequency of each input sample, in hertz. Finite, positive, and
///     strictly increasing.
/// bandwidth : float, optional
///     Konno-Ohmachi bandwidth ``b``. Lower values smooth more. Default is 40.
/// freq_out : array-like, optional
///     Frequencies at which to write the smooth. Defaults to ``freq_in``.
/// method : {"approx", "exact"}, optional
///     ``"approx"`` keeps frequencies with ``|b log10(f / fc)| <= 3`` and
///     evaluates the window with a polynomial. An output frequency with no
///     input sample in that range uses the full exact window instead.
///     ``"exact"`` sums the published window over every positive frequency.
///     Default is ``"approx"``.
/// n_cores : int or None, optional
///     How many cores to use. ``None`` uses every logical CPU. A value above
///     that count is an error. Default is 1.
#[pyfunction]
#[pyo3(signature = (signal, freq_in, bandwidth=40.0, *, freq_out=None, method="approx", n_cores=1))]
fn smooth<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
    freq_in: &Bound<'py, PyAny>,
    bandwidth: f64,
    freq_out: Option<&Bound<'py, PyAny>>,
    method: &str,
    n_cores: Option<isize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = as_f64_1d(py, signal)?;
    let freq_in = as_f64_1d(py, freq_in)?;
    let freq_out = match freq_out {
        Some(values) => Some(as_f64_1d(py, values)?),
        None => None,
    };
    let method = parse_method(method).map_err(value_error)?;
    let n_cores = resolve_cores(n_cores).map_err(value_error)?;
    let smoothed = py
        .detach(move || {
            let freq_out = output_frequencies(&freq_in, freq_out);
            smooth_signal(&freq_in, &freq_out, &signal, bandwidth, method, n_cores)
        })
        .map_err(value_error)?;
    Ok(vector_out(py, smoothed))
}

/// Weights of the Konno-Ohmachi window for one frequency grid.
///
/// Build this once when many spectra share ``freq_in``, ``freq_out``, and
/// ``bandwidth``. :meth:`apply` and :meth:`apply_many` only multiply those
/// stored weights. An ``"approx"`` frequency with no input sample inside the
/// cutoff stores the full exact window for that row.
#[pyclass(name = "PrecomputedWeights")]
struct PrecomputedWeights {
    plan: Arc<Plan>,
}

#[pymethods]
impl PrecomputedWeights {
    #[new]
    #[pyo3(signature = (freq_in, bandwidth=40.0, *, freq_out=None, method="approx", dtype="float64"))]
    fn new(
        py: Python<'_>,
        freq_in: &Bound<'_, PyAny>,
        bandwidth: f64,
        freq_out: Option<&Bound<'_, PyAny>>,
        method: &str,
        dtype: &str,
    ) -> PyResult<Self> {
        let freq_in = as_f64_1d(py, freq_in)?;
        let freq_out = match freq_out {
            Some(values) => Some(as_f64_1d(py, values)?),
            None => None,
        };
        let method = parse_method(method).map_err(value_error)?;
        let dtype = parse_dtype(dtype).map_err(value_error)?;
        let plan = py
            .detach(move || {
                let freq_out = output_frequencies(&freq_in, freq_out);
                build_plan(freq_in, freq_out, bandwidth, method, dtype)
            })
            .map_err(value_error)?;
        Ok(Self {
            plan: Arc::new(plan),
        })
    }

    /// Apply the stored weights to one spectrum.
    #[pyo3(signature = (signal, n_cores=1))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        signal: &Bound<'py, PyAny>,
        n_cores: Option<isize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let signal = as_f64_1d(py, signal)?;
        let n_cores = resolve_cores(n_cores).map_err(value_error)?;
        let plan = Arc::clone(&self.plan);
        let smoothed = py
            .detach(move || apply_plan(&plan, &signal, n_cores))
            .map_err(value_error)?;
        Ok(vector_out(py, smoothed))
    }

    /// Apply the stored weights to a batch of spectra.
    ///
    /// ``signals`` has shape ``(n_records, n_freq)``. ``n_cores`` defaults to
    /// every logical CPU. A value above that count is an error.
    #[pyo3(signature = (signals, n_cores=None))]
    fn apply_many<'py>(
        &self,
        py: Python<'py>,
        signals: &Bound<'py, PyAny>,
        n_cores: Option<isize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (signals, rows, cols) = as_f64_2d(py, signals)?;
        let n_cores = resolve_cores(n_cores).map_err(value_error)?;
        let plan = Arc::clone(&self.plan);
        let n_out = plan.freq_out.len();
        let smoothed = py
            .detach(move || apply_many(&plan, &signals, rows, cols, n_cores))
            .map_err(value_error)?;
        matrix_out(py, smoothed, rows, n_out)
    }
}

#[pymodule]
fn fast_konno_ohmachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_class::<PrecomputedWeights>()?;
    Ok(())
}
