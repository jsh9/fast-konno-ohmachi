//! Konno-Ohmachi spectral smoothing.

use std::sync::Arc;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyType};

use crate::kernel::{
    apply_many, apply_plan, bad_core_count, build_plan, parse_dtype, parse_method, resolve_cores,
    smooth_signal, KernelError, Plan,
};

mod kernel;

/// ``(copyreg.__newobj_ex__, (cls, args, kwargs))`` for ``__reduce__``.
type Reduce<'py> = (
    Bound<'py, PyAny>,
    (
        Bound<'py, PyType>,
        (Bound<'py, PyArray1<f64>>, f64),
        Bound<'py, PyDict>,
    ),
);

fn value_error(message: impl ToString) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// Bad arguments raise ``ValueError``; weights that do not fit raise ``MemoryError``.
fn kernel_error(err: KernelError) -> PyErr {
    match err {
        KernelError::Invalid(message) => PyValueError::new_err(message),
        KernelError::Memory(message) => PyMemoryError::new_err(message),
    }
}

/// Extracts ``n_cores``. An integer too large for ``isize`` is out of range like
/// any other count above the CPU limit, so it raises ``ValueError`` rather than
/// the ``OverflowError`` a plain ``isize`` argument would.
fn n_cores_arg(obj: &Bound<'_, PyAny>) -> PyResult<Option<isize>> {
    if obj.is_none() {
        return Ok(None);
    }
    match obj.extract::<isize>() {
        Ok(n) => Ok(Some(n)),
        Err(err) if err.is_instance_of::<PyOverflowError>(obj.py()) => {
            Err(kernel_error(bad_core_count(obj, obj.gt(0)?)))
        }
        Err(err) => Err(err),
    }
}

fn unreadable(err: impl ToString) -> PyErr {
    value_error(format!(
        "could not read a numeric array: {}",
        err.to_string()
    ))
}

/// `numpy.require(array, float64, ["C", "A"])`: C-contiguous, aligned, native
/// float64, flattened first when `flatten` is set. A contiguous array can
/// still be misaligned (a byte buffer with an odd offset), so contiguity alone
/// is not enough. The result is a view of `obj` when it already qualifies, so
/// the kernel reads the caller's buffer without a copy.
fn f64_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    flatten: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let mut array = numpy.call_method1("asarray", (obj,))?;
    if flatten {
        array = array.call_method0("ravel")?;
    }
    numpy.call_method1(
        "require",
        (array, numpy.getattr("float64")?, vec!["C", "A"]),
    )
}

fn f64_array_1d<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    f64_array(py, obj, true)?
        .cast_into::<PyArray1<f64>>()
        .map_err(unreadable)
}

fn f64_array_2d<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let array = f64_array(py, obj, false)?;
    let ndim: usize = array.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(value_error(
            "signals must be a 2-D array of shape (n_records, n_freq)",
        ));
    }
    array.cast_into::<PyArray2<f64>>().map_err(unreadable)
}

/// Owned copy of a 1-D array, for values the plan keeps.
fn as_f64_vec(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    f64_array_1d(py, obj)?.to_vec().map_err(unreadable)
}

fn vector_out(py: Python<'_>, data: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    PyArray1::from_vec(py, data)
}

/// Hand the result to NumPy without copying it.
fn matrix_out(
    py: Python<'_>,
    data: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), data)
        .map_err(|err| value_error(format!("could not shape the result: {err}")))?;
    Ok(array.into_pyarray(py))
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
///     How many cores to use. ``None`` uses every logical CPU available to
///     the process. A value below 1 or above that count raises
///     ``ValueError``. Default is 1.
#[pyfunction]
#[pyo3(signature = (signal, freq_in, bandwidth=40.0, *, freq_out=None, method="approx", n_cores=1))]
fn smooth<'py>(
    py: Python<'py>,
    signal: &Bound<'py, PyAny>,
    freq_in: &Bound<'py, PyAny>,
    bandwidth: f64,
    freq_out: Option<&Bound<'py, PyAny>>,
    method: &str,
    #[pyo3(from_py_with = n_cores_arg)] n_cores: Option<isize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let signal = f64_array_1d(py, signal)?;
    let signal = signal.readonly();
    let signal = signal.as_slice().map_err(unreadable)?;
    let freq_in = as_f64_vec(py, freq_in)?;
    let freq_out = match freq_out {
        Some(values) => as_f64_vec(py, values)?,
        None => freq_in.clone(),
    };
    let method = parse_method(method).map_err(kernel_error)?;
    let n_cores = resolve_cores(n_cores).map_err(kernel_error)?;
    let smoothed = py
        .detach(|| smooth_signal(&freq_in, &freq_out, signal, bandwidth, method, n_cores))
        .map_err(kernel_error)?;
    Ok(vector_out(py, smoothed))
}

/// ``dtype`` as NumPy spells it: ``"float32"`` for ``np.float32``, a dtype
/// object, or the string itself. ``None`` is ``"float64"``.
fn dtype_name(py: Python<'_>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(dtype) = dtype else {
        return Ok("float64".to_string());
    };
    // Strings go through NumPy too, so "f4", "single" and np.float32 agree.
    let numpy = py.import("numpy")?;
    let dtype = numpy
        .call_method1("dtype", (dtype,))
        .map_err(|_| value_error("dtype must be \"float64\" or \"float32\""))?;
    dtype.getattr("name")?.extract()
}

/// Weights of the Konno-Ohmachi window for one frequency grid.
///
/// Build this once when many spectra share ``freq_in``, ``freq_out``, and
/// ``bandwidth``. :meth:`apply` and :meth:`apply_many` only multiply those
/// stored weights. An ``"approx"`` frequency with no input sample inside the
/// cutoff stores the full exact window for that row.
///
/// ``dtype`` is ``"float64"`` or ``"float32"``, as a string or a NumPy dtype,
/// and sets the precision of the stored weights; results stay float64. The
/// attributes ``freq_in``, ``freq_out``, ``bandwidth``, ``method`` and
/// ``dtype`` read back the constructor arguments, and the object pickles and
/// copies by rebuilding from them.
#[pyclass(name = "PrecomputedWeights", module = "fast_konno_ohmachi")]
struct PrecomputedWeights {
    plan: Arc<Plan>,
}

#[pymethods]
impl PrecomputedWeights {
    #[new]
    #[pyo3(signature = (freq_in, bandwidth=40.0, *, freq_out=None, method="approx", dtype=None))]
    fn new(
        py: Python<'_>,
        freq_in: &Bound<'_, PyAny>,
        bandwidth: f64,
        freq_out: Option<&Bound<'_, PyAny>>,
        method: &str,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let freq_in = as_f64_vec(py, freq_in)?;
        let freq_out = match freq_out {
            Some(values) => as_f64_vec(py, values)?,
            None => freq_in.clone(),
        };
        let method = parse_method(method).map_err(kernel_error)?;
        let dtype = parse_dtype(&dtype_name(py, dtype)?).map_err(kernel_error)?;
        let plan = py
            .detach(move || build_plan(freq_in, freq_out, bandwidth, method, dtype))
            .map_err(kernel_error)?;
        Ok(Self {
            plan: Arc::new(plan),
        })
    }

    /// Frequency of each input sample, in hertz.
    #[getter]
    fn freq_in<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.plan.freq_in)
    }

    /// Frequencies at which the smooth is written.
    #[getter]
    fn freq_out<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.plan.freq_out)
    }

    /// Konno-Ohmachi bandwidth ``b``.
    #[getter]
    fn bandwidth(&self) -> f64 {
        self.plan.bandwidth
    }

    /// ``"approx"`` or ``"exact"``.
    #[getter]
    fn method(&self) -> &'static str {
        self.plan.method.name()
    }

    /// ``"float64"`` or ``"float32"``: the precision of the stored weights.
    #[getter]
    fn dtype(&self) -> &'static str {
        self.plan.dtype.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "PrecomputedWeights(n_in={}, n_out={}, bandwidth={:?}, method='{}', dtype='{}')",
            self.plan.freq_in.len(),
            self.plan.freq_out.len(),
            self.plan.bandwidth,
            self.method(),
            self.dtype()
        )
    }

    /// ``pickle`` and ``copy`` rebuild the weights from the constructor
    /// arguments. ``copyreg.__newobj_ex__`` calls ``cls.__new__(cls, *args,
    /// **kwargs)`` and, unlike ``__getnewargs_ex__``, works for every pickle
    /// protocol, 0 and 1 included.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduce<'py>> {
        let py = slf.py();
        let this = slf.borrow();
        let kwargs = PyDict::new(py);
        kwargs.set_item("freq_out", this.freq_out(py))?;
        kwargs.set_item("method", this.method())?;
        kwargs.set_item("dtype", this.dtype())?;
        let rebuild = py.import("copyreg")?.getattr("__newobj_ex__")?;
        let args = (this.freq_in(py), this.plan.bandwidth);
        Ok((rebuild, (slf.get_type(), args, kwargs)))
    }

    /// Apply the stored weights to one spectrum.
    #[pyo3(signature = (signal, n_cores=1))]
    fn apply<'py>(
        &self,
        py: Python<'py>,
        signal: &Bound<'py, PyAny>,
        #[pyo3(from_py_with = n_cores_arg)] n_cores: Option<isize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let signal = f64_array_1d(py, signal)?;
        let signal = signal.readonly();
        let signal = signal.as_slice().map_err(unreadable)?;
        let n_cores = resolve_cores(n_cores).map_err(kernel_error)?;
        let plan = Arc::clone(&self.plan);
        let smoothed = py
            .detach(|| apply_plan(&plan, signal, n_cores))
            .map_err(kernel_error)?;
        Ok(vector_out(py, smoothed))
    }

    /// Apply the stored weights to a batch of spectra.
    ///
    /// ``signals`` has shape ``(n_records, n_freq)``. ``n_cores`` defaults to
    /// every logical CPU available to the process. A value below 1 or above
    /// that count raises ``ValueError``.
    #[pyo3(signature = (signals, n_cores=None))]
    fn apply_many<'py>(
        &self,
        py: Python<'py>,
        signals: &Bound<'py, PyAny>,
        #[pyo3(from_py_with = n_cores_arg)] n_cores: Option<isize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let signals = f64_array_2d(py, signals)?;
        let signals = signals.readonly();
        let dims = signals.dims();
        let (rows, cols) = (dims[0], dims[1]);
        let signals: &[f64] = if signals.is_empty() {
            &[]
        } else {
            signals.as_slice().map_err(unreadable)?
        };
        let n_cores = resolve_cores(n_cores).map_err(kernel_error)?;
        let plan = Arc::clone(&self.plan);
        let n_out = plan.freq_out.len();
        let smoothed = py
            .detach(|| apply_many(&plan, signals, rows, cols, n_cores))
            .map_err(kernel_error)?;
        matrix_out(py, smoothed, rows, n_out)
    }
}

#[pymodule]
fn fast_konno_ohmachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_class::<PrecomputedWeights>()?;
    Ok(())
}
