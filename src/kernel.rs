//! Konno-Ohmachi smoothing kernels.
//!
//! `method="approx"` keeps samples with `|b log10(f / fc)| <= 3`. When that
//! range contains no input sample, that output uses the full exact window.
//! `method="exact"` sums the published window over every positive frequency.

use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use crate::simd;

/// Why a kernel call could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelError {
    /// Bad arguments. Python sees a `ValueError`.
    Invalid(String),
    /// The weights do not fit in memory. Python sees a `MemoryError`.
    Memory(String),
}

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::Invalid(message) | KernelError::Memory(message) => f.write_str(message),
        }
    }
}

fn invalid(message: impl Into<String>) -> KernelError {
    KernelError::Invalid(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Approx,
    Exact,
}

impl Method {
    /// The spelling the Python API uses.
    pub fn name(self) -> &'static str {
        match self {
            Method::Approx => "approx",
            Method::Exact => "exact",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
}

impl DType {
    /// The NumPy spelling.
    pub fn name(self) -> &'static str {
        match self {
            DType::F64 => "float64",
            DType::F32 => "float32",
        }
    }

    fn bytes(self) -> usize {
        match self {
            DType::F64 => std::mem::size_of::<f64>(),
            DType::F32 => std::mem::size_of::<f32>(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Values {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

#[derive(Clone, Debug)]
pub enum Kind {
    /// Short kernel for a log-uniform grid. Index 0 is the most negative lag.
    Convolution { kernel: Values },
    /// Contiguous normalized weights. Row `i` uses `values[start[i]..start[i+1]]`
    /// against `signal[col0[i]..]`.
    Sparse {
        col0: Vec<u32>,
        start: Vec<u32>,
        values: Values,
    },
    /// Exact window with `freq_out == freq_in`. The unnormalized window is
    /// symmetric with ones on the diagonal, so only the strict upper triangle
    /// is stored: row `i` holds the weights of inputs `i+1..n`, starting at
    /// `symmetric_offset(n, i)`. `scale[i]` is one over row `i`'s full sum.
    Symmetric { values: Values, scale: Vec<f64> },
}

#[derive(Debug)]
pub struct Plan {
    pub freq_in: Vec<f64>,
    pub freq_out: Vec<f64>,
    pub bandwidth: f64,
    pub method: Method,
    pub dtype: DType,
    pub kind: Kind,
    /// Flips on every one-spectrum `apply` of stored weights, which then runs
    /// through the rows in the other direction. Weights slightly larger than
    /// the last-level cache then start each call on the rows the previous
    /// call left cached. Rows are independent, so results do not change.
    sweep: AtomicBool,
}

pub fn parse_method(method: &str) -> Result<Method, KernelError> {
    match method {
        "approx" => Ok(Method::Approx),
        "exact" => Ok(Method::Exact),
        _ => Err(invalid("method must be \"approx\" or \"exact\"")),
    }
}

pub fn parse_dtype(dtype: &str) -> Result<DType, KernelError> {
    match dtype {
        "float64" => Ok(DType::F64),
        "float32" => Ok(DType::F32),
        _ => Err(invalid("dtype must be \"float64\" or \"float32\"")),
    }
}

/// Logical CPUs this process may use. CPU quotas and affinity masks count, so
/// this can be fewer than the machine has.
pub fn available_cores() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
}

/// The error for an `n_cores` outside `1..=available_cores()`, quoting the value
/// as given. `positive` says which side of the range it fell off.
pub fn bad_core_count(n_cores: impl std::fmt::Display, positive: bool) -> KernelError {
    if positive {
        invalid(format!(
            "n_cores ({n_cores}) exceeds the {} logical CPUs available to this process",
            available_cores()
        ))
    } else {
        invalid(format!("n_cores must be >= 1 or None, got {n_cores}"))
    }
}

pub fn resolve_cores(n_cores: Option<isize>) -> Result<usize, KernelError> {
    let available = available_cores();
    match n_cores {
        None => Ok(available),
        Some(n) if n >= 1 && (n as usize) <= available => Ok(n as usize),
        Some(n) => Err(bad_core_count(n, n >= 1)),
    }
}

pub fn validate_freq(freq: &[f64], name: &str) -> Result<(), KernelError> {
    if freq.is_empty() {
        return Err(invalid(format!("{name} must not be empty")));
    }
    for (i, &value) in freq.iter().enumerate() {
        if !value.is_finite() {
            return Err(invalid(format!("{name}[{i}] must be finite")));
        }
        if value <= 0.0 {
            return Err(invalid(format!(
                "{name}[{i}] must be positive, got {value}"
            )));
        }
        if i > 0 && value <= freq[i - 1] {
            return Err(invalid(format!("{name} must be strictly increasing")));
        }
    }
    Ok(())
}

pub fn validate_bandwidth(bandwidth: f64) -> Result<(), KernelError> {
    if bandwidth.is_finite() && bandwidth > 0.0 {
        Ok(())
    } else {
        Err(invalid("bandwidth must be a positive finite number"))
    }
}

pub fn same_grid(freq_in: &[f64], freq_out: &[f64]) -> bool {
    freq_in.len() == freq_out.len() && freq_in.iter().zip(freq_out).all(|(a, b)| a == b)
}

/// The log step when `method="approx"` can run as a convolution: the output
/// grid is the input grid and that grid is log-uniform.
fn convolution_spacing(freq_in: &[f64], freq_out: &[f64], method: Method) -> Option<f64> {
    if method == Method::Approx && same_grid(freq_in, freq_out) {
        log_spacing(freq_in)
    } else {
        None
    }
}

pub fn build_plan(
    freq_in: Vec<f64>,
    freq_out: Vec<f64>,
    bandwidth: f64,
    method: Method,
    dtype: DType,
) -> Result<Plan, KernelError> {
    validate_freq(&freq_in, "freq_in")?;
    validate_freq(&freq_out, "freq_out")?;
    validate_bandwidth(bandwidth)?;
    let kind = match convolution_spacing(&freq_in, &freq_out, method) {
        Some(du) => Kind::Convolution {
            kernel: store_values(approx_kernel(bandwidth, du, freq_in.len())?, dtype),
        },
        None if method == Method::Exact && same_grid(&freq_in, &freq_out) => {
            build_symmetric(&freq_in, bandwidth, dtype)?
        }
        None => build_sparse(&freq_in, &freq_out, bandwidth, method, dtype)?,
    };
    Ok(Plan {
        freq_in,
        freq_out,
        bandwidth,
        method,
        dtype,
        kind,
        sweep: AtomicBool::new(false),
    })
}

pub fn smooth_signal(
    freq_in: &[f64],
    freq_out: &[f64],
    signal: &[f64],
    bandwidth: f64,
    method: Method,
    n_cores: usize,
) -> Result<Vec<f64>, KernelError> {
    validate_freq(freq_in, "freq_in")?;
    validate_freq(freq_out, "freq_out")?;
    validate_bandwidth(bandwidth)?;
    if signal.len() != freq_in.len() {
        return Err(invalid(format!(
            "signal length ({}) must match freq_in ({})",
            signal.len(),
            freq_in.len()
        )));
    }
    if let Some(du) = convolution_spacing(freq_in, freq_out, method) {
        let kernel = approx_kernel(bandwidth, du, freq_in.len())?;
        let mut out = vec![0.0; signal.len()];
        convolve_into(signal, &kernel, &mut out, n_cores)?;
        return Ok(out);
    }
    pointwise(freq_in, freq_out, signal, bandwidth, method, n_cores)
}

pub fn apply_plan(plan: &Plan, signal: &[f64], n_cores: usize) -> Result<Vec<f64>, KernelError> {
    if signal.len() != plan.freq_in.len() {
        return Err(invalid(format!(
            "signal length ({}) must match freq_in ({})",
            signal.len(),
            plan.freq_in.len()
        )));
    }
    let mut out = vec![0.0; plan.freq_out.len()];
    let reverse = plan.sweep.fetch_xor(true, Ordering::Relaxed);
    apply_into(plan, signal, &mut out, n_cores, reverse)?;
    Ok(out)
}

pub fn apply_many(
    plan: &Plan,
    signals: &[f64],
    n_rows: usize,
    n_cols: usize,
    n_cores: usize,
) -> Result<Vec<f64>, KernelError> {
    if n_cols != plan.freq_in.len() {
        return Err(invalid(format!(
            "signals has {} columns but freq_in has length {}",
            n_cols,
            plan.freq_in.len()
        )));
    }
    if signals.len() != n_rows * n_cols {
        return Err(invalid("signals length does not match its shape"));
    }
    let n_out = plan.freq_out.len();
    let mut flat = zeroed_output(n_rows, n_out)?;
    if n_rows == 0 {
        return Ok(flat);
    }
    if let Kind::Sparse {
        col0,
        start,
        values,
    } = &plan.kind
    {
        let rows = SparseRows {
            col0,
            start,
            values,
        };
        rows.apply_many(signals, n_cols, n_cores, &mut flat)?;
        return Ok(flat);
    }
    if let Kind::Symmetric { values, scale } = &plan.kind {
        let rows = SymmetricRows { values, scale };
        // As many spectra per pass over the weights as keep the per-chunk
        // accumulators near 16 MiB.
        let per_spectrum = (n_out + rows.accumulator_len()) * std::mem::size_of::<f64>();
        let block = ((16 << 20) / per_spectrum).clamp(1, 64);
        for r0 in (0..n_rows).step_by(block) {
            let r1 = (r0 + block).min(n_rows);
            let block_signals: Vec<&[f64]> = (r0..r1)
                .map(|r| &signals[r * n_cols..(r + 1) * n_cols])
                .collect();
            let mut outs: Vec<&mut [f64]> =
                flat[r0 * n_out..r1 * n_out].chunks_mut(n_out).collect();
            rows.apply_block(&block_signals, &mut outs, n_cores)?;
        }
        return Ok(flat);
    }
    // Rows cost the same, so they split evenly across workers. Cores left
    // over when there are fewer rows than cores split each row instead.
    let workers = n_cores.max(1).min(n_rows);
    let inner = (n_cores / workers).max(1);
    if workers == 1 {
        for (row, dest) in flat.chunks_mut(n_out).enumerate() {
            apply_into(
                plan,
                &signals[row * n_cols..(row + 1) * n_cols],
                dest,
                inner,
                false,
            )?;
        }
        return Ok(flat);
    }
    let rows_per = n_rows.div_ceil(workers);
    let failure = Mutex::new(None);
    std::thread::scope(|scope| {
        for (group, part) in flat.chunks_mut(rows_per * n_out).enumerate() {
            let row0 = group * rows_per;
            let rows_here = part.len() / n_out;
            let failure = &failure;
            scope.spawn(move || {
                for row in 0..rows_here {
                    let sample = &signals[(row0 + row) * n_cols..(row0 + row + 1) * n_cols];
                    let dest = &mut part[row * n_out..(row + 1) * n_out];
                    if let Err(err) = apply_into(plan, sample, dest, inner, false) {
                        *failure.lock().expect("window error lock") = Some(err);
                    }
                }
            });
        }
    });
    if let Some(err) = failure.into_inner().expect("window error lock") {
        return Err(err);
    }
    Ok(flat)
}

/// A zeroed `n_rows * n_out` result, or `MemoryError` instead of an abort when
/// it does not fit. `n_out` does not depend on the input width, so this can be
/// far larger than anything the caller passed in.
fn zeroed_output(n_rows: usize, n_out: usize) -> Result<Vec<f64>, KernelError> {
    let too_large = |bytes: String| {
        KernelError::Memory(format!(
            "could not allocate {bytes} bytes for a result of {n_rows} rows and {n_out} output \
             frequencies; pass fewer rows per call or fewer output frequencies"
        ))
    };
    let len = n_rows
        .checked_mul(n_out)
        .ok_or_else(|| too_large("more than usize::MAX".to_string()))?;
    let mut flat = Vec::new();
    flat.try_reserve_exact(len)
        .map_err(|_| too_large(len.saturating_mul(std::mem::size_of::<f64>()).to_string()))?;
    flat.resize(len, 0.0);
    Ok(flat)
}

fn apply_into(
    plan: &Plan,
    signal: &[f64],
    out: &mut [f64],
    n_cores: usize,
    reverse: bool,
) -> Result<(), KernelError> {
    match &plan.kind {
        Kind::Convolution { kernel } => {
            let values = kernel.as_f64();
            convolve_into(signal, &values, out, n_cores)?;
        }
        Kind::Sparse {
            col0,
            start,
            values,
        } => {
            let rows = SparseRows {
                col0,
                start,
                values,
            };
            for_each_block(n_cores, out, 1, rows.grain(), reverse, &|part, first| {
                rows.dot_range(first, signal, part, reverse);
            });
        }
        Kind::Symmetric { values, scale } => {
            let rows = SymmetricRows { values, scale };
            rows.apply_block(&[signal], &mut [out], n_cores)?;
        }
    }
    Ok(())
}

/// Spectra multiplied together per pass over the stored weights: as many as
/// fit in about 8 MiB, so a block stays in the last-level cache while every
/// output row is applied to it.
fn spectra_per_block(n_cols: usize) -> usize {
    let fit = (8 << 20) / (n_cols.max(1) * std::mem::size_of::<f64>());
    (fit.clamp(4, 256) / 4) * 4
}

/// The stored weights of a `Kind::Sparse` plan.
struct SparseRows<'a> {
    col0: &'a [u32],
    start: &'a [u32],
    values: &'a Values,
}

impl SparseRows<'_> {
    /// First input column, and the stored range, of output row `i`.
    fn row(&self, i: usize) -> (usize, usize, usize) {
        (
            self.col0[i] as usize,
            self.start[i] as usize,
            self.start[i + 1] as usize,
        )
    }

    /// Rows per scheduling block: enough that a block holds about 64K weights.
    fn grain(&self) -> usize {
        let rows = self.col0.len().max(1);
        let average = (*self.start.last().unwrap_or(&0) as usize / rows).max(1);
        (65536 / average).clamp(4, 64)
    }

    /// `dest[k] = row first+k · signal`, four rows at a time so four weight
    /// streams are in flight; each value equals the one-row dot product.
    fn dot_range(&self, first: usize, signal: &[f64], dest: &mut [f64], reverse: bool) {
        match self.values {
            Values::F64(stored) => self.dot_range_typed(stored, first, signal, dest, reverse),
            Values::F32(stored) => self.dot_range_typed(stored, first, signal, dest, reverse),
        }
    }

    fn dot_range_typed<W: simd::Weight>(
        &self,
        stored: &[W],
        first: usize,
        signal: &[f64],
        dest: &mut [f64],
        reverse: bool,
    ) {
        let pair = |i: usize| {
            let (column, a, b) = self.row(i);
            (&stored[a..b], &signal[column..column + (b - a)])
        };
        let four = |i: usize, group: &mut [f64; 4]| {
            let rows: [(&[W], &[f64]); 4] = core::array::from_fn(|r| pair(i + r));
            *group = simd::dot_rows4(rows.map(|(w, _)| w), rows.map(|(_, s)| s));
        };
        let (groups, rest) = dest.as_chunks_mut::<4>();
        let full = 4 * groups.len();
        let tail = |dest: &mut [f64]| {
            for (k, value) in dest.iter_mut().enumerate() {
                let (w, s) = pair(first + full + k);
                *value = simd::dot(w, s);
            }
        };
        if reverse {
            tail(rest);
            for (g, group) in groups.iter_mut().enumerate().rev() {
                four(first + 4 * g, group);
            }
        } else {
            for (g, group) in groups.iter_mut().enumerate() {
                four(first + 4 * g, group);
            }
            tail(rest);
        }
    }

    /// `apply_many` for stored weights. Each output row is read once per block
    /// of spectra and multiplied with all of them, instead of once per
    /// spectrum, and every value equals what `dot` gives for that spectrum.
    fn apply_many(
        &self,
        signals: &[f64],
        n_cols: usize,
        n_cores: usize,
        flat: &mut [f64],
    ) -> Result<(), KernelError> {
        let n_out = self.col0.len();
        let n_rows = flat.len() / n_out;
        let block = spectra_per_block(n_cols).min(n_rows);
        // Output-major results for one block of spectra.
        let mut scratch = zeroed_output(n_out, block)?;
        for r0 in (0..n_rows).step_by(block) {
            let width = block.min(n_rows - r0);
            let part = &mut scratch[..n_out * width];
            for_each_block(
                n_cores,
                part,
                width,
                (64 / width).max(1),
                false,
                &|chunk, first| {
                    let mut refs: Vec<&[f64]> = Vec::with_capacity(width);
                    for (local, dest) in chunk.chunks_mut(width).enumerate() {
                        let (column, a, b) = self.row(first + local);
                        let len = b - a;
                        refs.clear();
                        refs.extend((r0..r0 + width).map(|r| {
                            let begin = r * n_cols + column;
                            &signals[begin..begin + len]
                        }));
                        match self.values {
                            Values::F64(stored) => simd::dot_batch(&stored[a..b], &refs, dest),
                            Values::F32(stored) => simd::dot_batch(&stored[a..b], &refs, dest),
                        }
                    }
                },
            );
            for (i, values) in part.chunks(width).enumerate() {
                for (r, &value) in values.iter().enumerate() {
                    flat[(r0 + r) * n_out + i] = value;
                }
            }
        }
        Ok(())
    }
}

fn pointwise(
    freq_in: &[f64],
    freq_out: &[f64],
    signal: &[f64],
    bandwidth: f64,
    method: Method,
    n_cores: usize,
) -> Result<Vec<f64>, KernelError> {
    let log_in: Vec<f64> = freq_in.iter().map(|f| f.log10()).collect();
    let mut out = vec![0.0; freq_out.len()];
    let failure = Mutex::new(None);
    for_each_chunk(n_cores, &mut out, &|part, offset| {
        for (local, dest) in part.iter_mut().enumerate() {
            let fc = freq_out[offset + local];
            let log_fc = fc.log10();
            let (lo, hi, approx) = row_window(&log_in, log_fc, bandwidth, method);
            match normalized_sum(&SampleWindow {
                signal,
                freq_in,
                log_in: &log_in,
                lo,
                hi,
                fc,
                log_fc,
                bandwidth,
                approx,
            }) {
                Ok(value) => *dest = value,
                Err(err) => {
                    *failure.lock().expect("window error lock") = Some(err);
                }
            }
        }
    });
    if let Some(err) = failure.into_inner().expect("window error lock") {
        return Err(err);
    }
    Ok(out)
}

struct SampleWindow<'a> {
    signal: &'a [f64],
    freq_in: &'a [f64],
    log_in: &'a [f64],
    lo: usize,
    hi: usize,
    fc: f64,
    log_fc: f64,
    bandwidth: f64,
    approx: bool,
}

fn zero_window_error(fc: f64) -> KernelError {
    invalid(format!(
        "the Konno-Ohmachi window at output frequency {fc} summed to zero"
    ))
}

fn normalized_sum(window: &SampleWindow<'_>) -> Result<f64, KernelError> {
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    for j in window.lo..window.hi {
        let weight = weight_at(
            window.freq_in[j],
            window.log_in[j],
            window.fc,
            window.log_fc,
            window.bandwidth,
            window.approx,
        );
        total += weight * window.signal[j];
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        Ok(total / weight_sum)
    } else {
        Err(zero_window_error(window.fc))
    }
}

/// Number of stored weights for `freq_out` against `freq_in`, before any of
/// them is allocated. `method="exact"` stores `n_in * n_out` of them.
fn count_weights(log_in: &[f64], freq_out: &[f64], bandwidth: f64, method: Method) -> u64 {
    freq_out
        .iter()
        .map(|&fc| {
            let (lo, hi, _) = row_window(log_in, fc.log10(), bandwidth, method);
            (hi - lo) as u64
        })
        // Saturate rather than wrap: anything above u32::MAX is rejected anyway.
        .fold(0u64, u64::saturating_add)
}

fn build_sparse(
    freq_in: &[f64],
    freq_out: &[f64],
    bandwidth: f64,
    method: Method,
    dtype: DType,
) -> Result<Kind, KernelError> {
    let log_in: Vec<f64> = freq_in.iter().map(|f| f.log10()).collect();
    let total = count_weights(&log_in, freq_out, bandwidth, method);
    if total > u64::from(u32::MAX) {
        return Err(invalid(format!(
            "the weights for {} input and {} output frequencies would hold {total} values, \
             more than the {} this class can index; use smooth(method=\"exact\"), which \
             stores no weights, or pass fewer output frequencies",
            freq_in.len(),
            freq_out.len(),
            u32::MAX
        )));
    }
    let total = total as usize;
    let mut values = Values::with_capacity(total, dtype).map_err(|bytes| {
        KernelError::Memory(format!(
            "could not allocate {bytes} bytes for the {total} Konno-Ohmachi weights of {} \
             input and {} output frequencies; use smooth(method=\"exact\"), which stores \
             no weights, pass fewer output frequencies, or use dtype=\"float32\"",
            freq_in.len(),
            freq_out.len()
        ))
    })?;
    let mut col0 = Vec::with_capacity(freq_out.len());
    let mut start = Vec::with_capacity(freq_out.len() + 1);
    let mut row: Vec<f64> = Vec::new();
    start.push(0);
    for &fc in freq_out {
        let log_fc = fc.log10();
        let (lo, hi, approx) = row_window(&log_in, log_fc, bandwidth, method);
        row.clear();
        let mut weight_sum = 0.0;
        for j in lo..hi {
            let weight = weight_at(freq_in[j], log_in[j], fc, log_fc, bandwidth, approx);
            row.push(weight);
            weight_sum += weight;
        }
        if weight_sum.is_nan() || weight_sum <= 0.0 {
            return Err(zero_window_error(fc));
        }
        values.extend_scaled(&row, 1.0 / weight_sum);
        col0.push(u32::try_from(lo).map_err(|_| invalid("frequency grid is too long"))?);
        // `values.len() <= total <= u32::MAX` by the check above.
        start.push(values.len() as u32);
    }
    Ok(Kind::Sparse {
        col0,
        start,
        values,
    })
}

/// Start of row `i`'s stored weights in a symmetric plan over `n` frequencies:
/// rows `0..i` hold `(n - 1) + (n - 2) + ... + (n - i)` weights.
fn symmetric_offset(n: usize, i: usize) -> usize {
    i * (2 * n - i - 1) / 2
}

fn build_symmetric(freq: &[f64], bandwidth: f64, dtype: DType) -> Result<Kind, KernelError> {
    let n = freq.len();
    let total = (n as u64) * (n as u64 - 1) / 2;
    if total > u64::from(u32::MAX) {
        return Err(invalid(format!(
            "the exact weights for {n} frequencies would hold {total} values, more than \
             the {} this class can store; use smooth(method=\"exact\"), which stores no \
             weights, or pass fewer frequencies",
            u32::MAX
        )));
    }
    let total = total as usize;
    let mut values = Values::with_capacity(total, dtype).map_err(|bytes| {
        KernelError::Memory(format!(
            "could not allocate {bytes} bytes for the {total} exact Konno-Ohmachi weights of \
             {n} frequencies; use smooth(method=\"exact\"), which stores no weights, pass \
             fewer frequencies, or use dtype=\"float32\""
        ))
    })?;
    let log_f: Vec<f64> = freq.iter().map(|f| f.log10()).collect();
    // Each row's window sum is its diagonal (1), the lower triangle that
    // earlier rows add to `lower`, and its own upper triangle.
    let mut lower = vec![0.0f64; n];
    let mut scale = Vec::with_capacity(n);
    let mut row: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        row.clear();
        let mut upper = 0.0;
        for j in i + 1..n {
            let weight = weight_at(freq[j], log_f[j], freq[i], log_f[i], bandwidth, false);
            row.push(weight);
            upper += weight;
            lower[j] += weight;
        }
        let sum = 1.0 + lower[i] + upper;
        if sum.is_nan() || sum <= 0.0 {
            return Err(zero_window_error(freq[i]));
        }
        scale.push(1.0 / sum);
        values.extend_scaled(&row, 1.0);
    }
    Ok(Kind::Symmetric { values, scale })
}

/// Store a short kernel in the requested precision. Kernel weights lie in
/// `[0, 1]`, so narrowing to float32 cannot overflow.
fn store_values(raw: Vec<f64>, dtype: DType) -> Values {
    match dtype {
        DType::F64 => Values::F64(raw),
        DType::F32 => Values::F32(raw.into_iter().map(|value| value as f32).collect()),
    }
}

/// Ask Linux to back a large weight buffer with transparent huge pages, as
/// NumPy does for its arrays: streaming through the weights then takes far
/// fewer TLB misses, and filling them far fewer page faults. The advice never
/// changes the contents. Elsewhere this does nothing.
fn advise_huge_pages<T>(buffer: &Vec<T>) {
    #[cfg(target_os = "linux")]
    {
        const HUGE: usize = 2 << 20;
        let bytes = buffer.capacity() * std::mem::size_of::<T>();
        if bytes < 2 * HUGE {
            return;
        }
        let start = buffer.as_ptr() as usize;
        let first = (start + HUGE - 1) & !(HUGE - 1);
        let last = (start + bytes) & !(HUGE - 1);
        if last > first {
            // SAFETY: the range lies inside the buffer's own allocation, and
            // MADV_HUGEPAGE only changes how the kernel backs those pages.
            // The return value is ignored: the advice is optional.
            unsafe {
                libc::madvise(
                    first as *mut libc::c_void,
                    last - first,
                    libc::MADV_HUGEPAGE,
                );
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = buffer;
}

impl Values {
    /// Reserve room for `total` weights, or report the byte count that failed.
    fn with_capacity(total: usize, dtype: DType) -> Result<Self, usize> {
        let bytes = total.saturating_mul(dtype.bytes());
        match dtype {
            DType::F64 => {
                let mut values = Vec::new();
                values.try_reserve_exact(total).map_err(|_| bytes)?;
                advise_huge_pages(&values);
                Ok(Values::F64(values))
            }
            DType::F32 => {
                let mut values = Vec::new();
                values.try_reserve_exact(total).map_err(|_| bytes)?;
                advise_huge_pages(&values);
                Ok(Values::F32(values))
            }
        }
    }

    /// Append `row * scale` in the stored precision.
    fn extend_scaled(&mut self, row: &[f64], scale: f64) {
        match self {
            Values::F64(values) => values.extend(row.iter().map(|weight| weight * scale)),
            Values::F32(values) => values.extend(row.iter().map(|weight| (weight * scale) as f32)),
        }
    }

    fn len(&self) -> usize {
        match self {
            Values::F64(values) => values.len(),
            Values::F32(values) => values.len(),
        }
    }

    fn as_f64(&self) -> Cow<'_, [f64]> {
        match self {
            Values::F64(values) => Cow::Borrowed(values),
            Values::F32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        }
    }
}

/// Half-width of the approx window in `x = b * log10(f / fc)`.
const X_CUTOFF: f64 = 3.0;
/// Keeps a sample that lands on the cutoff in exact arithmetic, whether `x` is
/// reached through log10 differences or through `lag * du`.
const X_TOL: f64 = 1e-9;

/// Indices `[lo, hi)` of `log_in` whose `|b * (log_in - log_fc)| <= X_CUTOFF + X_TOL`.
fn window_bounds(log_in: &[f64], log_fc: f64, bandwidth: f64) -> (usize, usize) {
    let edge = X_CUTOFF + X_TOL;
    let lo = log_in.partition_point(|&value| bandwidth * (value - log_fc) < -edge);
    let hi = log_in.partition_point(|&value| bandwidth * (value - log_fc) <= edge);
    (lo, hi)
}

/// Input range `[lo, hi)` and whether the Taylor sinc may be used.
///
/// An approximate window with no sample falls back to the full exact sum.
fn row_window(log_in: &[f64], log_fc: f64, bandwidth: f64, method: Method) -> (usize, usize, bool) {
    if method == Method::Approx {
        let (lo, hi) = window_bounds(log_in, log_fc, bandwidth);
        if hi > lo {
            return (lo, hi, true);
        }
    }
    (0, log_in.len(), false)
}

/// The window depends only on `f / fc`, so the center shortcut must not use an
/// absolute tolerance in hertz; `window_from_x` handles `x` near zero.
fn weight_at(freq: f64, log_freq: f64, fc: f64, log_fc: f64, bandwidth: f64, approx: bool) -> f64 {
    if freq == fc {
        return 1.0;
    }
    window_from_x(bandwidth * (log_freq - log_fc), approx)
}

fn window_from_x(x: f64, approx: bool) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let sinc = if approx { sinc_taylor(x) } else { x.sin() / x };
    let squared = sinc * sinc;
    squared * squared
}

/// `sin(x) / x` for `|x| <= 3`, from the Taylor series through `x^16`.
fn sinc_taylor(x: f64) -> f64 {
    let x2 = x * x;
    let mut acc = 1.0 / 355_687_428_096_000.0;
    acc = 1.0 / 1_307_674_368_000.0 - x2 * acc;
    acc = 1.0 / 6_227_020_800.0 - x2 * acc;
    acc = 1.0 / 39_916_800.0 - x2 * acc;
    acc = 1.0 / 362_880.0 - x2 * acc;
    acc = 1.0 / 5_040.0 - x2 * acc;
    acc = 1.0 / 120.0 - x2 * acc;
    acc = 1.0 / 6.0 - x2 * acc;
    1.0 - x2 * acc
}

fn log_spacing(freq: &[f64]) -> Option<f64> {
    if freq.len() < 3 {
        return None;
    }
    let mut first = None;
    for pair in freq.windows(2) {
        let step = pair[1].log10() - pair[0].log10();
        if !(step.is_finite() && step > 0.0) {
            return None;
        }
        match first {
            None => first = Some(step),
            Some(reference) => {
                if (step - reference).abs() > 1e-6 * reference {
                    return None;
                }
            }
        }
    }
    first
}

/// Approx window on a log grid with step `du`, as lags `-half..=half`.
///
/// Only lags up to `n_in - 1` can pair two of `n_in` samples, so `half` is
/// capped there. That keeps a tiny `bandwidth * du` from asking for an
/// unbounded kernel.
fn approx_kernel(bandwidth: f64, du: f64, n_in: usize) -> Result<Vec<f64>, KernelError> {
    let max_half = n_in.saturating_sub(1);
    let step = bandwidth * du;
    let half = if step == 0.0 {
        max_half
    } else if step.is_infinite() {
        0
    } else {
        let width = (X_CUTOFF + X_TOL) / step;
        if !width.is_finite() || width >= max_half as f64 {
            max_half
        } else {
            width.floor() as usize
        }
    };
    let len = half
        .checked_mul(2)
        .and_then(|taps| taps.checked_add(1))
        .ok_or_else(|| invalid("convolution kernel is too long"))?;
    let mut kernel = Vec::new();
    kernel.try_reserve_exact(len).map_err(|_| {
        KernelError::Memory(format!("could not allocate a {len}-tap convolution kernel"))
    })?;
    for index in 0..len {
        if index == half {
            kernel.push(1.0);
            continue;
        }
        let lag = index as isize - half as isize;
        kernel.push(window_from_x(bandwidth * (lag as f64) * du, true));
    }
    Ok(kernel)
}

/// Normalized convolution of `signal` with a symmetric `kernel` into `out`.
///
/// Workers split the outputs; each output sums its taps in increasing sample
/// order, so the result does not depend on `n_cores`.
fn convolve_into(
    signal: &[f64],
    kernel: &[f64],
    out: &mut [f64],
    n_cores: usize,
) -> Result<(), KernelError> {
    let n = signal.len();
    let half = kernel.len() / 2;
    let failure = Mutex::new(None);
    for_each_chunk(n_cores, out, &|part, offset| {
        for (local, dest) in part.iter_mut().enumerate() {
            let i = offset + local;
            // Taps k with 0 <= i + k - half < n.
            let k_lo = half.saturating_sub(i);
            let k_hi = kernel.len().min(n + half - i);
            let taps = &kernel[k_lo..k_hi];
            let first = i + k_lo - half;
            let samples = &signal[first..first + taps.len()];
            let mut total = 0.0;
            let mut weight_sum = 0.0;
            for (&weight, &value) in taps.iter().zip(samples) {
                total += weight * value;
                weight_sum += weight;
            }
            if weight_sum == 0.0 {
                *failure.lock().expect("window error lock") = Some(invalid(
                    "the Konno-Ohmachi convolution kernel summed to zero",
                ));
                continue;
            }
            *dest = total / weight_sum;
        }
    });
    if let Some(err) = failure.into_inner().expect("window error lock") {
        return Err(err);
    }
    Ok(())
}

/// Row chunks of a symmetric plan over `n` frequencies, fixed by `n` alone so
/// results do not depend on the worker count: chunk `c` covers rows
/// `bounds[c]..bounds[c + 1]`, holding about the same number of weights.
fn symmetric_bounds(n: usize) -> Vec<usize> {
    const CHUNKS: u64 = 32;
    // Row `i` costs its `n - 1 - i` stored weights plus its diagonal.
    let total = (n as u64) * (n as u64 + 1) / 2;
    let mut bounds = vec![0usize];
    let mut done = 0u64;
    for i in 0..n {
        done += (n - i) as u64;
        let next = bounds.len() as u64;
        if i + 1 < n && next < CHUNKS && done * CHUNKS >= next * total {
            bounds.push(i + 1);
        }
    }
    bounds.push(n);
    bounds
}

/// Split `rest` into consecutive pieces of the given lengths.
fn split_lengths(mut rest: &mut [f64], lengths: impl Iterator<Item = usize>) -> Vec<&mut [f64]> {
    let mut pieces = Vec::new();
    for len in lengths {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(len);
        pieces.push(head);
        rest = tail;
    }
    pieces
}

/// The stored weights of a `Kind::Symmetric` plan.
struct SymmetricRows<'a> {
    values: &'a Values,
    scale: &'a [f64],
}

impl SymmetricRows<'_> {
    /// Values each spectrum needs for the per-chunk lower-triangle sums: chunk
    /// `c` collects what its rows add to outputs `bounds[c] + 1..n`.
    fn accumulator_len(&self) -> usize {
        let n = self.scale.len();
        let bounds = symmetric_bounds(n);
        bounds[..bounds.len() - 1].iter().map(|&a| n - a - 1).sum()
    }

    /// Smooth every signal in `signals` into the matching `out`.
    ///
    /// Each stored row `i` is read once per call: in one pass it gives the
    /// upper-triangle dot product of output `i` and adds its weights, times
    /// `signal[i]`, to the lower-triangle sums of the outputs after `i`. Those
    /// sums are kept per chunk and added up in chunk order, so the result is
    /// the same for any `n_cores` and for any number of signals per call.
    fn apply_block(
        &self,
        signals: &[&[f64]],
        out: &mut [&mut [f64]],
        n_cores: usize,
    ) -> Result<(), KernelError> {
        match self.values {
            Values::F64(stored) => self.apply_typed(stored, signals, out, n_cores),
            Values::F32(stored) => self.apply_typed(stored, signals, out, n_cores),
        }
    }

    fn apply_typed<W: simd::Weight>(
        &self,
        stored: &[W],
        signals: &[&[f64]],
        out: &mut [&mut [f64]],
        n_cores: usize,
    ) -> Result<(), KernelError> {
        let n = self.scale.len();
        let count = signals.len();
        let bounds = symmetric_bounds(n);
        let chunks = bounds.len() - 1;
        let lengths: Vec<usize> = (0..chunks).map(|c| n - bounds[c] - 1).collect();
        let offsets: Vec<usize> = lengths
            .iter()
            .scan(0, |at, &len| {
                let here = *at;
                *at += len;
                Some(here)
            })
            .collect();
        let width = lengths.iter().sum::<usize>();
        // `upper[r][i]` is `signal[i]` plus output `i`'s upper-triangle dot
        // product; `lower[r]` holds every chunk's lower-triangle sums.
        let mut upper = zeroed_output(count, n)?;
        let mut lower = zeroed_output(count, width)?;
        {
            type Item<'b> = (usize, Vec<&'b mut [f64]>, Vec<&'b mut [f64]>);
            let mut items: Vec<Item<'_>> = (0..chunks)
                .map(|c| (c, Vec::with_capacity(count), Vec::with_capacity(count)))
                .collect();
            let mut upper_rest: &mut [f64] = &mut upper;
            let mut lower_rest: &mut [f64] = &mut lower;
            for _ in 0..count {
                let (row_upper, tail) = std::mem::take(&mut upper_rest).split_at_mut(n);
                upper_rest = tail;
                let (row_lower, tail) = std::mem::take(&mut lower_rest).split_at_mut(width);
                lower_rest = tail;
                let rows = split_lengths(row_upper, (0..chunks).map(|c| bounds[c + 1] - bounds[c]));
                let sums = split_lengths(row_lower, lengths.iter().copied());
                for (item, (rows, sums)) in items.iter_mut().zip(rows.into_iter().zip(sums)) {
                    item.1.push(rows);
                    item.2.push(sums);
                }
            }
            let process = |(c, mut parts, mut sums): Item<'_>| {
                let a = bounds[c];
                let mut dots = vec![0.0; count];
                let mut heads = vec![0.0; count];
                for i in a..bounds[c + 1] {
                    let len = n - 1 - i;
                    let at = symmetric_offset(n, i);
                    let tails: Vec<&[f64]> = signals.iter().map(|s| &s[i + 1..n]).collect();
                    for (head, s) in heads.iter_mut().zip(signals) {
                        *head = s[i];
                    }
                    let mut targets: Vec<&mut [f64]> = sums
                        .iter_mut()
                        .map(|z| &mut z[i - a..i - a + len])
                        .collect();
                    simd::dot_axpy_batch(
                        &stored[at..at + len],
                        &tails,
                        &mut targets,
                        &heads,
                        &mut dots,
                    );
                    for r in 0..count {
                        parts[r][i - a] = heads[r] + dots[r];
                    }
                }
            };
            let workers = n_cores.max(1).min(chunks);
            if workers <= 1 {
                items.into_iter().for_each(process);
            } else {
                let queue = Mutex::new(items.into_iter());
                std::thread::scope(|scope| {
                    for _ in 0..workers {
                        scope.spawn(|| loop {
                            let next = queue.lock().expect("chunk queue lock").next();
                            match next {
                                Some(item) => process(item),
                                None => return,
                            }
                        });
                    }
                });
            }
        }
        for r in 0..count {
            let row_upper = &upper[r * n..(r + 1) * n];
            let row_lower = &lower[r * width..(r + 1) * width];
            for (i, dest) in out[r].iter_mut().enumerate() {
                let mut total = row_upper[i];
                for c in 0..chunks {
                    let a = bounds[c];
                    if a >= i {
                        break;
                    }
                    total += row_lower[offsets[c] + i - a - 1];
                }
                *dest = total * self.scale[i];
            }
        }
        Ok(())
    }
}

fn for_each_chunk(n_cores: usize, out: &mut [f64], f: &(impl Fn(&mut [f64], usize) + Sync)) {
    for_each_block(n_cores, out, 1, 64, false, f);
}

/// Run `f(part, first)` over disjoint blocks of `out` on up to `n_cores`
/// workers, where `out` holds items of `width` values each and `first` is the
/// index of the block's first item. Blocks are handed out as workers free up,
/// so uneven work (the approx window widens with frequency on a linear grid)
/// still balances. Each item is computed by exactly one worker, in a fixed
/// order, so the result does not depend on the worker count.
fn for_each_block(
    n_cores: usize,
    out: &mut [f64],
    width: usize,
    grain: usize,
    reverse: bool,
    f: &(impl Fn(&mut [f64], usize) + Sync),
) {
    debug_assert!(width > 0 && out.len().is_multiple_of(width));
    let n = out.len() / width;
    let workers = n_cores.max(1).min(n.max(1));
    // About 32 blocks per worker, and at least `grain` items per block so the
    // lock around the block iterator stays negligible.
    let block = (n / (workers * 32)).clamp(grain.max(1), 4096.max(grain));
    // A short output has fewer blocks than workers; a thread per block is
    // enough, and one block runs on the calling thread.
    let workers = workers.min(n.div_ceil(block));
    if workers <= 1 && !reverse {
        f(out, 0);
        return;
    }
    // Blocks go out last-first when `reverse` is set.
    let order: Box<dyn Iterator<Item = (usize, &mut [f64])> + Send> = if reverse {
        Box::new(out.chunks_mut(block * width).enumerate().rev())
    } else {
        Box::new(out.chunks_mut(block * width).enumerate())
    };
    if workers <= 1 {
        for (index, part) in order {
            f(part, index * block);
        }
        return;
    }
    let blocks = Mutex::new(order);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let next = blocks.lock().expect("block iterator lock").next();
                match next {
                    Some((index, part)) => f(part, index * block),
                    None => return,
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_signal_is_unchanged() {
        let freq: Vec<f64> = (1..50).map(|i| i as f64).collect();
        let signal = vec![3.5; freq.len()];
        for method in [Method::Approx, Method::Exact] {
            let out = smooth_signal(&freq, &freq, &signal, 40.0, method, 2).unwrap();
            for value in out {
                assert!((value - 3.5).abs() < 1e-9, "{value}");
            }
        }
    }

    #[test]
    fn linear_grid_is_not_log_uniform() {
        let freq: Vec<f64> = (1..30).map(|i| i as f64 * 0.2).collect();
        assert!(log_spacing(&freq).is_none());
    }

    fn assert_log_grid_matches(freq: Vec<f64>, bandwidth: f64) {
        let signal: Vec<f64> = freq.iter().map(|f| (f.ln().sin()).powi(2) + 0.2).collect();
        let via_point = pointwise(&freq, &freq, &signal, bandwidth, Method::Approx, 1).unwrap();
        let via_smooth =
            smooth_signal(&freq, &freq, &signal, bandwidth, Method::Approx, 1).unwrap();
        let plan = build_plan(
            freq.clone(),
            freq.clone(),
            bandwidth,
            Method::Approx,
            DType::F64,
        )
        .unwrap();
        assert!(matches!(plan.kind, Kind::Convolution { .. }));
        let via_plan = apply_plan(&plan, &signal, 2).unwrap();
        for ((&point, &smooth), &cached) in via_point.iter().zip(&via_smooth).zip(&via_plan) {
            let scale = point.abs().max(1.0);
            assert!(
                (smooth - point).abs() <= 1e-8 * scale,
                "{smooth} {point} bandwidth={bandwidth}"
            );
            assert!(
                (cached - smooth).abs() <= 1e-12 * scale,
                "{cached} {smooth}"
            );
        }
    }

    #[test]
    fn log_grid_convolution_matches_pointwise() {
        let freq: Vec<f64> = (0..80)
            .map(|i| 10f64.powf(-1.0 + i as f64 * 0.05))
            .collect();
        assert_log_grid_matches(freq, 40.0);
        // 3 / (b * du) = 10 exactly, so the cutoff tap is the interesting one.
        let freq: Vec<f64> = (0..201).map(|i| 10f64.powf(i as f64 * 0.01)).collect();
        assert_log_grid_matches(freq, 30.0);
        // The uncapped half-width would be 150 taps on a 50-sample grid.
        let freq: Vec<f64> = (0..50).map(|i| 10f64.powf(i as f64 * 0.02)).collect();
        assert_log_grid_matches(freq, 1.0);
    }

    fn assert_bounded_kernel(kernel: &[f64], n_in: usize) {
        assert!(kernel.len() < 2 * n_in, "{} taps", kernel.len());
        assert_eq!(kernel.len() % 2, 1);
        assert_eq!(kernel[kernel.len() / 2], 1.0);
        assert!(kernel.iter().all(|weight| weight.is_finite()));
    }

    #[test]
    fn kernel_is_capped_by_the_grid_length() {
        let kernel = approx_kernel(1e-20, 1e-6, 5).unwrap();
        assert_bounded_kernel(&kernel, 5);
        assert_eq!(kernel.len(), 9);
        assert_eq!(approx_kernel(1e-20, 1e-6, 1).unwrap(), vec![1.0]);
        // 3 / (b * du) = 10, below the cap of 199.
        let kernel = approx_kernel(30.0, 0.01, 200).unwrap();
        assert_bounded_kernel(&kernel, 200);
        assert_eq!(kernel.len(), 21);
    }

    #[test]
    fn kernel_survives_step_underflow_and_overflow() {
        let smallest = f64::from_bits(1);
        assert_eq!(smallest * 0.25, 0.0);
        let kernel = approx_kernel(smallest, 0.25, 7).unwrap();
        assert_bounded_kernel(&kernel, 7);
        assert_eq!(kernel.len(), 13);

        // Finite step whose reciprocal overflows.
        let step = 1e-310;
        assert!(((X_CUTOFF + X_TOL) / step).is_infinite());
        let kernel = approx_kernel(step, 1.0, 4).unwrap();
        assert_bounded_kernel(&kernel, 4);
        assert_eq!(kernel.len(), 7);

        assert!((1e300 * 1e10_f64).is_infinite());
        assert_eq!(approx_kernel(1e300, 1e10, 3).unwrap(), vec![1.0]);
    }

    #[test]
    fn tiny_bandwidth_on_a_log_grid_averages_every_sample() {
        let freq = vec![1.0, 10.0, 100.0];
        let signal = vec![1.0, 2.0, 3.0];
        let direct = smooth_signal(&freq, &freq, &signal, 1e-20, Method::Approx, 1).unwrap();
        let exact = smooth_signal(&freq, &freq, &signal, 1e-20, Method::Exact, 1).unwrap();
        for (value, reference) in direct.iter().zip(&exact) {
            assert!((value - 2.0).abs() < 1e-12, "{value}");
            assert!((value - reference).abs() < 1e-12, "{value} {reference}");
        }
        let plan = build_plan(freq.clone(), freq, 1e-20, Method::Approx, DType::F32).unwrap();
        let cached = apply_plan(&plan, &signal, 1).unwrap();
        assert!(cached.iter().all(|value| (value - 2.0).abs() < 1e-6));
    }

    #[test]
    fn taylor_matches_libm_inside_cutoff() {
        for i in -30..=30 {
            let x = i as f64 * 0.1;
            if x.abs() < 1e-12 {
                continue;
            }
            let reference = x.sin() / x;
            let approx = sinc_taylor(x);
            assert!(
                (approx - reference).abs() < 1e-8,
                "x={x} taylor={approx} libm={reference}"
            );
        }
    }

    #[test]
    fn empty_approx_window_matches_exact() {
        let freq: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let signal: Vec<f64> = (0..10).map(|i| 0.2 + i as f64 * 0.15).collect();
        let freq_out = vec![1.5, 2.0, 5.5];
        let approx = smooth_signal(&freq, &freq_out, &signal, 100.0, Method::Approx, 1).unwrap();
        let exact = smooth_signal(&freq, &freq_out, &signal, 100.0, Method::Exact, 1).unwrap();
        assert!(approx.iter().all(|value| value.is_finite()));
        for index in [0, 2] {
            let scale = exact[index].abs().max(1.0);
            assert!((approx[index] - exact[index]).abs() <= 1e-12 * scale);
        }
    }

    #[test]
    fn log_grid_convolution_is_identical_for_any_worker_count() {
        let freq: Vec<f64> = (0..301).map(|i| 10f64.powf(i as f64 / 300.0)).collect();
        let signal: Vec<f64> = freq.iter().map(|f| (f.ln() * 3.0).sin() + 1.5).collect();
        // 0.5 spans the whole grid; 40 is a short kernel.
        for bandwidth in [0.5, 7.0, 40.0] {
            let serial =
                smooth_signal(&freq, &freq, &signal, bandwidth, Method::Approx, 1).unwrap();
            for dtype in [DType::F64, DType::F32] {
                let plan = build_plan(freq.clone(), freq.clone(), bandwidth, Method::Approx, dtype)
                    .unwrap();
                assert!(matches!(plan.kind, Kind::Convolution { .. }));
                let cached = apply_plan(&plan, &signal, 1).unwrap();
                for n_cores in [2, 3, 7, 301, 1000] {
                    let direct =
                        smooth_signal(&freq, &freq, &signal, bandwidth, Method::Approx, n_cores)
                            .unwrap();
                    assert_eq!(direct, serial, "bandwidth={bandwidth} n_cores={n_cores}");
                    assert_eq!(apply_plan(&plan, &signal, n_cores).unwrap(), cached);
                }
            }
        }
    }

    #[test]
    fn center_weight_does_not_depend_on_the_frequency_scale() {
        for approx in [false, true] {
            let reference = weight_at(2.0, 2f64.log10(), 1.0, 0.0, 2.0, approx);
            assert!(reference < 1.0);
            for scale in [1e-20, 1e-300, 1e20] {
                let (freq, fc) = (2.0 * scale, scale);
                let weight = weight_at(freq, freq.log10(), fc, fc.log10(), 2.0, approx);
                assert!(
                    (weight - reference).abs() <= 1e-12,
                    "scale={scale} {weight} {reference}"
                );
                assert_eq!(weight_at(fc, fc.log10(), fc, fc.log10(), 2.0, approx), 1.0);
            }
        }
    }

    #[test]
    fn exact_weights_are_counted_before_they_are_allocated() {
        let freq_in: Vec<f64> = (1..=5).map(|i| i as f64).collect();
        let freq_out = vec![1.5, 2.5, 4.0];
        let log_in: Vec<f64> = freq_in.iter().map(|f| f.log10()).collect();
        assert_eq!(count_weights(&log_in, &freq_out, 40.0, Method::Exact), 15);
        let approx = count_weights(&log_in, &freq_out, 40.0, Method::Approx);
        assert!(approx < 15, "{approx}");
        let plan = build_plan(freq_in, freq_out, 40.0, Method::Approx, DType::F64).unwrap();
        match plan.kind {
            Kind::Sparse { values, start, .. } => {
                assert_eq!(values.len() as u64, approx);
                assert_eq!(*start.last().unwrap() as u64, approx);
            }
            _ => panic!("expected sparse weights"),
        }
    }

    #[test]
    fn rejects_weight_matrix_above_u32() {
        // Different grids: 70000 * 69999 weights. Same grid, stored once per
        // pair: 100000 * 99999 / 2. Both exceed u32::MAX and must fail before
        // any weight is allocated.
        let freq: Vec<f64> = (0..70_000).map(|i| 1.0 + i as f64 * 1e-5).collect();
        let out = freq[1..].to_vec();
        let same: Vec<f64> = (0..100_000).map(|i| 1.0 + i as f64 * 1e-5).collect();
        for err in [
            build_plan(freq, out, 40.0, Method::Exact, DType::F64).unwrap_err(),
            build_plan(same.clone(), same, 40.0, Method::Exact, DType::F64).unwrap_err(),
        ] {
            assert!(matches!(err, KernelError::Invalid(_)), "{err:?}");
            assert!(
                err.to_string().contains("smooth(method=\"exact\")"),
                "{err:?}"
            );
        }
    }

    fn uneven_grid(n: usize) -> Vec<f64> {
        let mut f = 0.2;
        (0..n)
            .map(|i| {
                f += 0.05 + 0.03 * ((i * 7919) % 13) as f64 / 13.0;
                f
            })
            .collect()
    }

    #[test]
    fn alternating_sweeps_give_the_same_result() {
        let freq = uneven_grid(203);
        let out: Vec<f64> = freq.iter().step_by(3).map(|f| f * 1.001).collect();
        let signal: Vec<f64> = freq.iter().map(|f| f.cos() + 2.0).collect();
        for method in [Method::Approx, Method::Exact] {
            let plan = build_plan(freq.clone(), out.clone(), 15.0, method, DType::F64).unwrap();
            assert!(matches!(plan.kind, Kind::Sparse { .. }));
            let first = apply_plan(&plan, &signal, 1).unwrap();
            for n_cores in [1, 1, 3, 3, 2] {
                assert_eq!(
                    apply_plan(&plan, &signal, n_cores).unwrap(),
                    first,
                    "{method:?}"
                );
            }
        }
    }

    #[test]
    fn symmetric_bounds_cover_every_row_once() {
        for n in [1usize, 2, 3, 31, 32, 33, 100, 15000] {
            let bounds = symmetric_bounds(n);
            assert_eq!(bounds[0], 0);
            assert_eq!(*bounds.last().unwrap(), n);
            assert!(
                bounds.windows(2).all(|pair| pair[0] < pair[1]),
                "n={n} {bounds:?}"
            );
            assert!(bounds.len() <= 33, "n={n}");
        }
        assert_eq!(symmetric_offset(5, 0), 0);
        assert_eq!(symmetric_offset(5, 1), 4);
        assert_eq!(symmetric_offset(5, 4), 10);
        assert_eq!(symmetric_offset(5, 5), 10);
    }

    #[test]
    fn symmetric_plan_matches_the_dense_exact_weights() {
        for n in [1usize, 2, 17, 137] {
            let freq = uneven_grid(n);
            let signal: Vec<f64> = freq.iter().map(|f| (f * 1.7).sin() + 1.3).collect();
            let dense = build_sparse(&freq, &freq, 9.0, Method::Exact, DType::F64).unwrap();
            let reference = match &dense {
                Kind::Sparse {
                    col0,
                    start,
                    values,
                } => {
                    let rows = SparseRows {
                        col0,
                        start,
                        values,
                    };
                    {
                        let mut reference = vec![0.0; n];
                        rows.dot_range(0, &signal, &mut reference, false);
                        reference
                    }
                }
                _ => unreachable!(),
            };
            for (dtype, tolerance) in [(DType::F64, 1e-13), (DType::F32, 2e-6)] {
                let plan =
                    build_plan(freq.clone(), freq.clone(), 9.0, Method::Exact, dtype).unwrap();
                assert!(matches!(plan.kind, Kind::Symmetric { .. }));
                let got = apply_plan(&plan, &signal, 1).unwrap();
                for (g, r) in got.iter().zip(&reference) {
                    assert!(
                        (g - r).abs() <= tolerance * r.abs(),
                        "n={n} {dtype:?} {g} {r}"
                    );
                }
            }
        }
        let one = build_plan(vec![3.0], vec![3.0], 40.0, Method::Exact, DType::F64).unwrap();
        assert_eq!(apply_plan(&one, &[2.5], 1).unwrap(), vec![2.5]);
    }

    #[test]
    fn symmetric_plan_is_identical_for_any_worker_count_and_batch() {
        let freq = uneven_grid(613);
        let n = freq.len();
        let plan = build_plan(freq.clone(), freq.clone(), 12.0, Method::Exact, DType::F64).unwrap();
        let signals: Vec<f64> = (0..5 * n)
            .map(|k| ((k * 31 % 977) as f64).sqrt() + 0.1)
            .collect();
        let serial: Vec<Vec<f64>> = (0..5)
            .map(|r| apply_plan(&plan, &signals[r * n..(r + 1) * n], 1).unwrap())
            .collect();
        for n_cores in [2, 3, 7, 64] {
            assert_eq!(
                apply_plan(&plan, &signals[..n], n_cores).unwrap(),
                serial[0]
            );
            let batch = apply_many(&plan, &signals, 5, n, n_cores).unwrap();
            for r in 0..5 {
                assert_eq!(
                    &batch[r * n..(r + 1) * n],
                    &serial[r][..],
                    "n_cores={n_cores} r={r}"
                );
            }
        }
    }

    #[test]
    fn float32_weights_are_the_narrowed_float64_weights() {
        let freq: Vec<f64> = (1..40).map(|i| i as f64 * 0.3).collect();
        let wide =
            build_plan(freq.clone(), freq.clone(), 20.0, Method::Approx, DType::F64).unwrap();
        let narrow = build_plan(freq.clone(), freq, 20.0, Method::Approx, DType::F32).unwrap();
        assert_eq!(narrow.dtype, DType::F32);
        match (&wide.kind, &narrow.kind) {
            (
                Kind::Sparse {
                    values: Values::F64(wide),
                    ..
                },
                Kind::Sparse {
                    values: Values::F32(narrow),
                    ..
                },
            ) => {
                assert_eq!(wide.len(), narrow.len());
                for (&w, &n) in wide.iter().zip(narrow) {
                    assert_eq!(w as f32, n);
                }
            }
            _ => panic!("expected sparse weights"),
        }
    }

    #[test]
    fn for_each_chunk_visits_every_index_once() {
        for n in [0usize, 1, 63, 64, 65, 1000, 12345] {
            for workers in [1usize, 2, 3, 8] {
                let mut out = vec![-1.0; n];
                for_each_chunk(workers, &mut out, &|part, offset| {
                    for (j, dest) in part.iter_mut().enumerate() {
                        *dest = (offset + j) as f64;
                    }
                });
                for (i, &value) in out.iter().enumerate() {
                    assert_eq!(value, i as f64, "n={n} workers={workers}");
                }
            }
        }
    }

    #[test]
    fn apply_many_uses_spare_cores_on_a_single_row() {
        let freq: Vec<f64> = (1..=400).map(|i| i as f64 * 0.05).collect();
        let signal: Vec<f64> = freq.iter().map(|f| f.sin().powi(2) + 0.1).collect();
        let plan = build_plan(freq.clone(), freq, 20.0, Method::Approx, DType::F64).unwrap();
        let serial = apply_plan(&plan, &signal, 1).unwrap();
        let n = signal.len();
        for n_cores in [1, 2, 3, 8] {
            assert_eq!(apply_many(&plan, &signal, 1, n, n_cores).unwrap(), serial);
            let two = [signal.clone(), signal.iter().rev().copied().collect()].concat();
            let batch = apply_many(&plan, &two, 2, n, n_cores).unwrap();
            assert_eq!(&batch[..n], &serial[..]);
            assert_eq!(&batch[n..], &apply_plan(&plan, &two[n..], 1).unwrap()[..]);
        }
    }

    #[test]
    fn oversized_results_are_memory_errors() {
        assert!(matches!(
            zeroed_output(usize::MAX, 2),
            Err(KernelError::Memory(_))
        ));
        // 2^60 values is more than any allocator will hand out.
        assert!(matches!(
            zeroed_output(1 << 40, 1 << 20),
            Err(KernelError::Memory(_))
        ));
        assert_eq!(zeroed_output(0, 7).unwrap(), Vec::<f64>::new());
        assert_eq!(zeroed_output(2, 3).unwrap(), vec![0.0; 6]);
    }

    #[test]
    fn weight_count_saturates_instead_of_wrapping() {
        let log_in = vec![0.0; 4];
        let freq_out = vec![1.0; 3];
        assert_eq!(count_weights(&log_in, &freq_out, 40.0, Method::Exact), 12);
        assert_eq!(
            [u64::MAX, 5].into_iter().fold(0u64, u64::saturating_add),
            u64::MAX
        );
    }

    #[test]
    fn rejects_more_cores_than_the_machine_has() {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        assert_eq!(resolve_cores(None).unwrap(), available);
        assert_eq!(resolve_cores(Some(1)).unwrap(), 1);
        let err = resolve_cores(Some(available as isize + 1)).unwrap_err();
        assert!(err.to_string().contains("n_cores"), "{err:?}");
        assert!(err
            .to_string()
            .contains(&format!("the {available} logical CPUs")));
        let huge = bad_core_count("1180591620717411303424", true).to_string();
        assert!(
            huge.starts_with("n_cores (1180591620717411303424) exceeds"),
            "{huge}"
        );
        let tiny = bad_core_count("-5", false).to_string();
        assert_eq!(tiny, "n_cores must be >= 1 or None, got -5");
        assert!(resolve_cores(Some(0))
            .unwrap_err()
            .to_string()
            .contains("n_cores"));
    }

    #[test]
    fn rejects_non_positive_frequency() {
        let err = validate_freq(&[0.1, 0.0, 0.3], "freq_in").unwrap_err();
        assert!(err.to_string().contains("positive"));
    }
}
