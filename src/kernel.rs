//! Konno-Ohmachi smoothing kernels.
//!
//! `method="approx"` keeps frequencies inside `10^(±3/b)`. `method="exact"`
//! sums the published window over every positive frequency.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Approx,
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    F32,
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
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Plan {
    pub freq_in: Vec<f64>,
    pub freq_out: Vec<f64>,
    pub bandwidth: f64,
    pub method: Method,
    pub kind: Kind,
}

pub fn parse_method(method: &str) -> Result<Method, String> {
    match method {
        "approx" => Ok(Method::Approx),
        "exact" => Ok(Method::Exact),
        _ => Err("method must be \"approx\" or \"exact\"".to_string()),
    }
}

pub fn parse_dtype(dtype: &str) -> Result<DType, String> {
    match dtype {
        "float64" => Ok(DType::F64),
        "float32" => Ok(DType::F32),
        _ => Err("dtype must be \"float64\" or \"float32\"".to_string()),
    }
}

pub fn resolve_cores(n_cores: Option<isize>) -> Result<usize, String> {
    match n_cores {
        None => Ok(std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)),
        Some(n) if n >= 1 => Ok(n as usize),
        Some(n) => Err(format!("n_cores must be >= 1 or None, got {n}")),
    }
}

pub fn validate_freq(freq: &[f64], name: &str) -> Result<(), String> {
    if freq.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    for (i, &value) in freq.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{name}[{i}] must be finite"));
        }
        if value <= 0.0 {
            return Err(format!("{name}[{i}] must be positive, got {value}"));
        }
        if i > 0 && value <= freq[i - 1] {
            return Err(format!("{name} must be strictly increasing"));
        }
    }
    Ok(())
}

pub fn validate_bandwidth(bandwidth: f64) -> Result<(), String> {
    if bandwidth.is_finite() && bandwidth > 0.0 {
        Ok(())
    } else {
        Err("bandwidth must be a positive finite number".to_string())
    }
}

pub fn same_grid(freq_in: &[f64], freq_out: &[f64]) -> bool {
    freq_in.len() == freq_out.len() && freq_in.iter().zip(freq_out).all(|(a, b)| a == b)
}

pub fn build_plan(
    freq_in: Vec<f64>,
    freq_out: Vec<f64>,
    bandwidth: f64,
    method: Method,
    dtype: DType,
) -> Result<Plan, String> {
    validate_freq(&freq_in, "freq_in")?;
    validate_freq(&freq_out, "freq_out")?;
    validate_bandwidth(bandwidth)?;
    let kind = if method == Method::Approx && same_grid(&freq_in, &freq_out) {
        if let Some(du) = log_spacing(&freq_in) {
            Kind::Convolution {
                kernel: store_values(&approx_kernel(bandwidth, du), dtype)?,
            }
        } else {
            build_sparse(&freq_in, &freq_out, bandwidth, method, dtype)?
        }
    } else {
        build_sparse(&freq_in, &freq_out, bandwidth, method, dtype)?
    };
    Ok(Plan {
        freq_in,
        freq_out,
        bandwidth,
        method,
        kind,
    })
}

pub fn smooth_signal(
    freq_in: &[f64],
    freq_out: &[f64],
    signal: &[f64],
    bandwidth: f64,
    method: Method,
    n_cores: usize,
) -> Result<Vec<f64>, String> {
    validate_freq(freq_in, "freq_in")?;
    validate_freq(freq_out, "freq_out")?;
    validate_bandwidth(bandwidth)?;
    if signal.len() != freq_in.len() {
        return Err(format!(
            "signal length ({}) must match freq_in ({})",
            signal.len(),
            freq_in.len()
        ));
    }
    if method == Method::Approx && same_grid(freq_in, freq_out) {
        if let Some(du) = log_spacing(freq_in) {
            return Ok(convolve(signal, &approx_kernel(bandwidth, du)));
        }
    }
    Ok(pointwise(
        freq_in, freq_out, signal, bandwidth, method, n_cores,
    ))
}

pub fn apply_plan(plan: &Plan, signal: &[f64], n_cores: usize) -> Result<Vec<f64>, String> {
    if signal.len() != plan.freq_in.len() {
        return Err(format!(
            "signal length ({}) must match freq_in ({})",
            signal.len(),
            plan.freq_in.len()
        ));
    }
    let mut out = vec![0.0; plan.freq_out.len()];
    apply_into(plan, signal, &mut out, n_cores);
    Ok(out)
}

pub fn apply_many(
    plan: &Plan,
    signals: &[f64],
    n_rows: usize,
    n_cols: usize,
    n_cores: usize,
) -> Result<Vec<f64>, String> {
    if n_cols != plan.freq_in.len() {
        return Err(format!(
            "signals has {} columns but freq_in has length {}",
            n_cols,
            plan.freq_in.len()
        ));
    }
    if signals.len() != n_rows * n_cols {
        return Err("signals length does not match its shape".to_string());
    }
    let n_out = plan.freq_out.len();
    let mut flat = vec![0.0; n_rows * n_out];
    if n_rows == 0 {
        return Ok(flat);
    }
    let workers = n_cores.max(1).min(n_rows);
    let rows_per = (n_rows + workers - 1) / workers;
    std::thread::scope(|scope| {
        for (group, part) in flat.chunks_mut(rows_per * n_out).enumerate() {
            let row0 = group * rows_per;
            let rows_here = part.len() / n_out;
            scope.spawn(move || {
                for row in 0..rows_here {
                    let sample = &signals[(row0 + row) * n_cols..(row0 + row + 1) * n_cols];
                    let dest = &mut part[row * n_out..(row + 1) * n_out];
                    apply_into(plan, sample, dest, 1);
                }
            });
        }
    });
    Ok(flat)
}

fn apply_into(plan: &Plan, signal: &[f64], out: &mut [f64], n_cores: usize) {
    match &plan.kind {
        Kind::Convolution { kernel } => {
            let values = kernel.as_f64();
            let smoothed = convolve(signal, &values);
            out.copy_from_slice(&smoothed);
        }
        Kind::Sparse {
            col0,
            start,
            values,
        } => {
            for_each_chunk(n_cores, out, &|part, offset| {
                for (local, dest) in part.iter_mut().enumerate() {
                    let i = offset + local;
                    let a = start[i] as usize;
                    let b = start[i + 1] as usize;
                    if a == b {
                        *dest = f64::NAN;
                        continue;
                    }
                    let column = col0[i] as usize;
                    let mut total = 0.0;
                    match values {
                        Values::F64(stored) => {
                            for (k, &weight) in stored[a..b].iter().enumerate() {
                                total += weight * signal[column + k];
                            }
                        }
                        Values::F32(stored) => {
                            for (k, &weight) in stored[a..b].iter().enumerate() {
                                total += f64::from(weight) * signal[column + k];
                            }
                        }
                    }
                    *dest = total;
                }
            });
        }
    }
}

fn pointwise(
    freq_in: &[f64],
    freq_out: &[f64],
    signal: &[f64],
    bandwidth: f64,
    method: Method,
    n_cores: usize,
) -> Vec<f64> {
    let log_in: Vec<f64> = freq_in.iter().map(|f| f.log10()).collect();
    let max_ratio = 10f64.powf(3.0 / bandwidth);
    let min_ratio = 1.0 / max_ratio;
    let mut out = vec![0.0; freq_out.len()];
    for_each_chunk(n_cores, &mut out, &|part, offset| {
        for (local, dest) in part.iter_mut().enumerate() {
            let fc = freq_out[offset + local];
            let log_fc = fc.log10();
            let (lo, hi) = if method == Method::Approx {
                window_bounds(freq_in, fc, min_ratio, max_ratio)
            } else {
                (0, freq_in.len())
            };
            *dest = normalized_sum(signal, freq_in, &log_in, lo, hi, fc, log_fc, bandwidth, method);
        }
    });
    out
}

fn normalized_sum(
    signal: &[f64],
    freq_in: &[f64],
    log_in: &[f64],
    lo: usize,
    hi: usize,
    fc: f64,
    log_fc: f64,
    bandwidth: f64,
    method: Method,
) -> f64 {
    let approx = method == Method::Approx;
    let mut total = 0.0;
    let mut weight_sum = 0.0;
    for j in lo..hi {
        let weight = weight_at(freq_in[j], log_in[j], fc, log_fc, bandwidth, approx);
        total += weight * signal[j];
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        total / weight_sum
    } else {
        f64::NAN
    }
}

fn build_sparse(
    freq_in: &[f64],
    freq_out: &[f64],
    bandwidth: f64,
    method: Method,
    dtype: DType,
) -> Result<Kind, String> {
    let log_in: Vec<f64> = freq_in.iter().map(|f| f.log10()).collect();
    let max_ratio = 10f64.powf(3.0 / bandwidth);
    let min_ratio = 1.0 / max_ratio;
    let mut col0 = Vec::with_capacity(freq_out.len());
    let mut start = Vec::with_capacity(freq_out.len() + 1);
    let mut raw: Vec<f64> = Vec::new();
    start.push(0);
    for &fc in freq_out {
        let log_fc = fc.log10();
        let (lo, hi) = if method == Method::Approx {
            window_bounds(freq_in, fc, min_ratio, max_ratio)
        } else {
            (0, freq_in.len())
        };
        let row_at = raw.len();
        let mut weight_sum = 0.0;
        for j in lo..hi {
            let weight = weight_at(freq_in[j], log_in[j], fc, log_fc, bandwidth, method == Method::Approx);
            raw.push(weight);
            weight_sum += weight;
        }
        if hi == lo || weight_sum == 0.0 {
            raw.truncate(row_at);
            col0.push(0);
        } else {
            let scale = 1.0 / weight_sum;
            for weight in &mut raw[row_at..] {
                *weight *= scale;
            }
            col0.push(u32::try_from(lo).map_err(|_| "frequency grid is too long".to_string())?);
        }
        let end = u32::try_from(raw.len()).map_err(|_| "precomputed weights are too large".to_string())?;
        start.push(end);
    }
    Ok(Kind::Sparse {
        col0,
        start,
        values: store_values(&raw, dtype)?,
    })
}

fn store_values(raw: &[f64], dtype: DType) -> Result<Values, String> {
    match dtype {
        DType::F64 => Ok(Values::F64(raw.to_vec())),
        DType::F32 => Ok(Values::F32(
            raw.iter()
                .map(|value| {
                    let narrow = *value as f32;
                    if value.is_finite() && !narrow.is_finite() {
                        Err("weight does not fit in float32".to_string())
                    } else {
                        Ok(narrow)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?,
        )),
    }
}

impl Values {
    fn as_f64(&self) -> Vec<f64> {
        match self {
            Values::F64(values) => values.clone(),
            Values::F32(values) => values.iter().map(|value| f64::from(*value)).collect(),
        }
    }
}

fn window_bounds(freq_in: &[f64], fc: f64, min_ratio: f64, max_ratio: f64) -> (usize, usize) {
    let lo = freq_in.partition_point(|value| *value < fc * min_ratio);
    let hi = freq_in.partition_point(|value| *value <= fc * max_ratio);
    (lo, hi)
}

fn weight_at(
    freq: f64,
    log_freq: f64,
    fc: f64,
    log_fc: f64,
    bandwidth: f64,
    approx: bool,
) -> f64 {
    if (freq - fc).abs() <= f64::EPSILON * fc.max(1.0) {
        return 1.0;
    }
    window_from_x(bandwidth * (log_freq - log_fc), approx)
}

fn window_from_x(x: f64, approx: bool) -> f64 {
    if x.abs() < 1e-12 {
        return 1.0;
    }
    let sinc = if approx {
        sinc_taylor(x)
    } else {
        x.sin() / x
    };
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

fn approx_kernel(bandwidth: f64, du: f64) -> Vec<f64> {
    let half = ((3.0 / bandwidth) / du + 1e-9).floor() as usize;
    let mut kernel = Vec::with_capacity(2 * half + 1);
    for index in 0..=(2 * half) {
        let lag = index as isize - half as isize;
        kernel.push(window_from_x(bandwidth * (lag as f64) * du, true));
    }
    kernel
}

fn convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut out = vec![0.0; signal.len()];
    for (i, dest) in out.iter_mut().enumerate() {
        let mut total = 0.0;
        let mut weight_sum = 0.0;
        for (k, &weight) in kernel.iter().enumerate() {
            let index = i as isize + k as isize - half as isize;
            if index >= 0 && (index as usize) < signal.len() {
                total += weight * signal[index as usize];
                weight_sum += weight;
            }
        }
        *dest = if weight_sum > 0.0 {
            total / weight_sum
        } else {
            f64::NAN
        };
    }
    out
}

fn for_each_chunk(n_cores: usize, out: &mut [f64], f: &(impl Fn(&mut [f64], usize) + Sync)) {
    let n = out.len();
    let workers = n_cores.max(1).min(n.max(1));
    if workers <= 1 || n == 0 {
        f(out, 0);
        return;
    }
    let chunk = (n + workers - 1) / workers;
    std::thread::scope(|scope| {
        let mut offset = 0;
        for part in out.chunks_mut(chunk) {
            let start = offset;
            offset += part.len();
            scope.spawn(move || f(part, start));
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

    #[test]
    fn log_grid_convolution_matches_pointwise() {
        let freq: Vec<f64> = (0..80).map(|i| 10f64.powf(-1.0 + i as f64 * 0.05)).collect();
        let signal: Vec<f64> = freq.iter().map(|f| (f.ln().sin()).powi(2) + 0.2).collect();
        let via_point = pointwise(&freq, &freq, &signal, 40.0, Method::Approx, 1);
        let via_smooth = smooth_signal(&freq, &freq, &signal, 40.0, Method::Approx, 1).unwrap();
        let plan = build_plan(freq.clone(), freq.clone(), 40.0, Method::Approx, DType::F64).unwrap();
        assert!(matches!(plan.kind, Kind::Convolution { .. }));
        let via_plan = apply_plan(&plan, &signal, 2).unwrap();
        for ((&point, &smooth), &cached) in via_point.iter().zip(&via_smooth).zip(&via_plan) {
            let scale = point.abs().max(1.0);
            assert!((smooth - point).abs() <= 1e-8 * scale, "{smooth} {point}");
            assert!((cached - smooth).abs() <= 1e-12 * scale, "{cached} {smooth}");
        }
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
    fn rejects_non_positive_frequency() {
        let err = validate_freq(&[0.1, 0.0, 0.3], "freq_in").unwrap_err();
        assert!(err.contains("positive"));
    }
}
