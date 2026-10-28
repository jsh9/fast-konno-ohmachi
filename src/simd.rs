//! Dot products with a fixed summation order.
//!
//! Element `k` of a dot product is added to lane `k % LANES`. The lanes are
//! then reduced pairwise in a fixed order, and the elements past the last full
//! group of `LANES` are added one by one. That order does not depend on the
//! SIMD width, so the SSE2, AVX2, AVX-512, NEON and scalar loops below, and a
//! batch computed several spectra or rows at a time, all give bit-identical
//! results. Multiplication and addition stay separate (no fused multiply-add)
//! for the same reason.
//!
//! Sixteen lanes keep enough independent additions in flight to stream weights
//! at memory bandwidth instead of waiting on one accumulator.

pub const LANES: usize = 16;

#[inline(always)]
fn reduce(mut lanes: [f64; LANES]) -> f64 {
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
    }
    lanes[0]
}

/// Generates, for one instruction set and one weight type, the three kernels:
///
/// - `shared`: `R` dot products of one weight row against `R` signals, sharing
///   each load of the weights;
/// - `pairs`: `R` dot products of `R` weight rows against `R` signals, run in
///   lockstep so `R` memory streams are in flight;
/// - `dot_axpy`: like `shared`, and in the same pass `z[r][k] += w[k] * a[r]`.
///
/// `$per` is the number of f64 values in one vector.
macro_rules! kernels {
    (
        $(#[$attr:meta])*
        $vec:ty, $per:expr,
        zero: $zero:expr,
        splat: $splat:expr,
        load: $load:expr,
        load_f32: $load_f32:expr,
        add: $add:expr,
        mul: $mul:expr,
        store: $store:expr $(,)?
    ) => {
        kernels!(@all $(#[$attr])* f64, shared_f64, pairs_f64, dot_axpy_f64,
            $vec, $per, $zero, $splat, $load, $load, $add, $mul, $store);
        kernels!(@all $(#[$attr])* f32, shared_f32, pairs_f32, dot_axpy_f32,
            $vec, $per, $zero, $splat, $load, $load_f32, $add, $mul, $store);
    };
    (@all $(#[$attr:meta])* $w:ty, $shared:ident, $pairs:ident, $dot_axpy:ident,
        $vec:ty, $per:expr, $zero:expr, $splat:expr, $load:expr, $load_w:expr,
        $add:expr, $mul:expr, $store:expr) => {
        $(#[$attr])*
        pub(crate) unsafe fn $shared<const R: usize>(w: &[$w], s: [&[f64]; R]) -> [f64; R] {
            const PER: usize = $per;
            const VECS: usize = LANES / PER;
            let n = w.len();
            let full = n - n % LANES;
            let wp = w.as_ptr();
            let mut sp = [core::ptr::null::<f64>(); R];
            for r in 0..R {
                assert!(s[r].len() >= n, "signal shorter than its weights");
                sp[r] = s[r].as_ptr();
            }
            let mut acc: [[$vec; VECS]; R] = [[$zero(); VECS]; R];
            let mut k = 0;
            while k < full {
                let mut wv: [$vec; VECS] = [$zero(); VECS];
                for v in 0..VECS {
                    wv[v] = $load_w(wp.add(k + v * PER));
                }
                for r in 0..R {
                    for v in 0..VECS {
                        let sv = $load(sp[r].add(k + v * PER));
                        acc[r][v] = $add(acc[r][v], $mul(wv[v], sv));
                    }
                }
                k += LANES;
            }
            let mut out = [0.0f64; R];
            for r in 0..R {
                let mut lanes = [0.0f64; LANES];
                for v in 0..VECS {
                    $store(lanes.as_mut_ptr().add(v * PER), acc[r][v]);
                }
                let mut total = reduce(lanes);
                for j in full..n {
                    total += f64::from(w[j]) * s[r][j];
                }
                out[r] = total;
            }
            out
        }

        $(#[$attr])*
        pub(crate) unsafe fn $pairs<const R: usize>(w: [&[$w]; R], s: [&[f64]; R]) -> [f64; R] {
            const PER: usize = $per;
            const VECS: usize = LANES / PER;
            let mut full = [0usize; R];
            let mut common = usize::MAX;
            for r in 0..R {
                assert!(s[r].len() >= w[r].len(), "signal shorter than its weights");
                full[r] = w[r].len() - w[r].len() % LANES;
                common = common.min(full[r]);
            }
            let mut acc: [[$vec; VECS]; R] = [[$zero(); VECS]; R];
            // Lockstep over the full groups every row has.
            let mut k = 0;
            while k < common {
                for r in 0..R {
                    for v in 0..VECS {
                        let wv = $load_w(w[r].as_ptr().add(k + v * PER));
                        let sv = $load(s[r].as_ptr().add(k + v * PER));
                        acc[r][v] = $add(acc[r][v], $mul(wv, sv));
                    }
                }
                k += LANES;
            }
            let mut out = [0.0f64; R];
            for r in 0..R {
                // The rest of this row's full groups, into the same lanes.
                let mut k = common;
                while k < full[r] {
                    for v in 0..VECS {
                        let wv = $load_w(w[r].as_ptr().add(k + v * PER));
                        let sv = $load(s[r].as_ptr().add(k + v * PER));
                        acc[r][v] = $add(acc[r][v], $mul(wv, sv));
                    }
                    k += LANES;
                }
                let mut lanes = [0.0f64; LANES];
                for v in 0..VECS {
                    $store(lanes.as_mut_ptr().add(v * PER), acc[r][v]);
                }
                let mut total = reduce(lanes);
                for j in full[r]..w[r].len() {
                    total += f64::from(w[r][j]) * s[r][j];
                }
                out[r] = total;
            }
            out
        }

        $(#[$attr])*
        pub(crate) unsafe fn $dot_axpy<const R: usize>(
            w: &[$w],
            s: [&[f64]; R],
            z: [&mut [f64]; R],
            a: [f64; R],
        ) -> [f64; R] {
            const PER: usize = $per;
            const VECS: usize = LANES / PER;
            let n = w.len();
            let full = n - n % LANES;
            let wp = w.as_ptr();
            let mut sp = [core::ptr::null::<f64>(); R];
            let mut zp = [core::ptr::null_mut::<f64>(); R];
            let mut av: [$vec; R] = [$zero(); R];
            for r in 0..R {
                assert!(s[r].len() >= n && z[r].len() >= n, "slice shorter than its weights");
                sp[r] = s[r].as_ptr();
                zp[r] = z[r].as_mut_ptr();
                av[r] = $splat(a[r]);
            }
            let mut acc: [[$vec; VECS]; R] = [[$zero(); VECS]; R];
            let mut k = 0;
            while k < full {
                let mut wv: [$vec; VECS] = [$zero(); VECS];
                for v in 0..VECS {
                    wv[v] = $load_w(wp.add(k + v * PER));
                }
                for r in 0..R {
                    for v in 0..VECS {
                        let at = k + v * PER;
                        acc[r][v] = $add(acc[r][v], $mul(wv[v], $load(sp[r].add(at))));
                        $store(zp[r].add(at), $add($load(zp[r].add(at)), $mul(wv[v], av[r])));
                    }
                }
                k += LANES;
            }
            let mut out = [0.0f64; R];
            for r in 0..R {
                let mut lanes = [0.0f64; LANES];
                for v in 0..VECS {
                    $store(lanes.as_mut_ptr().add(v * PER), acc[r][v]);
                }
                let mut total = reduce(lanes);
                for j in full..n {
                    let weight = f64::from(w[j]);
                    total += weight * *sp[r].add(j);
                    *zp[r].add(j) += weight * a[r];
                }
                out[r] = total;
            }
            out
        }
    };
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{reduce, LANES};
    use core::arch::x86_64::*;

    kernels!(
        #[target_feature(enable = "avx512f")]
        __m512d, 8,
        zero: || _mm512_setzero_pd(),
        splat: |a: f64| _mm512_set1_pd(a),
        load: |p: *const f64| _mm512_loadu_pd(p),
        load_f32: |p: *const f32| _mm512_cvtps_pd(_mm256_loadu_ps(p)),
        add: |a, b| _mm512_add_pd(a, b),
        mul: |a, b| _mm512_mul_pd(a, b),
        store: |p: *mut f64, v| _mm512_storeu_pd(p, v),
    );

    pub(crate) mod avx2 {
        use super::super::{reduce, LANES};
        use core::arch::x86_64::*;

        kernels!(
            #[target_feature(enable = "avx2")]
            __m256d, 4,
            zero: || _mm256_setzero_pd(),
            splat: |a: f64| _mm256_set1_pd(a),
            load: |p: *const f64| _mm256_loadu_pd(p),
            load_f32: |p: *const f32| _mm256_cvtps_pd(_mm_loadu_ps(p)),
            add: |a, b| _mm256_add_pd(a, b),
            mul: |a, b| _mm256_mul_pd(a, b),
            store: |p: *mut f64, v| _mm256_storeu_pd(p, v),
        );
    }

    // SSE2 is part of every x86_64 CPU.
    pub(crate) mod sse2 {
        use super::super::{reduce, LANES};
        use core::arch::x86_64::*;

        kernels!(
            __m128d, 2,
            zero: || _mm_setzero_pd(),
            splat: |a: f64| _mm_set1_pd(a),
            load: |p: *const f64| _mm_loadu_pd(p),
            load_f32: |p: *const f32| _mm_cvtps_pd(_mm_castpd_ps(_mm_load_sd(p as *const f64))),
            add: |a, b| _mm_add_pd(a, b),
            mul: |a, b| _mm_mul_pd(a, b),
            store: |p: *mut f64, v| _mm_storeu_pd(p, v),
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{reduce, LANES};
    use core::arch::aarch64::*;

    // NEON is part of every aarch64 CPU.
    kernels!(
        float64x2_t, 2,
        zero: || vdupq_n_f64(0.0),
        splat: |a: f64| vdupq_n_f64(a),
        load: |p: *const f64| vld1q_f64(p),
        load_f32: |p: *const f32| vcvt_f64_f32(vld1_f32(p)),
        add: |a, b| vaddq_f64(a, b),
        mul: |a, b| vmulq_f64(a, b),
        store: |p: *mut f64, v| vst1q_f64(p, v),
    );
}

/// Portable kernels, and the reference statement of the order the others follow.
mod scalar {
    use super::{reduce, LANES};

    macro_rules! scalar_kernels {
        ($w:ty, $shared:ident, $pairs:ident, $dot_axpy:ident) => {
            pub(crate) fn $shared<const R: usize>(w: &[$w], s: [&[f64]; R]) -> [f64; R] {
                core::array::from_fn(|r| one(w, s[r]))
            }

            pub(crate) fn $pairs<const R: usize>(w: [&[$w]; R], s: [&[f64]; R]) -> [f64; R] {
                core::array::from_fn(|r| one(w[r], s[r]))
            }

            pub(crate) fn $dot_axpy<const R: usize>(
                w: &[$w],
                s: [&[f64]; R],
                z: [&mut [f64]; R],
                a: [f64; R],
            ) -> [f64; R] {
                let mut out = [0.0; R];
                for (r, z) in z.into_iter().enumerate() {
                    for (k, &weight) in w.iter().enumerate() {
                        z[k] += f64::from(weight) * a[r];
                    }
                    out[r] = one(w, s[r]);
                }
                out
            }
        };
    }

    fn one<W: Copy + Into<f64>>(w: &[W], s: &[f64]) -> f64 {
        let n = w.len();
        let s = &s[..n];
        let full = n - n % LANES;
        let mut lanes = [0.0f64; LANES];
        for k in 0..full {
            lanes[k % LANES] += w[k].into() * s[k];
        }
        let mut total = reduce(lanes);
        for k in full..n {
            total += w[k].into() * s[k];
        }
        total
    }

    scalar_kernels!(f64, shared_f64, pairs_f64, dot_axpy_f64);
    scalar_kernels!(f32, shared_f32, pairs_f32, dot_axpy_f32);
}

/// The widest instruction set the kernels can use on this CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Always available; the native kernels are checked against it.
    #[cfg_attr(any(target_arch = "x86_64", target_arch = "aarch64"), allow(dead_code))]
    Scalar,
}

impl Isa {
    pub fn detect() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return Isa::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Isa::Avx2;
            }
            Isa::Sse2
        }
        #[cfg(target_arch = "aarch64")]
        {
            Isa::Neon
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Isa::Scalar
        }
    }

    /// Spectra that share one load of the weights: as many as their
    /// accumulators (16 lanes each) leave registers for.
    pub fn batch_width(self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => 4,
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => 2,
            #[cfg(target_arch = "x86_64")]
            Isa::Sse2 => 1,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => 2,
            Isa::Scalar => 1,
        }
    }
}

/// A stored weight type, with the kernels for it on every instruction set.
pub trait Weight: Copy + Send + Sync + 'static {
    fn shared<const R: usize>(isa: Isa, w: &[Self], s: [&[f64]; R]) -> [f64; R];
    fn pairs<const R: usize>(isa: Isa, w: [&[Self]; R], s: [&[f64]; R]) -> [f64; R];
    fn dot_axpy<const R: usize>(
        isa: Isa,
        w: &[Self],
        s: [&[f64]; R],
        z: [&mut [f64]; R],
        a: [f64; R],
    ) -> [f64; R];
}

macro_rules! dispatch {
    ($isa:expr, $name:ident ( $($arg:expr),* )) => {
        match $isa {
            // SAFETY: `Isa::detect` found AVX-512F at run time.
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => unsafe { x86::$name($($arg),*) },
            // SAFETY: `Isa::detect` found AVX2 at run time.
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => unsafe { x86::avx2::$name($($arg),*) },
            // SAFETY: SSE2 is baseline on x86_64.
            #[cfg(target_arch = "x86_64")]
            Isa::Sse2 => unsafe { x86::sse2::$name($($arg),*) },
            // SAFETY: NEON is baseline on aarch64.
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => unsafe { arm::$name($($arg),*) },
            Isa::Scalar => scalar::$name($($arg),*),
        }
    };
}

macro_rules! weight_impl {
    ($w:ty, $shared:ident, $pairs:ident, $dot_axpy:ident) => {
        impl Weight for $w {
            #[inline]
            fn shared<const R: usize>(isa: Isa, w: &[Self], s: [&[f64]; R]) -> [f64; R] {
                dispatch!(isa, $shared(w, s))
            }

            #[inline]
            fn pairs<const R: usize>(isa: Isa, w: [&[Self]; R], s: [&[f64]; R]) -> [f64; R] {
                dispatch!(isa, $pairs(w, s))
            }

            #[inline]
            fn dot_axpy<const R: usize>(
                isa: Isa,
                w: &[Self],
                s: [&[f64]; R],
                z: [&mut [f64]; R],
                a: [f64; R],
            ) -> [f64; R] {
                dispatch!(isa, $dot_axpy(w, s, z, a))
            }
        }
    };
}

weight_impl!(f64, shared_f64, pairs_f64, dot_axpy_f64);
weight_impl!(f32, shared_f32, pairs_f32, dot_axpy_f32);

/// `w · s[..w.len()]`.
pub fn dot<W: Weight>(w: &[W], s: &[f64]) -> f64 {
    W::shared::<1>(Isa::detect(), w, [s])[0]
}

/// `out[r] = w · signals[r][..w.len()]` for every `r`, each equal bit for bit
/// to `dot(w, signals[r])`.
pub fn dot_batch<W: Weight>(w: &[W], signals: &[&[f64]], out: &mut [f64]) {
    assert_eq!(signals.len(), out.len());
    let isa = Isa::detect();
    let width = isa.batch_width();
    let mut r = 0;
    while r < signals.len() {
        let left = signals.len() - r;
        if width >= 4 && left >= 4 {
            let s = [signals[r], signals[r + 1], signals[r + 2], signals[r + 3]];
            out[r..r + 4].copy_from_slice(&W::shared::<4>(isa, w, s));
            r += 4;
        } else if width >= 2 && left >= 2 {
            let s = [signals[r], signals[r + 1]];
            out[r..r + 2].copy_from_slice(&W::shared::<2>(isa, w, s));
            r += 2;
        } else {
            out[r] = W::shared::<1>(isa, w, [signals[r]])[0];
            r += 1;
        }
    }
}

/// Four dot products of four weight rows against four signals, streamed
/// together; each equals `dot(w[r], s[r])` bit for bit.
pub fn dot_rows4<W: Weight>(w: [&[W]; 4], s: [&[f64]; 4]) -> [f64; 4] {
    W::pairs::<4>(Isa::detect(), w, s)
}

/// For every `r`: `out[r] = w · signals[r][..w.len()]`, equal bit for bit to
/// `dot`, and in the same pass `z[r][k] += w[k] * a[r]` for every `k`.
pub fn dot_axpy_batch<W: Weight>(
    w: &[W],
    signals: &[&[f64]],
    z: &mut [&mut [f64]],
    a: &[f64],
    out: &mut [f64],
) {
    let count = signals.len();
    assert!(z.len() == count && a.len() == count && out.len() == count);
    let isa = Isa::detect();
    let width = isa.batch_width();
    let mut r = 0;
    while r < count {
        let left = count - r;
        if width >= 4 && left >= 4 {
            let [z0, z1, z2, z3] = &mut z[r..r + 4] else {
                unreachable!()
            };
            let s = [signals[r], signals[r + 1], signals[r + 2], signals[r + 3]];
            let zs = [&mut **z0, &mut **z1, &mut **z2, &mut **z3];
            let av = [a[r], a[r + 1], a[r + 2], a[r + 3]];
            out[r..r + 4].copy_from_slice(&W::dot_axpy::<4>(isa, w, s, zs, av));
            r += 4;
        } else if width >= 2 && left >= 2 {
            let [z0, z1] = &mut z[r..r + 2] else {
                unreachable!()
            };
            let s = [signals[r], signals[r + 1]];
            out[r..r + 2].copy_from_slice(&W::dot_axpy::<2>(
                isa,
                w,
                s,
                [&mut **z0, &mut **z1],
                [a[r], a[r + 1]],
            ));
            r += 2;
        } else {
            out[r] = W::dot_axpy::<1>(isa, w, [signals[r]], [&mut *z[r]], [a[r]])[0];
            r += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.3
            })
            .collect()
    }

    fn isas() -> Vec<Isa> {
        let mut all = vec![Isa::Scalar];
        #[cfg(target_arch = "x86_64")]
        {
            all.push(Isa::Sse2);
            if std::arch::is_x86_feature_detected!("avx2") {
                all.push(Isa::Avx2);
            }
            if std::arch::is_x86_feature_detected!("avx512f") {
                all.push(Isa::Avx512);
            }
        }
        #[cfg(target_arch = "aarch64")]
        all.push(Isa::Neon);
        all
    }

    fn reference<W: Copy + Into<f64>>(w: &[W], s: &[f64]) -> f64 {
        let n = w.len();
        let full = n - n % LANES;
        let mut lanes = [0.0f64; LANES];
        for k in 0..full {
            lanes[k % LANES] += w[k].into() * s[k];
        }
        let mut total = reduce(lanes);
        for k in full..n {
            total += w[k].into() * s[k];
        }
        total
    }

    fn check<W: Weight + Into<f64>>(w: &[W], refs: &[&[f64]]) {
        let n = w.len();
        let expected: Vec<f64> = refs.iter().map(|s| reference(w, s)).collect();
        let bits = |v: f64| v.to_bits();
        for isa in isas() {
            for (r, s) in refs.iter().enumerate() {
                assert_eq!(
                    bits(W::shared::<1>(isa, w, [s])[0]),
                    bits(expected[r]),
                    "{isa:?} n={n}"
                );
            }
            let two = W::shared::<2>(isa, w, [refs[0], refs[1]]);
            let four = W::shared::<4>(isa, w, [refs[0], refs[1], refs[2], refs[3]]);
            assert_eq!(bits(two[1]), bits(expected[1]), "{isa:?} n={n}");
            assert_eq!(bits(four[3]), bits(expected[3]), "{isa:?} n={n}");

            // Rows of different lengths in lockstep.
            let cut = [n, n.saturating_sub(3), n / 2, n.saturating_sub(17)];
            let rows: [&[W]; 4] = core::array::from_fn(|r| &w[..cut[r]]);
            let got = W::pairs::<4>(isa, rows, [refs[0], refs[1], refs[2], refs[3]]);
            for r in 0..4 {
                assert_eq!(
                    bits(got[r]),
                    bits(reference(rows[r], refs[r])),
                    "{isa:?} pairs n={n}"
                );
            }

            // Fused dot and axpy against the separate operations.
            let mut zs: Vec<Vec<f64>> = (0..4).map(|r| data(n + 2, 40 + r as u64)).collect();
            let mut want = zs.clone();
            let a = [0.5, -1.25, 3.0, 1e-3];
            for r in 0..4 {
                for k in 0..n {
                    want[r][k] += w[k].into() * a[r];
                }
            }
            let [z0, z1, z2, z3] = &mut zs[..] else {
                unreachable!()
            };
            let got = W::dot_axpy::<4>(
                isa,
                w,
                [refs[0], refs[1], refs[2], refs[3]],
                [&mut z0[..], &mut z1[..], &mut z2[..], &mut z3[..]],
                a,
            );
            for r in 0..4 {
                assert_eq!(bits(got[r]), bits(expected[r]), "{isa:?} dot_axpy n={n}");
                for k in 0..n + 2 {
                    assert_eq!(bits(zs[r][k]), bits(want[r][k]), "{isa:?} axpy n={n} k={k}");
                }
            }
        }
        let mut out = vec![0.0; refs.len()];
        dot_batch(w, refs, &mut out);
        for (r, value) in out.iter().enumerate() {
            assert_eq!(bits(*value), bits(expected[r]), "batch n={n} r={r}");
            assert_eq!(bits(dot(w, refs[r])), bits(expected[r]), "dot n={n} r={r}");
        }
        let mut zs: Vec<Vec<f64>> = (0..refs.len()).map(|_| vec![0.0; n]).collect();
        let mut zrefs: Vec<&mut [f64]> = zs.iter_mut().map(|z| z.as_mut_slice()).collect();
        let a: Vec<f64> = (0..refs.len()).map(|r| r as f64 + 0.5).collect();
        dot_axpy_batch(w, refs, &mut zrefs, &a, &mut out);
        for (r, value) in out.iter().enumerate() {
            assert_eq!(
                bits(*value),
                bits(expected[r]),
                "dot_axpy_batch n={n} r={r}"
            );
        }
    }

    #[test]
    fn every_instruction_set_follows_the_documented_order() {
        for n in [0usize, 1, 15, 16, 17, 31, 32, 33, 100, 1000, 4097] {
            let w = data(n, 1);
            let w32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
            let signals: Vec<Vec<f64>> = (0..7).map(|r| data(n + 5, 10 + r)).collect();
            let refs: Vec<&[f64]> = signals.iter().map(|s| s.as_slice()).collect();
            check(&w, &refs);
            check(&w32, &refs);
        }
    }
}
