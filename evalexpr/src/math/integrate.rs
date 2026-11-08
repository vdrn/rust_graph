use crate::{EvalexprError, EvalexprFloat, EvalexprResult};

const N: usize = 11;
const M: [usize; N] = [6, 7, 13, 26, 53, 106, 212, 423, 846, 1693, 3385];

pub struct AbscissasWeights<F: EvalexprFloat> {
	r: [Vec<F>; N],
	w: [Vec<F>; N],
}

/// Precomputes abscissas and weights for the Tanh-Sinh quadrature
pub fn get_tanh_sinh_abscissas_and_weights<F: EvalexprFloat>() -> AbscissasWeights<F> {
	let mut r: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
	let mut w: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
	let zero = F::from_f64(0.0);
	let one = F::from_f64(1.0);
	let two = F::from_f64(2.0);
	let half = F::from_f64(0.5);

	let mut h = two;

	for i in 0..N {
		h = h * half;
		let eh = h.exp();
		let mut t = h.exp();

		r[i] = Vec::with_capacity(M[i]);
		w[i] = Vec::with_capacity(M[i]);

		let eh_squared = if i > 0 { eh * eh } else { eh };

		for _ in 0..M[i] {
			let t1 = t.recip();
			let u = (t1 - t).exp();
			let u1 = (one + u).recip();
			let d = two * u * u1;

			if d == zero {
				break;
			}

			r[i].push(d);
			w[i].push((t1 + t) * d * u1);
			t = t * eh_squared;
		}
	}

	AbscissasWeights { r, w }
}

pub struct IntegrationResult<F: EvalexprFloat> {
	pub value: F,
}
pub struct Precision<F: EvalexprFloat> {
	pub lower:                F,
	pub upper:                F,
	pub precision:            F,
	pub large_integral_value: F,
}

/// A quadrature node: the abscissa `x` together with its distances from the two interval ends.
///
/// `lo` and `hi` come straight from the abscissa table (`r * d`), so they stay accurate even when
/// `x` itself has already been rounded onto an endpoint (e.g. `1.0 - 1e-20 == 1.0`).
#[derive(Clone, Copy)]
pub struct Node<F: EvalexprFloat> {
	pub x:  F,
	pub lo: F, // x - x1
	pub hi: F, // x2 - x
}

/// Calculates the definite integral of the function `f` over [x1, x2] with Tanh-Sinh quadrature.
///
/// Plain version: Abscissas that round onto an endpoint are skipped so singularities exactly at an endpoint
/// are never evaluated.
pub fn tanh_sinh<F: EvalexprFloat>(
	x1: F, x2: F, mut f: impl FnMut(F) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<IntegrationResult<F>, F> {
	tanh_sinh_nodes(x1, x2, |n| if n.x > x1 && n.x < x2 { f(n.x) } else { Ok(F::NAN) }, precision)
}

/// Integrates over [origin, ∞) (sign = 1) or (-∞, origin] (sign = -1).
fn half_line<F: EvalexprFloat>(
	origin: F, sign: F, f: &mut impl FnMut(F) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<IntegrationResult<F>, F> {
	tanh_sinh_nodes(
		F::from_f64(0.0),
		F::from_f64(1.0),
		|n| {
			let x = origin + sign * (n.lo / n.hi);
			// Two divisions instead of `/ (hi * hi)` to avoid overflow of square for small hi
			Ok(f(x)? / n.hi / n.hi)
		},
		precision,
	)
}

/// Calculates the definite integral of the function `f`
/// in the interval [x1, x2] numerically, with the specified precision.
/// `f` also receives each node's exact distance from both endpoints (used when the integrand is singular or
/// needs resolving closer to an endpoint than the floating-point spacing of the abscissa allows (inf-interval
/// transformations))
///
/// Implements the Tanh-Sinh quadrature
///
/// Adapted from: https://github.com/Proektsoftbg/Numerical/blob/main/Numerical/Integrator/TanhSinh.cs
/// which is based on [Improving the Double Exponential Quadrature Tanh-Sinh, Sinh-Sinh and Exp-Sinh Formulas](https://www.genivia.com/files/qthsh.pdf)
pub fn tanh_sinh_nodes<F: EvalexprFloat>(
	x1: F, x2: F, mut f: impl FnMut(Node<F>) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<IntegrationResult<F>, F> {
	let data = F::abscissas_and_weights();

	let zero = F::from_f64(0.0);
	let half = F::from_f64(0.5);
	let two = F::from_f64(2.0);

	if x1 == x2 {
		return Ok(IntegrationResult { value: zero });
	}

	let width = x2 - x1;
	let c = (x1 + x2) * half;
	let d = width * half;
	let mut s = f(Node { x: c, lo: d, hi: d })?;

	let eps = (precision.precision * F::from_f64(0.1)).clamp(&precision.lower, &precision.upper);
	let tol = F::from_f64(10.0) * precision.precision;
	let mut err;
	let mut converged;
	let mut i = 0;

	loop {
		let mut p = zero;
		let mut fp = zero;
		let mut fm = zero;

		let len_i = data.r[i].len();
		let mut j = 0;
		loop {
			// Distance of this node pair from the nearest endpoint. Exact to full relative precision,
			// unlike `x1 + dist` / `x2 - dist`, which are rounded to the spacing of x1 / x2.
			let dist = data.r[i][j] * d;
			let far = width - dist; // always >= d: no cancellation

			let y = f(Node { x: x1 + dist, lo: dist, hi: far })?;
			if y.is_finite() {
				fp = y;
			}

			let y = f(Node { x: x2 - dist, lo: far, hi: dist })?;
			if y.is_finite() {
				fm = y;
			}

			let q = data.w[i][j] * (fp + fm);
			p = p + q;
			j += 1;

			if q.abs() <= eps * p.abs() || j >= len_i {
				break;
			}
		}

		err = two * s;
		s = s + p;
		err = (err - s).abs();
		i += 1;

		converged = err <= tol * s.abs();
		if converged || i >= N {
			break;
		}
	}

	let mut result = d * s * two.pow(&F::from_i32(1 - i as i32));

	// Divergence check. Only when we ran out of levels *without* meeting the tolerance, and with
	// the error measured relative to s (it used to be compared as an absolute number).
	if !converged && (result.abs() > precision.large_integral_value || err > tol * F::from_f64(10.0) * s.abs())
	{
		result = F::INFINITY * result.signum();
	}

	Ok(IntegrationResult { value: result })
}

/// Integrate over possibly infinite bounds using appropriate transformations.
///
/// # Transformations used:
/// * [a, b] (both finite): Direct integration
/// * [a, ∞): x = a + t/(1-t), t ∈ [0, 1)
/// * (-∞, b]: x = b - t/(1-t), t ∈ [0, 1)
/// * (-∞, ∞): split at 0 into the two cases above
pub fn integrate<F: EvalexprFloat>(
	mut lower: F, mut upper: F, mut f: impl FnMut(F) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<F, F> {
	if lower.is_nan() || upper.is_nan() {
		return Err(EvalexprError::CustomMessage("Integral bounds are undefined.".to_string()));
	}
	let mut swapped = false;
	if lower > upper {
		swapped = true;
		core::mem::swap(&mut lower, &mut upper);
	}
	let zero = F::from_f64(0.0);
	let one = F::from_f64(1.0);

	// Empty interval (also stops (∞, ∞) from being treated as the whole real line)
	if lower == upper {
		return Ok(zero);
	}

	let result = match (lower.is_finite(), upper.is_finite()) {
		// Both bounds finite
		(true, true) => tanh_sinh(lower, upper, f, precision),

		// [a, ∞)
		(true, false) => half_line(lower, one, &mut f, precision),

		// (-∞, b]
		(false, true) => half_line(upper, -one, &mut f, precision),

		// Both bounds infinite
		(false, false) => {
			const USE_CAUCHY: bool = false;
			if USE_CAUCHY {
				// Cauchy principal value: symmetric limit
				// Transform: x = t/(1-t²), dx = (1+t²)/(1-t²)² dt, t ∈ (-1, 1)
				// For the node, lo = 1 + t and hi = 1 - t, so 1 - t² = lo * hi exactly.
				tanh_sinh_nodes(
					-one,
					one,
					|n| {
						let w = n.lo * n.hi;
						let x = n.x / w;
						Ok(f(x)? * (one + n.x * n.x) / w / w)
					},
					precision,
				)
			} else {
				// Standard definition: sum the two halves independently
				let left_half = half_line(zero, -one, &mut f, precision)?;
				let right_half = half_line(zero, one, &mut f, precision)?;
				Ok(IntegrationResult { value: left_half.value + right_half.value })
			}
		},
	}?;
	if !result.value.is_finite() {
		return Ok(F::NAN);
	}
	Ok(result.value * if swapped { -F::ONE } else { F::ONE })
}

#[cfg(test)]
mod tests {
	use super::*;

	const PRECISION: Precision<f64> = Precision {
		lower:                1e-12,
		upper:                1e-8,
		precision:            1e-10,
		large_integral_value: 1e30,
	};
	const F64_PRECISION: Precision<f64> = Precision {
		lower:                1e-11,
		upper:                1e-7,
		precision:            1e-9,
		large_integral_value: 1e30,
	};
	const F32_PRECISION: Precision<f32> = Precision {
		lower:                1e-7,
		upper:                1e-4,
		precision:            1e-5,
		large_integral_value: 1e13,
	};

	#[test]
	fn test_simple_integral() {
		let result = tanh_sinh(0.0, 1.0, |x| Ok(x * x), &PRECISION).unwrap();
		assert!((result.value - 1.0 / 3.0).abs() < 1e-10);
	}

	#[test]
	fn test_sin_integral() {
		let result = tanh_sinh(0.0, std::f64::consts::PI, |x| Ok(x.sin()), &PRECISION).unwrap();
		assert!((result.value - 2.0).abs() < 1e-10);
	}

	#[test]
	fn test_exp_integral() {
		let result = tanh_sinh(0.0, 1.0, |x| Ok(x.exp()), &PRECISION).unwrap();
		assert!((result.value - (std::f64::consts::E - 1.0)).abs() < 1e-10);
	}

	#[test]
	fn test_infinite_upper_bound() {
		let result = integrate(0.0, f64::INFINITY, |x| Ok((-x).exp()), &PRECISION).unwrap();
		assert!((result - 1.0).abs() < 1e-9);
	}

	#[test]
	fn test_infinite_lower_bound() {
		let result = integrate(f64::NEG_INFINITY, 0.0, |x| Ok(x.exp()), &PRECISION).unwrap();
		assert!((result - 1.0).abs() < 1e-9);
	}

	#[test]
	fn test_both_infinite_bounds() {
		let result = integrate(f64::NEG_INFINITY, f64::INFINITY, |x| Ok((-x * x).exp()), &PRECISION).unwrap();
		let expected = std::f64::consts::PI.sqrt();
		assert!((result - expected).abs() < 1e-8);
	}

	#[test]
	fn test_finite_bounds_with_integrate() {
		let result = integrate(0.0, 1.0, |x| Ok(x * x), &PRECISION).unwrap();
		assert!((result - 1.0 / 3.0).abs() < 1e-10);
	}

	#[test]
	fn test_slowly_decaying_tail_upper_f64() {
		for eps in [0.4, 0.3, 0.1, 0.05] {
			let r = integrate(1.0, f64::INFINITY, |x| Ok(x.powf(-(1.0 + eps))), &F64_PRECISION).unwrap();
			assert!((r * eps - 1.0).abs() < 1e-8, "eps = {eps}: got {r}");
		}
	}

	#[test]
	fn test_slowly_decaying_tail_lower_f64() {
		for eps in [0.4, 0.3, 0.1, 0.05] {
			let r = integrate(f64::NEG_INFINITY, -1.0, |x| Ok(x.abs().powf(-(1.0 + eps))), &F64_PRECISION)
				.unwrap();
			assert!((r * eps - 1.0).abs() < 1e-8, "eps = {eps}: got {r}");
		}
	}

	#[test]
	fn test_slowly_decaying_tail_both_f64() {
		for eps in [0.4, 0.3, 0.1, 0.05] {
			let r = integrate(
				f64::NEG_INFINITY,
				f64::INFINITY,
				|x| Ok((1.0 + x.abs()).powf(-(1.0 + eps))),
				&F64_PRECISION,
			)
			.unwrap();
			assert!((r * eps / 2.0 - 1.0).abs() < 1e-8, "eps = {eps}: got {r}");
		}
	}

	// f32 can't represent x beyond ~3e38, so it is only trustworthy for eps >~ 0.1
	#[test]
	fn test_slowly_decaying_tail_f32() {
		for eps in [0.4f32, 0.2] {
			let r = integrate(1.0f32, f32::INFINITY, |x| Ok(x.powf(-(1.0 + eps))), &F32_PRECISION).unwrap();
			assert!((r * eps - 1.0).abs() < 1e-5, "eps = {eps}: got {r}");
		}
	}

	#[test]
	fn test_empty_interval() {
		assert_eq!(integrate(1.0, 1.0, |x| Ok(x), &F64_PRECISION).unwrap(), 0.0);
		assert_eq!(integrate(f64::INFINITY, f64::INFINITY, |x| Ok(x), &F64_PRECISION).unwrap(), 0.0);
	}

	// ∫_1^∞ 1/x diverges (f64 only: see the note about f32 range truncation)
	#[test]
	fn test_divergent_is_nan_f64() {
		assert!(integrate(1.0, f64::INFINITY, |x| Ok(1.0 / x), &F64_PRECISION).unwrap().is_nan());
		assert!(integrate(0.0, f64::INFINITY, |_| Ok(1.0), &F64_PRECISION).unwrap().is_nan());
	}
}
