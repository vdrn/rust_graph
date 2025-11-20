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
	let zero = F::f64_to_float(0.0);
	let one = F::f64_to_float(1.0);
	let two = F::f64_to_float(2.0);
	let half = F::f64_to_float(0.5);

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
	pub value:           F,
	pub iteration_count: usize,
}
pub struct Precision<F: EvalexprFloat> {
	pub lower:     F,
	pub upper:     F,
	pub precision: F,
}

/// Calculates the definite integral of the function `f`
/// in the interval [x1, x2] numerically, with the specified precision.
///
/// Implements the Tanh-Sinh quadrature
///
/// Adapted from: https://github.com/Proektsoftbg/Numerical/blob/main/Numerical/Integrator/TanhSinh.cs
pub fn tanh_sinh<F: EvalexprFloat>(
	x1: F, x2: F, mut f: impl FnMut(F) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<IntegrationResult<F>, F> {
	let data = F::abscissas_and_weights();

	let zero = F::f64_to_float(0.0);
	let half = F::f64_to_float(0.5);
	let two = F::f64_to_float(2.0);

	let mut iteration_count = 1;

	let c = (x1 + x2) * half;
	let d = (x2 - x1) * half;
	let mut s = f(c)?;

	let eps = precision.precision.clamp(&precision.lower, &precision.upper) * F::f64_to_float(0.1);
	let tol = F::f64_to_float(10.0) * precision.precision;
	let mut err;
	let mut i = 0;

	loop {
		let mut p = zero;
		let mut j = 0;

		loop {
			let x = data.r[i][j] * d;
			let mut fp = zero;
			let mut fm = zero;

			if x1 + x > x1 {
				let y = f(x1 + x)?;
				iteration_count += 1;
				if y.is_finite() {
					fp = y;
				}
			}

			if x2 - x < x2 {
				let y = f(x2 - x)?;
				iteration_count += 1;
				if y.is_finite() {
					fm = y;
				}
			}

			let q = data.w[i][j] * (fp + fm);
			p = p + q;
			j += 1;

			if q.abs() <= eps * p.abs() || j >= M[i] {
				break;
			}
		}

		err = two * s;
		s = s + p;
		err = (err - s).abs();
		i += 1;

		if err <= tol * s.abs() || i >= N {
			break;
		}
	}

	// When the integral is smaller than precision, the absolute error is evaluated
	// if s.abs() > one {
	//     err = err / s.abs();
	// }

	let result = d * s * two.pow(&F::from_i32(1 - i as i32));

	Ok(IntegrationResult { value: result, iteration_count })
}

/// Integrate over possibly infinite bounds using appropriate transformations.
///
/// # Transformations used:
/// * [a, b] (both finite): Direct integration
/// * [a, ∞): x = a + t/(1-t), t ∈ [0, 1)
/// * (-∞, b]: x = b - t/(1-t), t ∈ [0, 1)
/// * (-∞, ∞): x = t/(1-t²), t ∈ (-1, 1)
pub fn integrate<F: EvalexprFloat>(
	mut lower: F, mut upper: F, mut f: impl FnMut(F) -> EvalexprResult<F, F>, precision: &Precision<F>,
) -> EvalexprResult<IntegrationResult<F>, F> {
	if lower.is_nan() || upper.is_nan() {
		return Err(EvalexprError::CustomMessage("Integral bounds are undefined.".to_string()));
	}
	if lower > upper {
		core::mem::swap(&mut lower, &mut upper);
	}
	let zero = F::f64_to_float(0.0);
	let one = F::f64_to_float(1.0);

	match (lower.is_finite(), upper.is_finite()) {
		// Both bounds finite
		(true, true) => tanh_sinh(lower, upper, f, precision),

		// Lower bound finite, upper bound infinite: [a, ∞)
		// Transform: x = a + t/(1-t), dx = dt/(1-t)²
		(true, false) => {
			let transformed = |t: F| {
				if t >= one {
					return Ok(zero);
				}
				let one_minus_t = one - t;
				let x = lower + t / one_minus_t;
				let dx_dt = one / (one_minus_t * one_minus_t);
				Ok(f(x)? * dx_dt)
			};
			tanh_sinh(zero, one, transformed, precision)
		},

		// Lower bound infinite, upper bound finite: (-∞, b]
		// Transform: x = b - t/(1-t), dx = dt/(1-t)²
		(false, true) => {
			let transformed = |t: F| {
				if t >= one {
					return Ok(zero);
				}
				let one_minus_t = one - t;
				let x = upper - t / one_minus_t;
				let dx_dt = one / (one_minus_t * one_minus_t);
				Ok(f(x)? * dx_dt)
			};
			tanh_sinh(zero, one, transformed, precision)
		},

		// Both bounds infinite: (-∞, ∞)
		// Transform: x = t/(1-t²), dx = (1+t²)/(1-t²)² dt
		(false, false) => {
			let transformed = |t: F| {
				if t.abs() >= one {
					return Ok(zero);
				}
				let t_sq = t * t;
				let one_minus_t_sq = one - t_sq;
				let x = t / one_minus_t_sq;
				let dx_dt = (one + t_sq) / (one_minus_t_sq * one_minus_t_sq);
				Ok(f(x)? * dx_dt)
			};
			tanh_sinh(-one, one, transformed, precision)
		},
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	const PRECISION: Precision<f64> = Precision { lower: 1e-12, upper: 1e-8, precision: 1e-10 };
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
		assert!((result.value - 1.0).abs() < 1e-9);
	}

	#[test]
	fn test_infinite_lower_bound() {
		let result = integrate(f64::NEG_INFINITY, 0.0, |x| Ok(x.exp()), &PRECISION).unwrap();
		assert!((result.value - 1.0).abs() < 1e-9);
	}

	#[test]
	fn test_both_infinite_bounds() {
		let result = integrate(f64::NEG_INFINITY, f64::INFINITY, |x| Ok((-x * x).exp()), &PRECISION).unwrap();
		let expected = std::f64::consts::PI.sqrt();
		assert!((result.value - expected).abs() < 1e-8);
	}

	#[test]
	fn test_finite_bounds_with_integrate() {
		let result = integrate(0.0, 1.0, |x| Ok(x * x), &PRECISION).unwrap();
		assert!((result.value - 1.0 / 3.0).abs() < 1e-10);
	}
}
