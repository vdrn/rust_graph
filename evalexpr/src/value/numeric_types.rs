use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::str::FromStr;

use crate::math::integrate::Precision;
use crate::math::{float_to_rational, gcd};
use crate::EvalexprResult;

pub mod default_numeric_types;
pub mod f32_numeric_types;

/// A float type that can be used by `evalexpr`.
pub trait EvalexprFloat:
	Clone
	+ Copy
	+ Send
	+ Sync
	+ Debug
	+ Display
	+ FromStr
	+ PartialEq
	+ PartialOrd
	+ Add<Output = Self>
	+ Sub<Output = Self>
	+ Neg<Output = Self>
	+ Mul<Output = Self>
	+ Div<Output = Self>
	+ Rem<Output = Self>
	+ 'static {
	/// Precomputed abscissas and weights for the Tanh-Sinh quadrature
	fn abscissas_and_weights() -> &'static crate::math::integrate::AbscissasWeights<Self>;

	/// The number of significant digits to display for the human-readable representation of a float.
	const HUMAN_DISPLAY_SIG_DIGITS: u32;

	/// Precision bounds for integration
	const INTEGRATION_PRECISION: Precision<Self>;

	/// The smallest non-NaN floating point value.
	///
	/// Typically, this is negative infinity.
	const MIN: Self;

	/// The largest non-NaN floating point value.
	///
	/// Typically, this is positive infinity.
	const MAX: Self;

	/// 0.5
	const ZERO: Self;
	/// 1.0
	const ONE: Self;
	/// 0.5
	const HALF: Self;
	/// epsilon
	const EPSILON: f64;
	/// usize -> Self
	fn from_usize(int: usize) -> Self;

	/// i32 -> Self
	fn from_i32(int: i32) -> Self;

	/// self -> i64
	fn into_i64(&self) -> i64;

	/// Self -> usize
	fn into_usize(&self) -> EvalexprResult<usize, Self>;

	/// u64 -> Self
	fn from_u64(int: u64) -> Self;

	/// 0x -> usize
	fn from_hex_str(literal: &str) -> Result<Self, ()>;

	///  Self > f64
	fn to_f64(&self) -> f64;
	/// f64 -> Self
	fn f64_to_float(v: f64) -> Self;

	/// Perform a power operation.
	fn pow(&self, exponent: &Self) -> Self;

	/// Compute the natural logarithm.
	fn ln(&self) -> Self;

	/// Compute the logarithm to a certain base.
	fn log(&self, base: &Self) -> Self;

	/// Compute the logarithm base 2.
	fn log2(&self) -> Self;

	/// Compute the logarithm base 10.
	fn log10(&self) -> Self;

	/// Exponentiate with base `e`.
	fn exp(&self) -> Self;

	/// Exponentiate with base 2.
	fn exp2(&self) -> Self;

	/// Compute the cosine.
	fn cos(&self) -> Self;

	/// Compute the hyperbolic cosine.
	fn cosh(&self) -> Self;

	/// Compute the arccosine.
	fn acos(&self) -> Self;

	/// Compute the hyperbolic arccosine.
	fn acosh(&self) -> Self;

	/// Compute the sine.
	fn sin(&self) -> Self;

	/// Compute the hyperbolic sine.
	fn sinh(&self) -> Self;

	/// Compute the arcsine.
	fn asin(&self) -> Self;

	/// Compute the hyperbolic arcsine.
	fn asinh(&self) -> Self;

	/// Compute the tangent.
	fn tan(&self) -> Self;

	/// Compute the hyperbolic tangent.
	fn tanh(&self) -> Self;

	/// Compute the arctangent.
	fn atan(&self) -> Self;

	/// Compute the hyperbolic arctangent.
	fn atanh(&self) -> Self;

	/// Compute the four quadrant arctangent.
	fn atan2(&self, x: &Self) -> Self;

	/// Compute the square root.
	fn sqrt(&self) -> Self;

	/// Compute the cubic root.
	fn cbrt(&self) -> Self;

	/// Compute the distance between the origin and a point (`self`, `other`) on the Euclidean plane.
	fn hypot(&self, other: &Self) -> Self;

	/// Compute the largest integer less than or equal to `self`.
	fn floor(&self) -> Self;

	/// Returns the nearest integer to `self`. If a value is half-way between two integers, round away from
	/// `0.0`.
	fn round(&self) -> Self;

	/// Compute the largest integer greater than or equal to `self`.
	fn ceil(&self) -> Self;

	/// Returns true if `self` is not a number.
	fn is_nan(&self) -> bool;

	/// Returns true if `self` is finite.
	fn is_finite(&self) -> bool;

	/// Returns true if `self` is infinite.
	fn is_infinite(&self) -> bool;

	/// Returns true if `self` is normal.
	fn is_normal(&self) -> bool;

	/// Returns the absolute value of self.
	fn abs(&self) -> Self;

	/// Returns the minimum of the two numbers, ignoring NaN.
	fn min(&self, other: &Self) -> Self;

	/// Returns the maximum of the two numbers, ignoring NaN.
	fn max(&self, other: &Self) -> Self;

	/// Returns a number that represents the sign of `self`.
	fn signum(&self) -> Self;

	/// Clamps `self` between `min` and `max`.
	fn clamp(&self, min: &Self, max: &Self) -> Self;

	/// Reciprocal.
	fn recip(&self) -> Self;

	/// Returns the factorial of `self`.
	fn factorial(&self) -> Self;
	/// Generate a random float value between 0.0 and 1.0.
	///
	/// If the feature `rand` is not enabled, then this method always returns
	/// [`EvalexprError::RandNotEnabled`](crate::EvalexprError::RandNotEnabled).
	fn random() -> EvalexprResult<Self, Self>;

	/// Greatest common denominator.
	/// Values are rounded to nearest integer
	fn gcd(&self, other: &Self) -> Self {
		let a = self.into_i64();
		let b = other.into_i64();
		let a = a.unsigned_abs();
		let b = b.unsigned_abs();
		Self::from_u64(gcd::gcd(a, b))
	}

	/// Returns a tuple of the rational representation of `self` and the denominator.
	/// This will not be the exact float representation, but the slightly rounded value.
	/// For f64, we round to 14 significant digits, and for f32, we round to 6.
	fn rational_display(&self) -> (i64, u64) {
		float_to_rational::f64_to_rational_display(self.to_f64(), Self::HUMAN_DISPLAY_SIG_DIGITS)
	}

	/// Returns a string representation of `self` for human display.
	fn human_display(&self, rational: bool) -> String {
		if rational {
			let (numerator, denominator) = self.rational_display();
			format!("{}/{}", numerator, denominator)
		} else {
			let decimal_places =
				float_to_rational::decimal_places(self.to_f64(), Self::HUMAN_DISPLAY_SIG_DIGITS);
			format!("{:.prec$}", self.to_f64(), prec = decimal_places as usize)
				.trim_end_matches('0')
				.trim_end_matches('.')
				.to_string()
		}
	}
}
