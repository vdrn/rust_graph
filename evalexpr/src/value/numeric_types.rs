use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Rem, Sub},
    str::FromStr,
};

use crate::{EvalexprResult, math::integrate::Precision};

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
    + 'static
{
    /// Precomputed abscissas and weights for the Tanh-Sinh quadrature
    fn abscissas_and_weights() -> &'static crate::math::integrate::AbscissasWeights<Self>;

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

    /// Self -> usize
    fn into_usize(&self) -> EvalexprResult<usize, Self>;

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

    /// Returns the nearest integer to `self`. If a value is half-way between two integers, round away from `0.0`.
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
    /// If the feature `rand` is not enabled, then this method always returns [`EvalexprError::RandNotEnabled`](crate::EvalexprError::RandNotEnabled).
    fn random() -> EvalexprResult<Self, Self>;
}
