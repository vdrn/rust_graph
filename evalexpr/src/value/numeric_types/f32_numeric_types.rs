use std::sync::OnceLock;

use crate::{EvalexprError, EvalexprResult, math::integrate::Precision};

use super::EvalexprFloat;

/// The type used to represent floats in `Evalexpr`.
pub type F32NumericTypes = f32;

impl EvalexprFloat for f32 {
    const MIN: Self = Self::NEG_INFINITY;
    const MAX: Self = Self::INFINITY;

    const ZERO: Self = 0.0;
    const HALF: Self = 0.5;
    const ONE: Self = 1.0;
    const EPSILON: f64 = f64::EPSILON;
    fn from_usize(int: usize) -> Self {
        int as Self
    }
    const INTEGRATION_PRECISION: Precision<Self> = Precision {
        lower: 1e-7,
        upper: 1e-5,
        precision: 1e-6,
    };
    fn abscissas_and_weights() -> &'static crate::math::integrate::AbscissasWeights<Self> {
        static PRECOMPUTED: OnceLock<crate::math::integrate::AbscissasWeights<f32>> =
            OnceLock::new();
        PRECOMPUTED.get_or_init(crate::math::integrate::get_tanh_sinh_abscissas_and_weights)
    }

    fn into_usize(&self) -> EvalexprResult<usize, Self> {
        if *self >= 0.0 {
            Ok(*self as usize)
        } else {
            Err(EvalexprError::FloatIntoUsize { float: *self })
        }
    }
    fn from_i32(int: i32) -> Self {
        int as Self
    }

    fn from_hex_str(literal: &str) -> Result<Self, ()> {
        i64::from_str_radix(literal, 16)
            .map(|i| i as Self)
            .map_err(|_| ())
    }

    fn to_f64(&self) -> f64 {
        *self as f64
    }
    fn f64_to_float(v: f64) -> Self {
        v as f32
    }
    fn pow(&self, exponent: &Self) -> Self {
        (*self).powf(*exponent)
    }

    fn ln(&self) -> Self {
        (*self).ln()
    }

    fn log(&self, base: &Self) -> Self {
        (*self).log(*base)
    }

    fn log2(&self) -> Self {
        (*self).log2()
    }

    fn log10(&self) -> Self {
        (*self).log10()
    }

    fn exp(&self) -> Self {
        (*self).exp()
    }

    fn exp2(&self) -> Self {
        (*self).exp2()
    }

    fn cos(&self) -> Self {
        (*self).cos()
    }

    fn cosh(&self) -> Self {
        (*self).cosh()
    }

    fn acos(&self) -> Self {
        (*self).acos()
    }

    fn acosh(&self) -> Self {
        (*self).acosh()
    }

    fn sin(&self) -> Self {
        (*self).sin()
    }

    fn sinh(&self) -> Self {
        (*self).sinh()
    }

    fn asin(&self) -> Self {
        (*self).asin()
    }

    fn asinh(&self) -> Self {
        (*self).asinh()
    }

    fn tan(&self) -> Self {
        (*self).tan()
    }

    fn tanh(&self) -> Self {
        (*self).tanh()
    }

    fn atan(&self) -> Self {
        (*self).atan()
    }

    fn atanh(&self) -> Self {
        (*self).atanh()
    }

    fn atan2(&self, x: &Self) -> Self {
        (*self).atan2(*x)
    }

    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }

    fn cbrt(&self) -> Self {
        (*self).cbrt()
    }

    fn hypot(&self, other: &Self) -> Self {
        (*self).hypot(*other)
    }

    fn floor(&self) -> Self {
        (*self).floor()
    }

    fn round(&self) -> Self {
        (*self).round()
    }

    fn ceil(&self) -> Self {
        (*self).ceil()
    }

    fn is_nan(&self) -> bool {
        (*self).is_nan()
    }

    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }

    fn is_infinite(&self) -> bool {
        (*self).is_infinite()
    }

    fn is_normal(&self) -> bool {
        (*self).is_normal()
    }

    fn abs(&self) -> Self {
        (*self).abs()
    }

    fn min(&self, other: &Self) -> Self {
        (*self).min(*other)
    }

    fn max(&self, other: &Self) -> Self {
        (*self).max(*other)
    }

    fn clamp(&self, min: &Self, max: &Self) -> Self {
        (*self).clamp(*min, *max)
    }

    fn random() -> EvalexprResult<Self, f32> {
        #[cfg(feature = "rand")]
        let result = Ok(rand::random());

        #[cfg(not(feature = "rand"))]
        let result = Err(EvalexprError::RandNotEnabled);

        result
    }

    fn signum(&self) -> Self {
        (*self).signum()
    }

    fn factorial(&self) -> Self {
        <f64 as special::Gamma>::gamma(*self as f64 + 1.0) as Self
    }
    fn recip(&self) -> Self {
        (*self).recip()
    }
}
