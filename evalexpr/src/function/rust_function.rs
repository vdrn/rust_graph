use core::fmt;

#[cfg(feature = "regex")]
use regex::Regex;

use crate::error::{expect_function_argument_amount, EvalexprResultValue};
use crate::{DefaultNumericTypes, EvalexprError, EvalexprFloat, HashMapContext, Stack, Value, ValueType};

/// A helper trait to enable cloning through `Fn` trait objects.
trait ClonableFn<F: EvalexprFloat = DefaultNumericTypes>
where
	Self: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
	Self: Send + Sync + 'static, {
	fn dyn_clone(&self) -> Box<dyn ClonableFn<F>>;
}

impl<FN, F: EvalexprFloat> ClonableFn<F> for FN
where
	FN: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
	FN: Send + Sync + 'static,
	FN: Clone,
{
	fn dyn_clone(&self) -> Box<dyn ClonableFn<F>> { Box::new(self.clone()) as _ }
}

/// A user-defined function.
/// Functions can be used in expressions by storing them in a `Context`.
pub struct RustFunction<F: EvalexprFloat> {
	function: Box<dyn ClonableFn<F>>,
}

impl<F: EvalexprFloat> Clone for RustFunction<F> {
	fn clone(&self) -> Self { Self { function: self.function.dyn_clone() } }
}

impl<F: EvalexprFloat + 'static> RustFunction<F> {
	/// Creates a user-defined function.
	///
	/// The `function` is boxed for storage.
	pub fn new<FN>(function: FN) -> Self
	where
		FN: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
		FN: Send + Sync + 'static,
		FN: Clone, {
		Self { function: Box::new(function) as _ }
	}

	pub(crate) fn call(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, arguments: &[Value<F>],
	) -> EvalexprResultValue<F> {
		stack.push_args(arguments);

		let value = self.unchecked_call(stack, context);

		stack.pop_args();
		value
	}
	pub(crate) fn unchecked_call(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResultValue<F> {
		stack.function_called()?;
		let value = (self.function)(stack, context);
		stack.function_returned();
		value
	}
}

impl<F: EvalexprFloat> fmt::Debug for RustFunction<F> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> { write!(f, "Function {{ [...] }}") }
}
impl<F: EvalexprFloat> fmt::Display for RustFunction<F> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> { write!(f, "Function {{ [...] }}") }
}

/// A trait to ensure a type is `Send` and `Sync`.
/// If implemented for a type, the crate will not compile if the type is not `Send` and `Sync`.
#[allow(dead_code)]
#[doc(hidden)]
trait IsSendAndSync: Send + Sync {}

impl<F: EvalexprFloat> IsSendAndSync for RustFunction<F> {}

fn float_is<F: EvalexprFloat>(func: fn(&F) -> bool) -> Option<RustFunction<F>> {
	Some(RustFunction::new(move |s, _c| {
		Ok(func(&s.get_arg(0).ok_or_else(|| EvalexprError::wrong_function_argument_amount(0, 1))?.as_float()?)
			.into())
	}))
}

pub fn builtin_function<F: EvalexprFloat>(identifier: &str) -> Option<RustFunction<F>> {
	match identifier {
		"is_nan" => float_is(F::is_nan),
		"is_finite" => float_is(F::is_finite),
		"is_infinite" => float_is(F::is_infinite),
		"is_normal" => float_is(F::is_normal),
		// "if" => Some(RustFunction::new(|s, _c| {
		// 	expect_function_argument_amount(s.num_args(), 3)?;
		// 	let result_index = if s.get_arg(0).unwrap().as_boolean()? { 1 } else { 2 };
		// 	Ok(s.get_arg(result_index).unwrap().clone())
		// })),
		"contains" => Some(RustFunction::new(move |s, _c| {
			expect_function_argument_amount(s.num_args(), 2)?;
			if let (Value::Tuple(a), b) = (s.get_arg(0).unwrap(), s.get_arg(1).unwrap()) {
				if let Value::Float(_) | Value::Boolean(_) = b {
					Ok(a.contains(b).into())
				} else {
					Err(EvalexprError::type_error(
						b.clone(),
						vec![
							// ValueType::String,
							// ValueType::Int,
							ValueType::Float,
							ValueType::Boolean,
						],
					))
				}
			} else {
				Err(EvalexprError::expected_tuple(s.get_arg(0).unwrap().clone()))
			}
		})),
		"contains_any" => Some(RustFunction::new(move |s, _c| {
			expect_function_argument_amount(s.num_args(), 2)?;
			if let (Value::Tuple(a), b) = (&s.get_arg(0).unwrap(), s.get_arg(1).unwrap()) {
				if let Value::Tuple(b) = b {
					let mut contains = false;
					for value in b {
						if let //Value::String(_)
                        // | Value::Int(_) | 
                          Value::Float(_)
                        | Value::Boolean(_) = value
                        {
                            if a.contains(value) {
                                contains = true;
                            }
                        } else {
                            return Err(EvalexprError::type_error(
                                value.clone(),
                                vec![
                                    // ValueType::String,
                                    // ValueType::Int,
                                    ValueType::Float,
                                    ValueType::Boolean,
                                ],
                            ));
                        }
					}
					Ok(contains.into())
				} else {
					Err(EvalexprError::expected_tuple(b.clone()))
				}
			} else {
				Err(EvalexprError::expected_tuple(s.get_arg(0).unwrap().clone()))
			}
		})),
		"len" => Some(RustFunction::new(|s, _c| {
			expect_function_argument_amount(s.num_args(), 1)?;
			// if let Ok(subject) = arguments[0].as_str() {
			//     Ok(Value::Float(F::int_as_float(
			//         &F::Int::from_usize(subject.len())?,
			//     )))
			// } else
			if let Ok(subject) = s.get_arg(0).unwrap().as_tuple_ref() {
				Ok(Value::Float(F::from_usize(subject.len())))
			} else {
				Err(EvalexprError::type_error(s.get_arg(0).unwrap().clone(), vec![ValueType::Tuple]))
			}
		})),
		#[cfg(feature = "rand")]
		"random" => Some(RustFunction::new(|argument| {
			argument.as_empty()?;
			Ok(Value::Float(F::Float::random()?))
		})),
		// Bitwise operators
		// "bitand" => int_function!(bitand, 2),
		// "bitor" => int_function!(bitor, 2),
		// "bitxor" => int_function!(bitxor, 2),
		// "bitnot" => int_function!(bitnot),
		// "shl" => int_function!(bit_shift_left, 2),
		// "shr" => int_function!(bit_shift_right, 2),
		_ => None,
	}
}
