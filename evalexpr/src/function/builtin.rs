#[cfg(feature = "regex")]
use regex::Regex;

use crate::{
    error::expect_function_argument_amount, value::numeric_types::EvalexprFloat, 
    EvalexprError, Function, Value, ValueType,
};

fn float_is<F: EvalexprFloat>(
    func: fn(&F) -> bool,
) -> Option<Function<F>> {
    Some(Function::new(move |s, _c| {
        Ok(func(
            &s.get_arg(0)
                .ok_or_else(|| EvalexprError::wrong_function_argument_amount(0, 1))?
                .as_float()?,
        )
        .into())
    }))
}

pub fn builtin_function<F: EvalexprFloat>(
    identifier: &str,
) -> Option<Function<F>> {
    match identifier {
        "is_nan" => float_is(F::is_nan),
        "is_finite" => float_is(F::is_finite),
        "is_infinite" => float_is(F::is_infinite),
        "is_normal" => float_is(F::is_normal),
        "if" => Some(Function::new(|s, _c| {
            expect_function_argument_amount(s.num_args(), 3)?;
            let result_index = if s.get_arg(0).unwrap().as_boolean()? {
                1
            } else {
                2
            };
            Ok(s.get_arg(result_index).unwrap().clone())
        })),
        "contains" => Some(Function::new(move |s, _c| {
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
        "contains_any" => Some(Function::new(move |s, _c| {
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
        "len" => Some(Function::new(|s, _c| {
            expect_function_argument_amount(s.num_args(), 1)?;
            // if let Ok(subject) = arguments[0].as_str() {
            //     Ok(Value::Float(F::int_as_float(
            //         &F::Int::from_usize(subject.len())?,
            //     )))
            // } else
            if let Ok(subject) = s.get_arg(0).unwrap().as_tuple_ref() {
                Ok(Value::Float(F::from_usize(subject.len())))
            } else {
                Err(EvalexprError::type_error(
                    s.get_arg(0).unwrap().clone(),
                    vec![ValueType::Tuple],
                ))
            }
        })),
        #[cfg(feature = "rand")]
        "random" => Some(Function::new(|argument| {
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
