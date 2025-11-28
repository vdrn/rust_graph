#![cfg(not(tarpaulin_include))]

use evalexpr::error::*;
use evalexpr::{istr, *};
use std::convert::TryFrom;
use thin_vec::thin_vec;

#[track_caller]
fn test_eval(string: &str, expected: EvalexprResultValue) {
	// test unoptimizd flat node
	assert_eq!(eval(string), expected.clone());
	// test optimized flat node
	assert_eq!(eval_optimized(string), expected.clone());
}
#[track_caller]
fn test_eval_and_fold_to_const(string: &str, expected: EvalexprResultValue) {
	let mut ctx = HashMapContext::<DefaultNumericTypes>::new();

	// test unoptimizd flat node
	assert_eq!(eval_with_context(string, &ctx), expected.clone());

	match build_optimized_flat_node(string, &mut ctx) {
		Ok(optimized) => {
			// test optimized and folded flat node
			assert_eq!(optimized.eval(), expected.clone());

			// make sure optimization folded a flat node to a constant
			assert_eq!(
				optimized.as_constant(),
				Some(expected.clone().unwrap()),
				"Node should be optimized to a single constant, got {:?}",
				optimized
			);
		},
		Err(err) => {
			assert_eq!(expected, Err(err));
		},
	}
}

#[track_caller]
fn test_eval_with_context(string: &str, ctx: &HashMapContext, expected: EvalexprResultValue) {
	// test unoptimized flat node
	assert_eq!(eval_with_context(string, ctx), expected.clone());

	// test optimized flat node without inlining context
	match build_optimized_flat_node(string, &mut HashMapContext::<DefaultNumericTypes>::new()) {
		Ok(optimized) => {
			let mut stack = Stack::new();
			assert_eq!(optimized.eval_with_context(&mut stack, ctx), expected.clone());
		},
		Err(err) => {
			assert_eq!(expected, Err(err));
		},
	}

	// test optimized flat node with inlining context
	let mut cloned_ctx = ctx.clone();
	match build_optimized_flat_node(string, &mut cloned_ctx) {
		Ok(optimized) => {
			assert_eq!(optimized.eval(), expected.clone());
		},
		Err(err) => {
			assert_eq!(expected, Err(err));
		},
	}

	assert!(
		HashMapContext::contexts_almost_equal(&cloned_ctx, ctx),
		"Context should not change during optimization"
	);
}
#[track_caller]
fn test_eval_and_fold_to_const_with_context(
	string: &str, ctx: &HashMapContext, expected: EvalexprResultValue,
) {
	// test unoptimized flat node
	assert_eq!(eval_with_context(string, ctx), expected.clone());

	// test optimized flat node without inlining context
	match build_optimized_flat_node(string, &mut HashMapContext::<DefaultNumericTypes>::new()) {
		Ok(optimized) => {
			let mut stack = Stack::new();
			assert_eq!(optimized.eval_with_context(&mut stack, ctx), expected.clone());
		},
		Err(err) => {
			assert_eq!(expected, Err(err));
		},
	}

	// test optimized flat node with inlining context
	let mut cloned_ctx = ctx.clone();
	match build_optimized_flat_node(string, &mut cloned_ctx) {
		Ok(optimized) => {
			assert_eq!(optimized.eval(), expected.clone());

			// Make sure the final node is just a constant
			assert_eq!(
				optimized.as_constant(),
				Some(expected.clone().unwrap()),
				"Node should be optimized to a single constant, got {:?}",
				optimized
			);
		},
		Err(err) => {
			assert_eq!(expected, Err(err));
		},
	}
	assert!(
		HashMapContext::contexts_almost_equal(&cloned_ctx, ctx),
		"Context should not change during optimization"
	);
}

#[track_caller]
fn test_fold_to_const_expr_and_func(
	string: &str, args: &[(IStr, f64)], ctx: &mut HashMapContext, expected: EvalexprResultValue,
) {
	let mut fn_args = Vec::with_capacity(args.len());
	let mut fn_arg_values = String::with_capacity(args.len() * 3);
	for (i, (arg_name, arg_value)) in args.iter().enumerate() {
		// set the args in the context so the raw expression witohut function wrapping works
		ctx.set_value(*arg_name, Value::Float(*arg_value)).unwrap();

		fn_args.push(*arg_name);
		fn_arg_values.push_str(arg_value.to_string().as_str());
		if i != args.len() - 1 {
			fn_arg_values.push_str(", ");
		}
	}

	// test as expression
	test_eval_and_fold_to_const_with_context(string, ctx, expected.clone());

	// test as function
	let expr = build_optimized_flat_node(string, ctx).unwrap();
	let func = ExpressionFunction::new(expr, &fn_args, &mut Some(ctx)).unwrap();
	// indirectly call the function to check more code paths
	ctx.set_expression_function(istr("func"), func);
	test_eval_and_fold_to_const_with_context(&format!("func({fn_arg_values})"), ctx, expected);
}

#[test]
fn test_sum_operator() {
	let mut ctx = HashMapContext::<DefaultNumericTypes>::new();
	// number of iterations
	ctx.set_value(istr("c"), Value::Float(7.0)).unwrap();

	let args = &[(istr("x"), -2.7)];
	// approx sin(-2.7)
	let expected = Value::Float(-0.4273798209522364);
	test_fold_to_const_expr_and_func(
		// taylor series for sin(x) with `c` iterations
		"Sum(0..c, n, (-1)^n * x^(2*n+1) / (2*n+1)!)",
		args,
		&mut ctx,
		Ok(expected),
	);
}

#[test]
fn test_postfix() {
	test_eval_and_fold_to_const("5!", Ok(Value::Float(120.0)));
	test_eval_and_fold_to_const("1+2!", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("1+(2)!", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("(1+(2))!", Ok(Value::Float(6.0)));
	test_eval_and_fold_to_const("(1+2)!", Ok(Value::Float(6.0)));
	test_eval_and_fold_to_const("((1+2))!", Ok(Value::Float(6.0)));
	test_eval_and_fold_to_const("((1,2)*(3,4)).x", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("((1,2)*(3,4)).y", Ok(Value::Float(8.0)));
}

#[test]
fn test_unary_examples() {
	test_eval_and_fold_to_const("3", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("3.3", Ok(Value::Float(3.3)));
	test_eval_and_fold_to_const("true", Ok(Value::Boolean(true)));
	test_eval_and_fold_to_const("false", Ok(Value::Boolean(false)));
	test_eval("blub", Err(EvalexprError::VariableIdentifierNotFound("blub".to_string())));
	test_eval_and_fold_to_const("-3", Ok(Value::Float(-3.0)));
	test_eval_and_fold_to_const("-3.6", Ok(Value::Float(-3.6)));
	test_eval_and_fold_to_const("----3", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("1e0", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("1e-0", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("10e3", Ok(Value::Float(10000.0)));
	test_eval_and_fold_to_const("10e+3", Ok(Value::Float(10000.0)));
	test_eval_and_fold_to_const("10e-3", Ok(Value::Float(0.01)));
}

#[test]
fn test_binary_examples() {
	test_eval_and_fold_to_const("1+3", Ok(Value::Float(4.0)));
	test_eval_and_fold_to_const("3+1", Ok(Value::Float(4.0)));
	test_eval_and_fold_to_const("3-5", Ok(Value::Float(-2.0)));
	test_eval_and_fold_to_const("5-3", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("5 / 4", Ok(Value::Float(1.25)));
	test_eval_and_fold_to_const("5 *3", Ok(Value::Float(15.0)));
	test_eval_and_fold_to_const("1.0+3", Ok(Value::Float(4.0)));
	test_eval_and_fold_to_const("3.0+1", Ok(Value::Float(4.0)));
	test_eval_and_fold_to_const("3-5.0", Ok(Value::Float(-2.0)));
	test_eval_and_fold_to_const("5-3.0", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("5 / 4.0", Ok(Value::Float(1.25)));
	test_eval_and_fold_to_const("5.0 *3", Ok(Value::Float(15.0)));
	test_eval_and_fold_to_const("5.0 *-3", Ok(Value::Float(-15.0)));
	test_eval_and_fold_to_const("5.0 *- 3", Ok(Value::Float(-15.0)));
	test_eval_and_fold_to_const("5.0 * -3", Ok(Value::Float(-15.0)));
	test_eval_and_fold_to_const("5.0 * - 3", Ok(Value::Float(-15.0)));
	test_eval_and_fold_to_const("-5.0 *-3", Ok(Value::Float(15.0)));
	test_eval_and_fold_to_const("3+-1", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("-3-5", Ok(Value::Float(-8.0)));
	test_eval_and_fold_to_const("-5--3", Ok(Value::Float(-2.0)));
	test_eval_and_fold_to_const("5e2--3", Ok(Value::Float(503.0)));
	test_eval_and_fold_to_const("-5e-2--3", Ok(Value::Float(2.95)));
}

#[test]
fn test_arithmetic_precedence_examples() {
	test_eval_and_fold_to_const("1+3-2", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("3+1*5", Ok(Value::Float(8.0)));
	test_eval_and_fold_to_const("2*3-5", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("5-3/3", Ok(Value::Float(4.0)));
	test_eval_and_fold_to_const("5 / 4*2", Ok(Value::Float(2.5)));
	test_eval_and_fold_to_const("1-5 *3/15", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("15/8/2.0", Ok(Value::Float(0.9375)));
	test_eval_and_fold_to_const("15.0/7/2", Ok(Value::Float(15.0 / 7.0 / 2.0)));
	test_eval_and_fold_to_const("15.0/-7/2", Ok(Value::Float(15.0 / -7.0 / 2.0)));
	test_eval_and_fold_to_const("-15.0/7/2", Ok(Value::Float(-15.0 / 7.0 / 2.0)));
	test_eval_and_fold_to_const("-15.0/7/-2", Ok(Value::Float(-15.0 / 7.0 / -2.0)));
}

#[test]
fn test_braced_examples() {
	test_eval_and_fold_to_const("(1)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("( 1.0 )", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("( true)", Ok(Value::Boolean(true)));
	test_eval_and_fold_to_const("( -1 )", Ok(Value::Float(-1.0)));
	test_eval_and_fold_to_const("-(1)", Ok(Value::Float(-1.0)));
	test_eval_and_fold_to_const("-(1 + 3) * 7", Ok(Value::Float(-28.0)));
	test_eval_and_fold_to_const("(1 * 1) - 3", Ok(Value::Float(-2.0)));
	test_eval_and_fold_to_const("4 / (2 * 2)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("7/(7/(7/(7/(7/(7)))))", Ok(Value::Float(1.0)));
}

#[test]
fn test_mod_examples() {
	test_eval_and_fold_to_const("1 % 4", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("6 % 4", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("1 % 4 + 2", Ok(Value::Float(3.0)));
}

#[test]
fn test_pow_examples() {
	test_eval_and_fold_to_const("1 ^ 4", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("6 ^ 4", Ok(Value::Float((6.0 as DefaultNumericTypes).powf(4.0))));
	test_eval_and_fold_to_const("1 ^ 4 + 2", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("2 ^ (4 + 2)", Ok(Value::Float(64.0)));
}

#[test]
fn test_boolean_examples() {
	test_eval("true && false", Ok(Value::Boolean(false)));
	test_eval("true && false || true && true", Ok(Value::Boolean(true)));
	test_eval("5 > 4 && 1 <= 1", Ok(Value::Boolean(true)));
	test_eval("5.0 <= 4.9 || !(4 > 3.5)", Ok(Value::Boolean(false)));
}

#[test]
fn test_with_context() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	context.set_value(istr("tr"), Value::Boolean(true)).unwrap();
	context.set_value(istr("fa"), Value::Boolean(false)).unwrap();
	context.set_value(istr("five"), Value::Float(5.0)).unwrap();
	context.set_value(istr("six"), Value::Float(6.0)).unwrap();
	context.set_value(istr("half"), Value::Float(0.5)).unwrap();
	context.set_value(istr("zero"), Value::Float(0.0)).unwrap();

	test_eval_with_context("tr", &context, Ok(Value::Boolean(true)));
	test_eval_with_context("fa", &context, Ok(Value::Boolean(false)));
	test_eval_with_context("tr && false", &context, Ok(Value::Boolean(false)));
	test_eval_with_context("five + six", &context, Ok(Value::Float(11.0)));
	test_eval_with_context("five * half", &context, Ok(Value::Float(2.5)));
	test_eval_with_context("five < six && true", &context, Ok(Value::Boolean(true)));

	assert_eq!(context.remove_value(istr("half")), Ok(Some(Value::Float(0.5))));
	assert_eq!(context.remove_value(istr("zero")), Ok(Some(Value::Float(0.0))));
	assert_eq!(context.remove_value(istr("zero")), Ok(None));

	test_eval_with_context(
		"zero",
		&context,
		Err(EvalexprError::VariableIdentifierNotFound("zero".to_string())),
	);
}

#[test]
fn test_functions() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	context.set_function(
		istr("sub2"),
		RustFunction::new(|s, _| {
			// if let Value::Int(int) = argument {
			//     Ok(Value::Int(int - 2))
			// } else
			let arg = s.get_arg(0).unwrap();
			if let Value::Float(float) = arg {
				Ok(Value::Float(float - 2.0))
			} else {
				Err(EvalexprError::expected_float(arg.clone()))
			}
		}),
	);
	context.set_value(istr("five"), Value::Float(5.0)).unwrap();

	assert_eq!(eval_with_context("sub2 5", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("sub2(5)", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("sub2 five", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("sub2(five)", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("sub2(3) + five", &context), Ok(Value::Float(6.0)));
}

#[test]
fn test_n_ary_functions() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	context.set_function(
		istr("sub2"),
		RustFunction::new(|s, _| {
			// if let Value::Int(int) = argument {
			//     Ok(Value::Int(int - 2))
			// } else
			let arg = s.get_arg(0).unwrap();
			if let Value::Float(float) = arg {
				Ok(Value::Float(float - 2.0))
			} else {
				Err(EvalexprError::expected_float(arg.clone()))
			}
		}),
	);
	context.set_function(
		istr("avg"),
		RustFunction::new(|s, _| {
			expect_function_argument_amount(s.num_args(), 2)?;
			let a1 = s.get_arg(0).unwrap().as_float()?;
			let a2 = s.get_arg(1).unwrap().as_float()?;

			Ok(Value::Float((a1 + a2) / 2.0))
		}),
	);

	context.set_function(
		istr("muladd"),
		RustFunction::new(|s, _| {
			expect_function_argument_amount(s.num_args(), 3)?;
			let a = s.get_arg(0).unwrap().as_float()?;
			let b = s.get_arg(1).unwrap().as_float()?;
			let c = s.get_arg(2).unwrap().as_float()?;

			Ok(Value::Float(a * b + c))
		}),
	);
	context.set_function(
		istr("count"),
		RustFunction::new(|s, _| {
			if s.num_args() == 1 {
				match &s.get_arg(0).unwrap() {
					Value::Tuple(tuple) => Ok(Value::from_float(DefaultNumericTypes::from_usize(tuple.len()))),
					Value::Empty => Ok(Value::from_float(0.0)),
					_ => Ok(Value::from_float(1.0)),
				}
			} else {
				Ok(Value::Float(s.num_args() as f64))
			}
		}),
	);
	context.set_value(istr("five"), Value::Float(5.0)).unwrap();
	context.set_function(istr("function_four"), RustFunction::new(|_, _| Ok(Value::Float(4.0))));

	assert_eq!(eval_with_context("avg(7, 5)", &context), Ok(Value::Float(6.0)));
	assert_eq!(eval_with_context("avg(sub2 5, 5)", &context), Ok(Value::Float(4.0)));
	assert_eq!(eval_with_context("sub2(avg(3, 6))", &context), Ok(Value::Float(2.5)));
	assert_eq!(eval_with_context("muladd(3, 6, -4)", &context), Ok(Value::Float(14.0)));
	assert_eq!(eval_with_context("count()", &context), Ok(Value::Float(0.0)));
	assert_eq!(eval_with_context("count((1, 2, 3))", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("count(3, 5.5, 2)", &context), Ok(Value::Float(3.0)));
	assert_eq!(eval_with_context("count 5", &context), Ok(Value::Float(1.0)));
	assert_eq!(eval_with_context("function_four()", &context), Ok(Value::Float(4.0)));
}

#[test]
fn test_capturing_functions() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	// this variable is captured by the function
	let three = 3;
	context.set_function(
		istr("mult_3"),
		RustFunction::new(move |s, _| {
			if let Value::Float(float) = s.get_arg(0).unwrap() {
				Ok(Value::Float(float * three as DefaultNumericTypes))
			} else {
				Err(EvalexprError::expected_float(s.get_arg(0).unwrap().clone()))
			}
		}),
	);

	let four = 4.0;
	context.set_function(istr("function_four"), RustFunction::new(move |_, _| Ok(Value::Float(four))));

	assert_eq!(eval_with_context("mult_3 2", &context), Ok(Value::Float(6.0)));
	assert_eq!(eval_with_context("mult_3(3)", &context), Ok(Value::Float(9.0)));
	assert_eq!(eval_with_context("mult_3(function_four())", &context), Ok(Value::Float(12.0)));
}

#[test]
fn test_builtin_functions() {
	// Log
	test_eval_and_fold_to_const("ln(2.718281828459045)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("log(9, 9)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("log2(2)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("log10(10)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("exp(2)", Ok(Value::Float((2.0 as DefaultNumericTypes).exp())));
	test_eval_and_fold_to_const("exp2(2)", Ok(Value::Float((2.0 as DefaultNumericTypes).exp2())));
	test_eval_and_fold_to_const("cos(0)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("acos(1)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("cosh(0)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("acosh(1)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("sin(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("asin(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("sinh(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("asinh(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("tan(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("atan(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("tanh(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const("atanh(0)", Ok(Value::Float(0.0)));
	test_eval_and_fold_to_const(
		"atan2(1.2, -5.5)",
		Ok(Value::Float((1.2 as DefaultNumericTypes).atan2(-5.5))),
	);
	test_eval_and_fold_to_const("sqrt(25)", Ok(Value::Float(5.0)));
	test_eval_and_fold_to_const("cbrt(8)", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("hypot(8.2, 1.1)", Ok(Value::Float((8.2 as DefaultNumericTypes).hypot(1.1))));
	test_eval_and_fold_to_const("floor(1.1)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("floor(1.9)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("round(1.1)", Ok(Value::Float(1.0)));
	test_eval_and_fold_to_const("round(1.5)", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("round(2.5)", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("round(1.9)", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("ceil(1.1)", Ok(Value::Float(2.0)));
	test_eval_and_fold_to_const("ceil(1.9)", Ok(Value::Float(2.0)));
	test_eval("is_nan(1.0/0.0)", Ok(Value::Boolean(false)));
	test_eval("is_nan(0.0/0.0)", Ok(Value::Boolean(true)));
	test_eval("is_finite(1.0/0.0)", Ok(Value::Boolean(false)));
	test_eval("is_finite(0.0/0.0)", Ok(Value::Boolean(false)));
	test_eval("is_finite(0.0)", Ok(Value::Boolean(true)));
	test_eval("is_infinite(0.0/0.0)", Ok(Value::Boolean(false)));
	test_eval("is_infinite(1.0/0.0)", Ok(Value::Boolean(true)));
	test_eval("is_normal(1.0/0.0)", Ok(Value::Boolean(false)));
	test_eval("is_normal(0)", Ok(Value::Boolean(false)));
	test_eval_and_fold_to_const("abs(15.4)", Ok(Value::Float(15.4)));
	test_eval_and_fold_to_const("abs(-15.4)", Ok(Value::Float(15.4)));
	test_eval_and_fold_to_const("abs(15)", Ok(Value::Float(15.0)));
	test_eval_and_fold_to_const("abs(-15)", Ok(Value::Float(15.0)));
	test_eval_and_fold_to_const("min(4.0, 3)", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("max(4.0, 3)", Ok(Value::Float(4.0)));
	test_eval(
		"contains(1, 2, 3)",
		Err(EvalexprError::WrongFunctionArgumentAmount { expected: 2..=2, actual: 3 }),
	);
	assert_eq!(eval("contains_any((1,2,3), (3,4,5))"), Ok(Value::Boolean(true)));
	assert_eq!(eval("contains_any((1,2,3), (4,5,6))"), Ok(Value::Boolean(false)));
	assert_eq!(
		eval("contains_any((true, false, true, true), (false, false, false))"),
		Ok(Value::Boolean(true))
	);
}

fn error_expected_numeric(actual: Value<DefaultNumericTypes>, name: &'static str) -> EvalexprError {
	EvalexprError::wrong_type_combination(
		name,
		vec![(&actual).into()],
		vec![ValueType::Float, ValueType::Float2, ValueType::Tuple],
	)
}
#[test]
fn test_errors() {
	assert_eq!(eval("-true"), Err(error_expected_numeric(Value::Boolean(true), "-")));
	assert_eq!(eval("true-"), Err(EvalexprError::WrongOperatorArgumentAmount { actual: 1, expected: 2 }));
	assert_eq!(eval("!(()true)"), Err(EvalexprError::AppendedToLeafNode));
	// assert_eq!(
	//     eval("math::is_nan(\"xxx\")"),
	//     Err(EvalexprError::ExpectedFloat {
	//         actual: Value::String("xxx".to_string())
	//     })
	// );
}

#[test]
fn test_no_panic() {
	//     assert!(eval(&format!(
	//         "{} + {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "-{} - {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "-(-{} - 1)",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "{} * {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "{} / {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         0
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "{} % {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         0
	//     ))
	//     .is_err());
	//     assert!(eval(&format!(
	//         "{} ^ {}",
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX,
	//         <DefaultNumericTypes as EvalexprNumericTypes>::Int::MAX
	//     ))
	//     .is_ok());
	assert!(eval("if").is_err());
	assert!(eval("if()").is_err());
	assert!(eval("if(true, 1)").is_err());
	assert!(eval("if(false, 2)").is_err());
	assert!(eval("if(1,1,1)").is_err());
	assert!(eval("if(true,1,1,1)").is_err());
}

#[test]
fn test_shortcut_functions() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	// context
	//     .set_value("string".into(), Value::from("a string"))
	//     .unwrap();

	// assert_eq!(eval_string("\"3.3\""), Ok("3.3".to_owned()));
	// assert_eq!(
	//     eval_string("3.3"),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     eval_string("3..3"),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );
	// assert_eq!(
	//     eval_string_with_context("string", &context),
	//     Ok("a string".to_owned())
	// );
	// assert_eq!(
	//     eval_string_with_context("3.3", &context),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     eval_string_with_context("3..3", &context),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );
	// assert_eq!(
	//     eval_string_with_context_mut("string", &mut context),
	//     Ok("a string".to_string())
	// );
	// assert_eq!(
	//     eval_string_with_context_mut("3.3", &mut context),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     eval_string_with_context_mut("3..3", &mut context),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );

	assert_eq!(eval_float("3.3"), Ok(3.3));
	assert_eq!(eval_float("asd()"), Err(EvalexprError::FunctionIdentifierNotFound("asd".to_owned())));
	assert_eq!(eval_float_with_context("3.3", &context), Ok(3.3));
	assert_eq!(eval_float_with_context("asd)", &context), Err(EvalexprError::UnmatchedRBrace));
	assert_eq!(eval_float_with_context_mut("3.3", &mut context), Ok(3.3));
	assert_eq!(eval_float_with_context_mut("asd(", &mut context), Err(EvalexprError::UnmatchedLBrace));

	assert_eq!(eval_float("3"), Ok(3.0));
	assert_eq!(eval_float("true"), Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) }));
	assert_eq!(eval_float("abc"), Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned())));
	assert_eq!(eval_float_with_context("3.5", &context), Ok(3.5));
	assert_eq!(eval_float_with_context("3", &context), Ok(3.0));
	assert_eq!(
		eval_float_with_context("true", &context),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		eval_float_with_context("abc", &context),
		Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned()))
	);
	assert_eq!(eval_float_with_context_mut("3.5", &mut context), Ok(3.5));
	assert_eq!(eval_float_with_context_mut("3", &mut context), Ok(3.0));
	assert_eq!(
		eval_float_with_context_mut("true", &mut context),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		eval_float_with_context_mut("abc", &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned()))
	);

	assert_eq!(eval_boolean("true"), Ok(true));
	assert_eq!(eval_boolean("4"), Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) }));
	assert_eq!(eval_boolean("trueee"), Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned())));
	assert_eq!(eval_boolean_with_context("true", &context), Ok(true));
	assert_eq!(
		eval_boolean_with_context("4", &context),
		Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) })
	);
	assert_eq!(
		eval_boolean_with_context("trueee", &context),
		Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned()))
	);
	assert_eq!(eval_boolean_with_context_mut("true", &mut context), Ok(true));
	assert_eq!(
		eval_boolean_with_context_mut("4", &mut context),
		Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) })
	);
	assert_eq!(
		eval_boolean_with_context_mut("trueee", &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned()))
	);

	assert_eq!(eval_tuple("33"), Err(EvalexprError::ExpectedTuple { actual: Value::Float(33.0) }));
	assert_eq!(eval_tuple("3a3"), Err(EvalexprError::VariableIdentifierNotFound("3a3".to_owned())));
	assert_eq!(
		eval_tuple_with_context("33", &context),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(33.0) })
	);
	assert_eq!(
		eval_tuple_with_context("3a3", &context),
		Err(EvalexprError::VariableIdentifierNotFound("3a3".to_owned()))
	);

	assert_eq!(eval_empty(""), Ok(EMPTY_VALUE));
	assert_eq!(eval_empty("()"), Ok(EMPTY_VALUE));
	assert_eq!(
		eval_empty("(,)"),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(eval_empty("xaq"), Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned())));
	assert_eq!(eval_empty_with_context("", &context), Ok(EMPTY_VALUE));
	assert_eq!(eval_empty_with_context("()", &context), Ok(EMPTY_VALUE));
	assert_eq!(
		eval_empty_with_context("(,)", &context),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(
		eval_empty_with_context("xaq", &context),
		Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned()))
	);
	assert_eq!(eval_empty_with_context_mut("", &mut context), Ok(EMPTY_VALUE));
	assert_eq!(eval_empty_with_context_mut("()", &mut context), Ok(EMPTY_VALUE));
	assert_eq!(
		eval_empty_with_context_mut("(,)", &mut context),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(
		eval_empty_with_context_mut("xaq", &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned()))
	);

	// With detour via build_flat_node

	// assert_eq!(
	//     build_flat_node::<DefaultNumericTypes>("\"3.3\"")
	//         .unwrap()
	//         .eval_string(),
	//     Ok("3.3".to_owned())
	// );
	// assert_eq!(
	//     build_flat_node::<DefaultNumericTypes>("3.3")
	//         .unwrap()
	//         .eval_string(),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     build_flat_node::<DefaultNumericTypes>("3..3")
	//         .unwrap()
	//         .eval_string(),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );
	// assert_eq!(
	//     build_flat_node("string")
	//         .unwrap()
	//         .eval_string_with_context(&context),
	//     Ok("a string".to_owned())
	// );
	// assert_eq!(
	//     build_flat_node("3.3")
	//         .unwrap()
	//         .eval_string_with_context(&context),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     build_flat_node("3..3")
	//         .unwrap()
	//         .eval_string_with_context(&context),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );
	// assert_eq!(
	//     build_flat_node("string")
	//         .unwrap()
	//         .eval_string_with_context_mut(&mut context),
	//     Ok("a string".to_string())
	// );
	// assert_eq!(
	//     build_flat_node("3.3")
	//         .unwrap()
	//         .eval_string_with_context_mut(&mut context),
	//     Err(EvalexprError::ExpectedString {
	//         actual: Value::Float(3.3)
	//     })
	// );
	// assert_eq!(
	//     build_flat_node("3..3")
	//         .unwrap()
	//         .eval_string_with_context_mut(&mut context),
	//     Err(EvalexprError::VariableIdentifierNotFound("3..3".to_owned()))
	// );
	let mut stack = Stack::new();

	assert_eq!(build_flat_node::<DefaultNumericTypes>("3.3").unwrap().eval_float(), Ok(3.3));
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("asd()").unwrap().eval_float(),
		Err(EvalexprError::FunctionIdentifierNotFound("asd".to_owned()))
	);
	assert_eq!(build_flat_node("3.3").unwrap().eval_float_with_context(&mut stack, &context), Ok(3.3));
	assert_eq!(
		build_flat_node("asd").unwrap().eval_float_with_context(&mut stack, &context),
		Err(EvalexprError::VariableIdentifierNotFound("asd".to_owned()))
	);
	assert_eq!(build_flat_node("3.3").unwrap().eval_float_with_context_mut(&mut stack, &mut context), Ok(3.3));
	assert_eq!(
		build_flat_node("asd").unwrap().eval_float_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("asd".to_owned()))
	);

	assert_eq!(build_flat_node::<DefaultNumericTypes>("3").unwrap().eval_float(), Ok(3.0));
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("true").unwrap().eval_float(),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("abc").unwrap().eval_float(),
		Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned()))
	);
	assert_eq!(build_flat_node("3").unwrap().eval_float_with_context(&mut stack, &context), Ok(3.0));
	assert_eq!(
		build_flat_node("true").unwrap().eval_float_with_context(&mut stack, &context),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		build_flat_node("abc").unwrap().eval_float_with_context(&mut stack, &context),
		Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned()))
	);
	assert_eq!(build_flat_node("3").unwrap().eval_float_with_context_mut(&mut stack, &mut context), Ok(3.0));
	assert_eq!(
		build_flat_node("true").unwrap().eval_float_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		build_flat_node("abc").unwrap().eval_float_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("abc".to_owned()))
	);

	assert_eq!(build_flat_node::<DefaultNumericTypes>("true").unwrap().eval_boolean(), Ok(true));
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("4").unwrap().eval_boolean(),
		Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) })
	);
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("trueee").unwrap().eval_boolean(),
		Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned()))
	);
	assert_eq!(build_flat_node("true").unwrap().eval_boolean_with_context(&mut stack, &context), Ok(true));
	assert_eq!(
		build_flat_node("4").unwrap().eval_boolean_with_context(&mut stack, &context),
		Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) })
	);
	assert_eq!(
		build_flat_node("trueee").unwrap().eval_boolean_with_context(&mut stack, &context),
		Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned()))
	);
	assert_eq!(
		build_flat_node("true").unwrap().eval_boolean_with_context_mut(&mut stack, &mut context),
		Ok(true)
	);
	assert_eq!(
		build_flat_node("4").unwrap().eval_boolean_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::ExpectedBoolean { actual: Value::Float(4.0) })
	);
	assert_eq!(
		build_flat_node("trueee").unwrap().eval_boolean_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("trueee".to_owned()))
	);

	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("33").unwrap().eval_tuple(),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(33.0) })
	);
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("3a3").unwrap().eval_tuple(),
		Err(EvalexprError::VariableIdentifierNotFound("3a3".to_owned()))
	);
	assert_eq!(
		build_flat_node("33").unwrap().eval_tuple_with_context(&mut stack, &context),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(33.0) })
	);
	assert_eq!(
		build_flat_node("3a3").unwrap().eval_tuple_with_context(&mut stack, &context),
		Err(EvalexprError::VariableIdentifierNotFound("3a3".to_owned()))
	);
	assert_eq!(
		build_flat_node("33").unwrap().eval_tuple_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(33.0) })
	);
	assert_eq!(
		build_flat_node("3a3").unwrap().eval_tuple_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("3a3".to_owned()))
	);

	assert_eq!(build_flat_node::<DefaultNumericTypes>("").unwrap().eval_empty(), Ok(EMPTY_VALUE));
	assert_eq!(build_flat_node::<DefaultNumericTypes>("()").unwrap().eval_empty(), Ok(EMPTY_VALUE));
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("(,)").unwrap().eval_empty(),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(
		build_flat_node::<DefaultNumericTypes>("xaq").unwrap().eval_empty(),
		Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned()))
	);
	assert_eq!(build_flat_node("").unwrap().eval_empty_with_context(&mut stack, &context), Ok(EMPTY_VALUE));
	assert_eq!(build_flat_node("()").unwrap().eval_empty_with_context(&mut stack, &context), Ok(EMPTY_VALUE));
	assert_eq!(
		build_flat_node("(,)").unwrap().eval_empty_with_context(&mut stack, &context),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(
		build_flat_node("xaq").unwrap().eval_empty_with_context(&mut stack, &context),
		Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned()))
	);
	assert_eq!(
		build_flat_node("").unwrap().eval_empty_with_context_mut(&mut stack, &mut context),
		Ok(EMPTY_VALUE)
	);
	assert_eq!(
		build_flat_node("()").unwrap().eval_empty_with_context_mut(&mut stack, &mut context),
		Ok(EMPTY_VALUE)
	);
	assert_eq!(
		build_flat_node("(,)").unwrap().eval_empty_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(thin_vec![Value::Empty, Value::Empty]) })
	);
	assert_eq!(
		build_flat_node("xaq").unwrap().eval_empty_with_context_mut(&mut stack, &mut context),
		Err(EvalexprError::VariableIdentifierNotFound("xaq".to_owned()))
	);
}

#[test]
fn test_whitespace() {
	assert!(eval_boolean("2 < = 3").is_err());
}

// #[test]
// fn test_assignment() {
// 	let mut context = HashMapContext::<DefaultNumericTypes>::new();
// 	assert_eq!(eval_empty_with_context_mut("int = 3", &mut context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_empty_with_context_mut("float = 2.0", &mut context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_empty_with_context_mut("tuple = (1,1)", &mut context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_empty_with_context_mut("empty = ()", &mut context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_empty_with_context_mut("boolean = false", &mut context), Ok(EMPTY_VALUE));

// 	assert_eq!(eval_float_with_context("float", &context), Ok(2.0));
// 	assert_eq!(
// 		eval_tuple_with_context("tuple", &context),
// 		Ok(thin_vec![Value::from_float(1.0), Value::from_float(1.0)])
// 	);
// 	assert_eq!(eval_empty_with_context("empty", &context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_boolean_with_context("boolean", &context), Ok(false));

// 	assert_eq!(eval_empty_with_context_mut("b = a = 5", &mut context), Ok(EMPTY_VALUE));
// 	assert_eq!(eval_empty_with_context("b", &context), Ok(EMPTY_VALUE));
// }

// #[test]
// fn test_expression_chaining() {
// 	let mut context = HashMapContext::<DefaultNumericTypes>::new();
// 	assert_eq!(eval_float_with_context_mut("a = 5; a = a + 2; a", &mut context), Ok(7.0));
// }

// #[test]
// fn test_string_escaping() {
//     assert_eq!(
//         eval("\"\\\"str\\\\ing\\\"\""),
//         Ok(Value::from("\"str\\ing\""))
//     );
// }

#[test]
fn test_tuple_definitions() {
	test_eval_and_fold_to_const("()", Ok(Value::Empty));
	test_eval_and_fold_to_const("(3)", Ok(Value::Float(3.0)));
	test_eval_and_fold_to_const("(3, 4)", Ok(Value::Float2(3.0, 4.0)));
	test_eval_and_fold_to_const(
		"2, (5, 6)",
		Ok(Value::from(thin_vec![Value::from_float(2.0), Value::Float2(5.0, 6.0)])),
	);
	test_eval_and_fold_to_const("1, 2", Ok(Value::Float2(1.0, 2.0)));
	test_eval_and_fold_to_const(
		"1, 2, 3, 4",
		Ok(Value::from(thin_vec![
			Value::from_float(1.0),
			Value::from_float(2.0),
			Value::from_float(3.0),
			Value::from_float(4.0)
		])),
	);

	test_eval_and_fold_to_const(
		"(1, 2, 3), 5, 6, (true, false, 0)",
		Ok(Value::from(thin_vec![
			Value::from(thin_vec![Value::from_float(1.0), Value::from_float(2.0), Value::from_float(3.0)]),
			Value::from_float(5.0),
			Value::from_float(6.0),
			Value::from(thin_vec![Value::from(true), Value::from(false), Value::from_float(0.0)])
		])),
	);
	test_eval_and_fold_to_const("1, (2)", Ok(Value::Float2(1.0, 2.0)));
	test_eval_and_fold_to_const("1, ()", Ok(Value::from(thin_vec![Value::from_float(1.0), Value::from(())])));
	test_eval_and_fold_to_const("1, ((2))", Ok(Value::Float2(1.0, 2.0)));
}

// #[test]
// fn test_implicit_context() {
// 	assert_eq!(eval("a = 2 + 4 * 2; b = -5 + 3 * 5; a == b"), Ok(Value::from(true)));
// 	assert_eq!(eval_boolean("a = 2 + 4 * 2; b = -5 + 3 * 5; a == b"), Ok(true));
// 	assert_eq!(eval_float("a = 2 + 4 * 2; b = -5 + 3 * 5; a - b"), Ok(0.0));
// 	assert_eq!(eval_float("a = 2 + 4 * 2; b = -5 + 3 * 5; a - b + 0.5"), Ok(0.5));
// 	assert_eq!(eval_float("a = 2 + 4 * 2; b = -5 + 3 * 5; a - b"), Ok(0.0));
// 	assert_eq!(eval_empty("a = 2 + 4 * 2; b = -5 + 3 * 5;"), Ok(()));
// 	assert_eq!(
// 		eval_tuple("a = 2 + 4 * 2; b = -5 + 3 * 5; a, b + 0.5"),
// 		Ok(thin_vec![Value::from_float(10.0), Value::from_float(10.5)])
// 	);
// 	// assert_eq!(
// 	//     eval_string("a = \"xyz\"; b = \"abc\"; c = a + b; c"),
// 	//     Ok("xyzabc".to_string())
// 	// );
// }

// #[test]
// fn test_operator_assignments() {
// 	let mut context = HashMapContext::<DefaultNumericTypes>::new();
// 	assert_eq!(eval_empty_with_context_mut("a = 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("a += 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("a -= 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("a *= 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("b = 5.0", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("b /= 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("b %= 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("b ^= 5", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("c = true", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("c &&= false", &mut context), Ok(()));
// 	assert_eq!(eval_empty_with_context_mut("c ||= true", &mut context), Ok(()));

// 	let mut context = HashMapContext::<DefaultNumericTypes>::new();
// 	assert_eq!(eval_float_with_context_mut("a = 5; a", &mut context), Ok(5.0));
// 	assert_eq!(eval_float_with_context_mut("a += 3; a", &mut context), Ok(8.0));
// 	assert_eq!(eval_float_with_context_mut("a -= 5; a", &mut context), Ok(3.0));
// 	assert_eq!(eval_float_with_context_mut("a *= 5; a", &mut context), Ok(15.0));
// 	assert_eq!(eval_float_with_context_mut("b = 5.0; b", &mut context), Ok(5.0));
// 	assert_eq!(eval_float_with_context_mut("b /= 2; b", &mut context), Ok(2.5));
// 	assert_eq!(eval_float_with_context_mut("b %= 2; b", &mut context), Ok(0.5));
// 	assert_eq!(eval_float_with_context_mut("b ^= 2; b", &mut context), Ok(0.25));
// 	assert_eq!(eval_boolean_with_context_mut("c = true; c", &mut context), Ok(true));
// 	assert_eq!(eval_boolean_with_context_mut("c &&= false; c", &mut context), Ok(false));
// 	assert_eq!(eval_boolean_with_context_mut("c ||= true; c", &mut context), Ok(true));
// }

#[test]
fn test_type_errors_in_binary_operators() {
	// Only addition supports incompatible types, all others work only on numbers or only on booleans.
	// So only addition requires the more fancy error message.
	// assert_eq!(
	//     eval("4 + \"abc\""),
	//     Err(EvalexprError::wrong_type_combination(
	//         Operator::Add,
	//         vec![ValueType::Float, ValueType::String]
	//     ))
	// );
	// assert_eq!(
	//     eval("\"abc\" + 4"),
	//     Err(EvalexprError::wrong_type_combination(
	//         Operator::Add,
	//         vec![ValueType::String, ValueType::Float]
	//     ))
	// );
}

// #[test]
// fn test_hashmap_context_type_safety() {
// 	let mut context: HashMapContext<DefaultNumericTypes> =
// 		context_map! {"a" => float 5, "b" => float 5.0}.unwrap();
// 	assert_eq!(eval_with_context_mut("a = 4", &mut context), Ok(Value::Empty));

// 	assert_eq!(eval_with_context_mut("b = 4.0", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b += 4", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b -= 4", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b *= 4", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b /= 4", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b %= 4", &mut context), Ok(Value::Empty));
// 	assert_eq!(eval_with_context_mut("b ^= 4", &mut context), Ok(Value::Empty));
// }

#[test]
fn test_hashmap_context_clone_debug() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	// this variable is captured by the function
	let three = 3;
	context.set_function(
		istr("mult_3"),
		RustFunction::new(move |s, _| {
			if let Value::Float(float) = s.get_arg(0).unwrap() {
				Ok(Value::Float(float * three as DefaultNumericTypes))
			} else {
				Err(EvalexprError::expected_float(s.get_arg(0).unwrap().clone()))
			}
		}),
	);

	let four = 4.0;
	context.set_function(istr("function_four"), RustFunction::new(move |_, _| Ok(Value::Float(four))));
	context.set_value(istr("variable_five"), Value::from_float(5.0)).unwrap();
	let context = context;
	#[allow(clippy::redundant_clone)]
	let cloned_context = context.clone();

	assert_eq!(format!("{:?}", &context), format!("{:?}", &cloned_context));
	assert_eq!(cloned_context.get_value(istr("variable_five")), Some(&Value::from_float(5.0)));
	assert_eq!(eval_with_context("mult_3 2", &cloned_context), Ok(Value::Float(6.0)));
	assert_eq!(eval_with_context("mult_3(3)", &cloned_context), Ok(Value::Float(9.0)));
	assert_eq!(eval_with_context("mult_3(function_four())", &cloned_context), Ok(Value::Float(12.0)));
}

#[test]
fn test_error_constructors() {
	// assert_eq!(
	//     eval("a = true && \"4\""),
	//     Err(EvalexprError::ExpectedBoolean {
	//         actual: Value::from("4")
	//     })
	// );
	assert_eq!(eval_tuple("4"), Err(EvalexprError::ExpectedTuple { actual: Value::Float(4.0) }));
	assert_eq!(
		Value::Tuple(thin_vec![Value::<DefaultNumericTypes>::Float(4.0), Value::Float(5.0)])
			.as_fixed_len_tuple(3),
		Err(EvalexprError::ExpectedFixedLengthTuple {
			expected_length: 3,
			actual:          Value::Tuple(thin_vec![Value::Float(4.0), Value::Float(5.0)]),
		})
	);
	assert_eq!(eval_empty("4"), Err(EvalexprError::ExpectedEmpty { actual: Value::Float(4.0) }));
	assert_eq!(
		eval("&"),
		Err(EvalexprError::UnmatchedPartialToken { first: PartialToken::Ampersand, second: None })
	);

	assert_eq!(expect_function_argument_amount::<DefaultNumericTypes>(2, 2), Ok(()));
	assert_eq!(
		expect_function_argument_amount::<DefaultNumericTypes>(2, 3),
		Err(EvalexprError::WrongFunctionArgumentAmount { expected: 3..=3, actual: 2 })
	);
}

// #[test]
// fn test_iterators() {
// 	let tree = build_flat_node::<DefaultNumericTypes>("writevar = 5 + 3 + fun(4) + var").unwrap();
// 	let mut iter = tree.iter_identifiers();
// 	assert_eq!(iter.next(), Some("writevar"));
// 	assert_eq!(iter.next(), Some("fun"));
// 	assert_eq!(iter.next(), Some("var"));
// 	assert_eq!(iter.next(), None);

// 	let mut iter = tree.iter_variable_identifiers();
// 	assert_eq!(iter.next(), Some("writevar"));
// 	assert_eq!(iter.next(), Some("var"));
// 	assert_eq!(iter.next(), None);

// 	// let mut iter = tree.iter_read_variable_identifiers();
// 	// assert_eq!(iter.next(), Some("var"));
// 	// assert_eq!(iter.next(), None);

// 	// let mut iter = tree.iter_write_variable_identifiers();
// 	// assert_eq!(iter.next(), Some("writevar"));
// 	// assert_eq!(iter.next(), None);

// 	let mut iter = tree.iter_function_identifiers();
// 	assert_eq!(iter.next(), Some("fun"));
// 	assert_eq!(iter.next(), None);
// }

#[test]
fn test_same_operator_chains() {
	#![allow(clippy::eq_op)]
	assert_eq!(eval("3.0 / 3.0 / 3.0 / 3.0"), Ok(Value::from_float(3.0 / 3.0 / 3.0 / 3.0)));
	assert_eq!(eval("3.0 - 3.0 - 3.0 - 3.0"), Ok(Value::from_float(3.0 - 3.0 - 3.0 - 3.0)));
}

#[test]
fn test_long_expression_i89() {
	let tree = build_flat_node::<DefaultNumericTypes>(
		"x*0.2*5/4+x*2*4*1*1*1*1*1*1*1+7*sin(y)-z/sin(3.0/2.0/(1-x*4*1*1*1*1))",
	)
	.unwrap();
	let x = 0.0;
	let y: DefaultNumericTypes = 3.0;
	let z = 4.0;
	let context = context_map! {
		"x" => float 0.0,
		"y" => float 3.0,
		"z" => float 4.0
	}
	.unwrap();
	let expected =
		x * 0.2 * 5.0 / 4.0 + x * 2.0 * 4.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 * 1.0 + 7.0 * y.sin()
			- z / (3.0 / 2.0 / (1.0 - x * 4.0 * 1.0 * 1.0 * 1.0 * 1.0)).sin();
	let mut stack = Stack::new();
	let actual: DefaultNumericTypes = tree.eval_float_with_context(&mut stack, &context).unwrap();
	assert!(
		(expected - actual).abs() < expected.abs().min(actual.abs()) * 1e-12,
		"expected: {}, actual: {}",
		expected,
		actual
	);
}

#[test]
fn test_value_type() {
	// assert_eq!(
	//     ValueType::from(&Value::<DefaultNumericTypes>::String(String::new())),
	//     ValueType::String
	// );
	assert_eq!(ValueType::from(&Value::<DefaultNumericTypes>::Float(0.0)), ValueType::Float);
	assert_eq!(ValueType::from(&Value::<DefaultNumericTypes>::Float(0.0)), ValueType::Float);
	assert_eq!(ValueType::from(&Value::<DefaultNumericTypes>::Boolean(true)), ValueType::Boolean);
	assert_eq!(ValueType::from(&Value::<DefaultNumericTypes>::Tuple(Default::default())), ValueType::Tuple);
	assert_eq!(ValueType::from(&Value::<DefaultNumericTypes>::Empty), ValueType::Empty);

	// assert_eq!(
	//     ValueType::from(&mut Value::<DefaultNumericTypes>::String(String::new())),
	//     ValueType::String
	// );
	assert_eq!(ValueType::from(&mut Value::<DefaultNumericTypes>::Float(0.0)), ValueType::Float);
	assert_eq!(ValueType::from(&mut Value::<DefaultNumericTypes>::Float(0.0)), ValueType::Float);
	assert_eq!(ValueType::from(&mut Value::<DefaultNumericTypes>::Boolean(true)), ValueType::Boolean);
	assert_eq!(
		ValueType::from(&mut Value::<DefaultNumericTypes>::Tuple(Default::default())),
		ValueType::Tuple
	);
	assert_eq!(ValueType::from(&mut Value::<DefaultNumericTypes>::Empty), ValueType::Empty);

	// assert!(!Value::<DefaultNumericTypes>::String(String::new()).is_float());
	assert!(!Value::<DefaultNumericTypes>::Boolean(true).is_float());
	assert!(!Value::<DefaultNumericTypes>::Tuple(Default::default()).is_float());
	assert!(!Value::<DefaultNumericTypes>::Empty.is_float());

	// assert!(!Value::<DefaultNumericTypes>::String(String::new()).is_empty());
	assert!(!Value::<DefaultNumericTypes>::Float(0.0).is_empty());
	assert!(!Value::<DefaultNumericTypes>::Boolean(true).is_empty());
	assert!(!Value::<DefaultNumericTypes>::Tuple(Default::default()).is_empty());
	assert!(Value::<DefaultNumericTypes>::Empty.is_empty());

	// assert_eq!(
	//     Value::<DefaultNumericTypes>::String(String::new()).as_float(),
	//     Err(EvalexprError::ExpectedFloat {
	//         actual: Value::String(String::new())
	//     })
	// );
	assert_eq!(Value::<DefaultNumericTypes>::Float(0.0).as_float(), Ok(0.0));
	assert_eq!(
		Value::<DefaultNumericTypes>::Boolean(true).as_float(),
		Err(EvalexprError::ExpectedFloat { actual: Value::Boolean(true) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Tuple(Default::default()).as_float(),
		Err(EvalexprError::ExpectedFloat { actual: Value::Tuple(Default::default()) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Empty.as_float(),
		Err(EvalexprError::ExpectedFloat { actual: Value::Empty })
	);

	// assert_eq!(
	//     Value::<DefaultNumericTypes>::String(String::new()).as_tuple(),
	//     Err(EvalexprError::ExpectedTuple {
	//         actual: Value::String(String::new())
	//     })
	// );
	assert_eq!(
		Value::<DefaultNumericTypes>::Float(0.0).as_tuple(),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(0.0) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Boolean(true).as_tuple(),
		Err(EvalexprError::ExpectedTuple { actual: Value::Boolean(true) })
	);
	assert_eq!(Value::<DefaultNumericTypes>::Tuple(Default::default()).as_tuple(), Ok(Default::default()));
	assert_eq!(
		Value::<DefaultNumericTypes>::Empty.as_tuple(),
		Err(EvalexprError::ExpectedTuple { actual: Value::Empty })
	);

	// assert_eq!(
	//     Value::<DefaultNumericTypes>::String(String::new()).as_fixed_len_tuple(0),
	//     Err(EvalexprError::ExpectedTuple {
	//         actual: Value::String(String::new())
	//     })
	// );
	assert_eq!(
		Value::<DefaultNumericTypes>::Float(0.0).as_fixed_len_tuple(0),
		Err(EvalexprError::ExpectedTuple { actual: Value::Float(0.0) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Boolean(true).as_fixed_len_tuple(0),
		Err(EvalexprError::ExpectedTuple { actual: Value::Boolean(true) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Tuple(Default::default()).as_fixed_len_tuple(0),
		Ok(Default::default())
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Empty.as_fixed_len_tuple(0),
		Err(EvalexprError::ExpectedTuple { actual: Value::Empty })
	);

	// assert_eq!(
	//     Value::<DefaultNumericTypes>::String(String::new()).as_empty(),
	//     Err(EvalexprError::ExpectedEmpty {
	//         actual: Value::String(String::new())
	//     })
	// );
	assert_eq!(
		Value::<DefaultNumericTypes>::Float(0.0).as_empty(),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Float(0.0) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Boolean(true).as_empty(),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Boolean(true) })
	);
	assert_eq!(
		Value::<DefaultNumericTypes>::Tuple(Default::default()).as_empty(),
		Err(EvalexprError::ExpectedEmpty { actual: Value::Tuple(Default::default()) })
	);
	assert_eq!(Value::<DefaultNumericTypes>::Empty.as_empty(), Ok(()));

	// assert_eq!(
	//     Result::from(Value::<DefaultNumericTypes>::String(String::new())),
	//     Ok(Value::String(String::new()))
	// );
}

#[test]
fn test_parenthese_combinations() {
	// These are from issue #94
	assert_eq!(eval("123(1*2)"), Err(EvalexprError::MissingOperatorOutsideOfBrace));
	assert_eq!(eval("1()"), Err(EvalexprError::MissingOperatorOutsideOfBrace));
	assert_eq!(eval("1()()()()"), Err(EvalexprError::MissingOperatorOutsideOfBrace));
	assert_eq!(eval("1()()()(9)()()"), Err(EvalexprError::MissingOperatorOutsideOfBrace));
	assert_eq!(
		eval_with_context("a+100(a*2)", &context_map! {"a" => float 4}.unwrap()),
		Err(EvalexprError::<DefaultNumericTypes>::MissingOperatorOutsideOfBrace)
	);
	assert_eq!(eval_float("(((1+2)*(3+4)+(5-(6)))/((7-8)))"), Ok(-20.0));
	assert_eq!(eval_float("(((((5)))))"), Ok(5.0));
}

#[test]
fn test_try_from() {
	#![allow(clippy::redundant_clone)]

	// let value = Value::<DefaultNumericTypes>::String("abc".to_string());
	// assert_eq!(String::try_from(value.clone()), Ok("abc".to_string()));
	// assert_eq!(
	//     bool::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedBoolean {
	//         actual: value.clone()
	//     })
	// );
	// assert_eq!(
	//     TupleType::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedTuple {
	//         actual: value.clone()
	//     })
	// );
	// assert_eq!(
	//     EmptyType::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedEmpty {
	//         actual: value.clone()
	//     })
	// );

	let value = Value::<DefaultNumericTypes>::Float(1.3);
	// assert_eq!(
	//     String::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedString {
	//         actual: value.clone()
	//     })
	// );
	assert_eq!(bool::try_from(value.clone()), Err(EvalexprError::ExpectedBoolean { actual: value.clone() }));
	assert_eq!(
		TupleType::try_from(value.clone()),
		Err(EvalexprError::ExpectedTuple { actual: value.clone() })
	);
	assert_eq!(
		EmptyType::try_from(value.clone()),
		Err(EvalexprError::ExpectedEmpty { actual: value.clone() })
	);

	let value = Value::<DefaultNumericTypes>::Float(13.0);
	// assert_eq!(
	//     String::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedString {
	//         actual: value.clone()
	//     })
	// );
	assert_eq!(bool::try_from(value.clone()), Err(EvalexprError::ExpectedBoolean { actual: value.clone() }));
	assert_eq!(
		TupleType::try_from(value.clone()),
		Err(EvalexprError::ExpectedTuple { actual: value.clone() })
	);
	assert_eq!(
		EmptyType::try_from(value.clone()),
		Err(EvalexprError::ExpectedEmpty { actual: value.clone() })
	);

	let value = Value::<DefaultNumericTypes>::Boolean(true);
	// assert_eq!(
	//     String::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedString {
	//         actual: value.clone()
	//     })
	// );
	assert_eq!(bool::try_from(value.clone()), Ok(true));
	assert_eq!(
		TupleType::try_from(value.clone()),
		Err(EvalexprError::ExpectedTuple { actual: value.clone() })
	);
	assert_eq!(
		EmptyType::try_from(value.clone()),
		Err(EvalexprError::ExpectedEmpty { actual: value.clone() })
	);

	// let value =
	//     Value::<DefaultNumericTypes>::Tuple(vec![Value::Float(1.0), Value::String("abc".to_string())]);
	// assert_eq!(
	//     String::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedString {
	//         actual: value.clone()
	//     })
	// );
	// assert_eq!(
	//     bool::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedBoolean {
	//         actual: value.clone()
	//     })
	// );
	// assert_eq!(
	//     TupleType::try_from(value.clone()),
	//     Ok(vec![Value::Float(1.0), Value::String("abc".to_string())])
	// );
	// assert_eq!(
	//     EmptyType::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedEmpty {
	//         actual: value.clone()
	//     })
	// );

	let value = Value::<DefaultNumericTypes>::Empty;
	// assert_eq!(
	//     String::try_from(value.clone()),
	//     Err(EvalexprError::ExpectedString {
	//         actual: value.clone()
	//     })
	// );
	assert_eq!(bool::try_from(value.clone()), Err(EvalexprError::ExpectedBoolean { actual: value.clone() }));
	assert_eq!(
		TupleType::try_from(value.clone()),
		Err(EvalexprError::ExpectedTuple { actual: value.clone() })
	);
	assert_eq!(EmptyType::try_from(value.clone()), Ok(()));
}

// #[test]
// fn assignment_lhs_is_identifier() {
//     let tree = build_flat_node("a = 1").unwrap();
//     let operators: Vec<_> = tree.iter().map(|node| node.clone()).collect();

//     let mut context = HashMapContext::<DefaultNumericTypes>::new();
//     tree.eval_with_context_mut(&mut context).unwrap();
//     assert_eq!(context.get_value("a"), Some(&Value::Float(1.0)));

//     assert!(
//         matches!(
//             operators.as_slice(),
//             [
//                 Operator::Assign,
//                 Operator::VariableIdentifierWrite { identifier: value },
//                 Operator::Const {
//                     value: Value::Float(1.0)
//                 }
//             ] if value == "a"
//         ),
//         "actual: {:#?}",
//         operators
//     );
// }

// #[test]
// fn test_variable_assignment_and_iteration() {
// 	let mut context = HashMapContext::<DefaultNumericTypes>::new();
// 	eval_with_context_mut("a = 5; b = 5.0", &mut context).unwrap();

// 	let mut variables: Vec<_> = context.iter_variables().collect();
// 	variables.sort_unstable_by(|(name_a, _), (name_b, _)| name_a.cmp(name_b));
// 	assert_eq!(variables, vec![(&istr("a"), &Value::from_float(5.0)), (&istr("b"),
// &Value::from_float(5.0))],);

// 	let mut variables: Vec<_> = context.iter_variable_names().collect();
// 	variables.sort_unstable();
// 	assert_eq!(variables, vec![istr("a"), istr("b")],);
// }

#[test]
fn test_negative_power() {
	assert_eq!(eval("3^-2"), Ok(Value::Float(1.0 / 9.0)));
	assert_eq!(eval("3^(-2)"), Ok(Value::Float(1.0 / 9.0)));
	assert_eq!(eval("-3^2"), Ok(Value::Float(-9.0)));
	assert_eq!(eval("-(3)^2"), Ok(Value::Float(-9.0)));
	assert_eq!(eval("(-3)^-2"), Ok(Value::Float(1.0 / 9.0)));
	assert_eq!(eval("-(3^-2)"), Ok(Value::Float(-1.0 / 9.0)));
}

#[test]
fn test_builtin_functions_context() {
	let context = HashMapContext::<DefaultNumericTypes>::new();
	assert_eq!(eval_with_context("max(1,3)", &context), Ok(Value::from_float(3.0)));
}

#[test]
fn test_hex() {
	assert_eq!(eval("0x3"), Ok(Value::Float(3.0)));
	assert_eq!(eval("0xFF"), Ok(Value::Float(255.0)));
	assert_eq!(eval("-0xFF"), Ok(Value::Float(-255.0)));
	assert_eq!(
		eval("0x"),
		// The "VariableIdentifierNotFound" error is what evalexpr currently returns,
		// but ideally it would return more specific errors for "illegal" literals.
		Err(EvalexprError::VariableIdentifierNotFound("0x".into()))
	);
}

#[test]
fn test_broken_string() {
	assert_eq!(eval(r#""abc" == "broken string"#), Err(EvalexprError::UnmatchedDoubleQuote));
}

#[test]
fn test_comments() {
	assert_eq!(
		eval(
			"
            // input
            1;  // 1
            // output
            1 + 2  // add"
		),
		Ok(Value::Float(3.0))
	);

	assert_eq!(eval("0 /*"), Err(EvalexprError::CustomMessage("unmatched inline comment".into())));

	assert_eq!(eval("1 % 4 + /*inline comment*/ 6 /*END*/"), Ok(Value::Float(7.0)));

	assert_eq!(eval("/* begin */ 10 /* middle */ + 5 /* end */ + 6 // DONE"), Ok(Value::Float(21.0)));
}

#[test]
fn test_clear() {
	let mut context = HashMapContext::<DefaultNumericTypes>::new();
	// context.set_value("abc".into(), "def".into()).unwrap();
	// assert_eq!(context.get_value("abc"), Some(&("def".into())));
	// context.clear_functions();
	// assert_eq!(context.get_value("abc"), Some(&("def".into())));
	context.clear_variables();
	assert_eq!(context.get_value(istr("abc")), None);

	// context
	//     .set_function(
	//         "abc".into(),
	//         Function::new(|_, input| Ok(Value::String(format!("{}", input[0])))),
	//     )
	//     .unwrap();
	// assert_eq!(
	//     eval_with_context("abc(5)", &context).unwrap(),
	//     Value::String("5".into())
	// );
	// context.clear_variables();
	// assert_eq!(
	//     eval_with_context("abc(5)", &context).unwrap(),
	//     Value::String("5".into())
	// );
	context.clear_rust_functions();
	assert!(eval_with_context("abc(5)", &context).is_err());

	context.set_value(istr("five"), Value::from_float(5.0)).unwrap();
	// context
	//     .set_function(
	//         "abc".into(),
	//         Function::new(|_, input| Ok(Value::String(format!("{}", input[0])))),
	//     )
	//     .unwrap();
	// assert_eq!(
	//     eval_with_context("abc(five)", &context).unwrap(),
	//     Value::String("5".into())
	// );
	context.clear();
	assert!(context.get_value(istr("five")).is_none());
	assert!(eval_with_context("abc(5)", &context).is_err());
}

#[test]
fn test_compare_different_numeric_types() {
	assert_eq!(eval("1 < 2.0"), Ok(true.into()));
	assert_eq!(eval("1 >= 2"), Ok(false.into()));
	assert_eq!(eval("1 >= 2.0"), Ok(false.into()));
}

#[test]
fn test_escape_sequences() {
	assert_eq!(eval("\"\\x\""), Err(EvalexprError::IllegalEscapeSequence("\\x".to_string())));
	assert_eq!(eval("\"\\"), Err(EvalexprError::IllegalEscapeSequence("\\".to_string())));
}

#[test]
fn test_unmatched_partial_tokens() {
	assert_eq!(
		eval("|"),
		Err(EvalexprError::UnmatchedPartialToken { first: PartialToken::VerticalBar, second: None })
	);
}

// #[test]
// fn test_node_mutable_access() {
//     let mut node = build_flat_node::<DefaultNumericTypes>("5").unwrap();
//     assert_eq!(node.children_mut().len(), 1);
//     assert_eq!(*node.operator_mut(), Operator::RootNode);
// }
