use thin_vec::ThinVec;

use crate::flat_node::eval::{eval_range, eval_range_with_step};
use crate::flat_node::{FlatOperator, IntegralNode};
use crate::{EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, Value};

/// Inlines variables and folds the expression tree
pub fn inline_variables_and_fold<F: EvalexprFloat>(
	node: &FlatNode<F>, context: &mut HashMapContext<F>,
) -> EvalexprResult<FlatNode<F>, F> {
	// return Ok(node.clone());
	let mut new_ops = Vec::with_capacity(node.ops.len());
	new_ops.extend(node.ops.iter().take(node.num_local_var_ops as usize).cloned());

	for source_op in node.ops.iter().skip(node.num_local_var_ops as usize) {
		match source_op {
			FlatOperator::ReadVar { identifier } => {
				if let Some(value) = context.get_value(*identifier) {
					new_ops.push(FlatOperator::PushConst { value: value.clone() });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::ReadVarNeg { identifier } => {
				if let Some(value) = context.get_value(*identifier) {
					new_ops.push(FlatOperator::PushConst { value: Value::Float(-value.as_float()?) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::Range => {
				if let Some((start, end)) = get_last_2_if_const(&new_ops)? {
					new_ops.pop();
					new_ops.pop();
					let result = eval_range(start, end)?;

					new_ops.push(FlatOperator::PushConst { value: Value::Tuple(result) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::RangeWithStep => {
				if let Some((start, end, step)) = get_last_3_if_const(&new_ops)? {
					new_ops.pop();
					new_ops.pop();
					new_ops.pop();

					let result = eval_range_with_step(start, end, step)?;

					new_ops.push(FlatOperator::PushConst { value: Value::Tuple(result) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::Sum { expr, variable } | FlatOperator::Product { expr, variable } => {
				let is_sum = matches!(source_op, FlatOperator::Sum { .. });

				if let Some(tuple_param) = pop_last_if_const_tuple(&mut new_ops)? {
					if tuple_param.len() > 20 {
						new_ops.push(FlatOperator::PushConst { value: Value::Tuple(tuple_param.clone()) });
					} else {
						let param_len = tuple_param.len();

						let prev_variable_value = context.get_value(*variable).cloned();

						for value in tuple_param {
							context.set_value(*variable, value)?;
							let mut inlined_expr = inline_variables_and_fold(expr, context)?;
							new_ops.append(&mut inlined_expr.ops);
						}

						if param_len == 2 {
							if is_sum {
								new_ops.push(FlatOperator::Add);
							} else {
								new_ops.push(FlatOperator::Mul);
							}
						} else if param_len > 2 {
							if is_sum {
								new_ops.push(FlatOperator::AddN { n: param_len as u32 });
							} else {
								new_ops.push(FlatOperator::MulN { n: param_len as u32 });
							}
						}
						if let Some(prev_variable_value) = prev_variable_value {
							context.set_value(*variable, prev_variable_value)?;
						} else {
							context.remove_value(*variable)?;
						}
						continue;
					}
				}
				if is_sum {
					let new_expr = inline_variables_and_fold(expr, context)?;
					new_ops.push(FlatOperator::Sum { expr: Box::new(new_expr), variable: *variable });
				} else {
					let new_expr = inline_variables_and_fold(expr, context)?;
					new_ops.push(FlatOperator::Product { expr: Box::new(new_expr), variable: *variable });
				}
			},
			FlatOperator::Add => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a + b,
					|value| FlatOperator::AddConst { value },
				)?;
			},
			FlatOperator::AddConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base + *value)?;
			},
			FlatOperator::Sub => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a - b,
					|value| FlatOperator::SubConst { value },
				)?;
			},
			FlatOperator::SubConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base - *value)?;
			},
			FlatOperator::ConstSub { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |sub| *value - sub)?;
			},
			FlatOperator::Mul => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a * b,
					|value| FlatOperator::MulConst { value },
				)?;
			},
			FlatOperator::MulConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base * *value)?;
			},
			FlatOperator::Div => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a / b,
					|value| FlatOperator::DivConst { value },
				)?;
			},
			FlatOperator::DivConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base / *value)?;
			},
			FlatOperator::ConstDiv { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |div| *value / div)?;
			},
			FlatOperator::Exp => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a.pow(&b),
					|value| FlatOperator::ExpConst { value },
				)?;
			},
			FlatOperator::ExpConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base.pow(value))?;
			},
			FlatOperator::Mod => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| a % b,
					|value| FlatOperator::ModConst { value },
				)?;
			},
			FlatOperator::ModConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| base % *value)?;
			},
			FlatOperator::Neg => {
				fold_unary_op(&mut new_ops, source_op, |a| -a)?;
			},
			FlatOperator::Square => {
				fold_unary_op(&mut new_ops, source_op, |a| a * a)?;
			},
			FlatOperator::Cube => {
				fold_unary_op(&mut new_ops, source_op, |a| a * a * a)?;
			},
			FlatOperator::Sqrt => {
				fold_unary_op(&mut new_ops, source_op, |a| a.sqrt())?;
			},
			FlatOperator::Cbrt => {
				fold_unary_op(&mut new_ops, source_op, |a| a.cbrt())?;
			},
			FlatOperator::Abs => {
				fold_unary_op(&mut new_ops, source_op, |a| a.abs())?;
			},
			FlatOperator::Floor => {
				fold_unary_op(&mut new_ops, source_op, |a| a.floor())?;
			},
			FlatOperator::Round => {
				fold_unary_op(&mut new_ops, source_op, |a| a.round())?;
			},
			FlatOperator::Ceil => {
				fold_unary_op(&mut new_ops, source_op, |a| a.ceil())?;
			},
			FlatOperator::Ln => {
				fold_unary_op(&mut new_ops, source_op, |a| a.ln())?;
			},
			FlatOperator::Log2 => {
				fold_unary_op(&mut new_ops, source_op, |a| a.log2())?;
			},
			FlatOperator::Log10 => {
				fold_unary_op(&mut new_ops, source_op, |a| a.log10())?;
			},
			FlatOperator::ExpE => {
				fold_unary_op(&mut new_ops, source_op, |a| a.exp())?;
			},
			FlatOperator::Exp2 => {
				fold_unary_op(&mut new_ops, source_op, |a| a.exp2())?;
			},
			FlatOperator::Cos => {
				fold_unary_op(&mut new_ops, source_op, |a| a.cos())?;
			},
			FlatOperator::Acos => {
				fold_unary_op(&mut new_ops, source_op, |a| a.acos())?;
			},
			FlatOperator::CosH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.cosh())?;
			},
			FlatOperator::AcosH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.acosh())?;
			},
			FlatOperator::Sin => {
				fold_unary_op(&mut new_ops, source_op, |a| a.sin())?;
			},
			FlatOperator::Asin => {
				fold_unary_op(&mut new_ops, source_op, |a| a.asin())?;
			},
			FlatOperator::SinH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.sinh())?;
			},
			FlatOperator::AsinH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.asinh())?;
			},
			FlatOperator::Tan => {
				fold_unary_op(&mut new_ops, source_op, |a| a.tan())?;
			},
			FlatOperator::Atan => {
				fold_unary_op(&mut new_ops, source_op, |a| a.atan())?;
			},
			FlatOperator::TanH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.tanh())?;
			},
			FlatOperator::AtanH => {
				fold_unary_op(&mut new_ops, source_op, |a| a.atanh())?;
			},
			FlatOperator::Signum => {
				fold_unary_op(&mut new_ops, source_op, |a| a.signum())?;
			},
			FlatOperator::Factorial => {
				fold_unary_op(&mut new_ops, source_op, |a| a.factorial())?;
			},
			FlatOperator::Gcd => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.gcd(&b))?;
			},
			FlatOperator::MulAdd => {
				// a*b - c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a * b + c)?;
			},
			FlatOperator::MulSub => {
				// a*b - c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a * b - c)?;
			},
			FlatOperator::DivAdd => {
				// a/b + c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a / b + c)?;
			},
			FlatOperator::DivSub => {
				// a/b - c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a / b - c)?;
			},
			FlatOperator::AddMul => {
				// (a+b) * c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a + b * c)?;
			},
			FlatOperator::SubMul => {
				// (a-b) * c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a - b * c)?;
			},
			FlatOperator::AddDiv => {
				// (a+b) / c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| (a + b) / c)?;
			},
			FlatOperator::SubDiv => {
				// (a-b) / c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| (a - b) / c)?;
			},
			FlatOperator::MulDiv => {
				// a*b / c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a * b / c)?;
			},
			FlatOperator::DivMul => {
				// a/b * c
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a / b * c)?;
			},
			FlatOperator::Log => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.log(&b))?;
			},
			FlatOperator::Atan2 => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.atan2(&b))?;
			},
			FlatOperator::Hypot => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.hypot(&b))?;
			},
			FlatOperator::Min => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.min(&b))?;
			},
			FlatOperator::Max => {
				fold_binary_op(&mut new_ops, source_op, |a, b| a.max(&b))?;
			},
			FlatOperator::Clamp => {
				fold_ternary_op(&mut new_ops, source_op, |a, b, c| a.clamp(&b, &c))?;
			},
			FlatOperator::AddN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, |a, b| a + b)?;
			},
			FlatOperator::SubN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, |a, b| a - b)?;
			},

			FlatOperator::MulN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, |a, b| a * b)?;
			},
			FlatOperator::DivN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, |a, b| a / b)?;
			},
			FlatOperator::Tuple { len } => {
				if *len == 2 {
					if let Some((a, b)) = get_last_2_if_const(&new_ops)? {
						new_ops.pop();
						new_ops.pop();
						if a.is_float() && b.is_float() {
							new_ops.push(FlatOperator::PushConst {
								value: Value::Float2(a.as_float()?, b.as_float()?),
							});
						} else {
							use thin_vec::thin_vec;
							new_ops.push(FlatOperator::PushConst { value: Value::Tuple(thin_vec![a, b]) });
						}
					} else {
						new_ops.push(source_op.clone());
					}
				} else {
					let mut result = ThinVec::with_capacity(0);
					let mut valid = true;
					for value in new_ops[new_ops.len() - *len as usize..].iter() {
						if let FlatOperator::PushConst { value } = value {
							result.push(value.clone());
						} else {
							valid = false;
							break;
						}
					}
					if valid {
						new_ops.truncate(new_ops.len() - *len as usize);
						new_ops.push(FlatOperator::PushConst { value: Value::Tuple(result) });
					} else {
						new_ops.push(source_op.clone());
					}
				}
			},
			FlatOperator::Integral(int) => match int.as_ref() {
				IntegralNode::UnpreparedExpr { expr, variable } => {
					let new_expr = inline_variables_and_fold(expr, context)?;
					new_ops.push(FlatOperator::Integral(Box::new(IntegralNode::UnpreparedExpr {
						expr:     new_expr,
						variable: *variable,
					})));
				},
				IntegralNode::PreparedFunc { .. } => {
					new_ops.push(source_op.clone());
				},
			},
			FlatOperator::ReadLocalVar { .. }
			| FlatOperator::ReadParam { .. }
			| FlatOperator::ReadParamNeg { .. }
			| FlatOperator::Eq
			| FlatOperator::Neq
			| FlatOperator::Gt
			| FlatOperator::Lt
			| FlatOperator::Geq
			| FlatOperator::Leq
			| FlatOperator::And
			| FlatOperator::Or
			| FlatOperator::Not
			| FlatOperator::Assign
			| FlatOperator::AddAssign
			| FlatOperator::SubAssign
			| FlatOperator::MulAssign
			| FlatOperator::DivAssign
			| FlatOperator::ModAssign
			| FlatOperator::ExpAssign
			| FlatOperator::AndAssign
			| FlatOperator::OrAssign
			| FlatOperator::Chain { .. }
			| FlatOperator::FunctionCall { .. }
			| FlatOperator::PushConst { .. }
			| FlatOperator::WriteVar { .. } => {
				new_ops.push(source_op.clone());
			},
		}
	}

	Ok(FlatNode {
		ops:               new_ops,
		num_local_vars:    node.num_local_vars,
		num_local_var_ops: node.num_local_var_ops,
	})
}
fn fold_nary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, n: usize, source_op: &FlatOperator<F>, fold2: impl Fn(F, F) -> F,
) -> EvalexprResult<(), F> {
	let mut result = None;
	for value in new_ops[new_ops.len() - n..].iter().rev() {
		if let FlatOperator::PushConst { value } = value {
			let value = value.as_float()?;
			if let Some(result) = &mut result {
				*result = fold2(*result, value);
			} else {
				result = Some(value);
			}
		} else {
			result = None;
			break;
		}
	}
	if let Some(result) = result {
		new_ops.truncate(new_ops.len() - n);
		new_ops.push(FlatOperator::PushConst { value: Value::Float(result) });
	} else {
		new_ops.push(source_op.clone());
	}
	Ok(())
}
fn fold_unary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>, fold: impl FnOnce(F) -> F,
) -> EvalexprResult<(), F> {
	if let Some(last_const) = get_last_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.push(FlatOperator::PushConst { value: Value::Float(fold(last_const)) });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>, fold: impl FnOnce(F, F) -> F,
) -> EvalexprResult<(), F> {
	if let Some((a, b)) = get_last_2_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.pop();
		new_ops.push(FlatOperator::PushConst { value: Value::Float(fold(a, b)) });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_op_with_const_variant<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>, fold_2: impl FnOnce(F, F) -> F,
	fold_1: impl FnOnce(F) -> FlatOperator<F>,
) -> EvalexprResult<(), F> {
	if let Some((a, b)) = get_last_2_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.pop();
		new_ops.push(FlatOperator::PushConst { value: Value::Float(fold_2(a, b)) });
	} else if let Some(last_const) = get_last_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.push(fold_1(last_const));
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_const_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>, fold: impl FnOnce(F) -> F,
) -> EvalexprResult<(), F> {
	if let Some(last_const) = get_last_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.push(FlatOperator::PushConst { value: Value::Float(fold(last_const)) });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_ternary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>, fold_3: impl FnOnce(F, F, F) -> F,
) -> EvalexprResult<(), F> {
	if let Some((a, b, c)) = get_last_3_if_const_as_float(new_ops)? {
		new_ops.pop();
		new_ops.pop();
		new_ops.pop();
		new_ops.push(FlatOperator::PushConst { value: Value::Float(fold_3(a, b, c)) });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}

fn get_last_if_const_as_float<F: EvalexprFloat>(ops: &[FlatOperator<F>]) -> EvalexprResult<Option<F>, F> {
	if let Some(FlatOperator::PushConst { value }) = ops.last() {
		return Ok(Some(value.as_float()?));
	};

	Ok(None)
}
fn pop_last_if_const_tuple<F: EvalexprFloat>(
	ops: &mut Vec<FlatOperator<F>>,
) -> EvalexprResult<Option<ThinVec<Value<F>>>, F> {
	if let Some(FlatOperator::PushConst { .. }) = ops.last() {
		let Some(FlatOperator::PushConst { value }) = ops.pop() else {
			unreachable!()
		};
		return Ok(Some(value.into_tuple()?));
	};

	Ok(None)
}

fn get_last_2_if_const_as_float<F: EvalexprFloat>(
	ops: &[FlatOperator<F>],
) -> EvalexprResult<Option<(F, F)>, F> {
	if ops.len() > 1 {
		if let (FlatOperator::PushConst { value }, FlatOperator::PushConst { value: second }) =
			(&ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Ok(Some((value.as_float()?, second.as_float()?)));
		}
	}

	Ok(None)
}
fn get_last_3_if_const_as_float<F: EvalexprFloat>(
	ops: &[FlatOperator<F>],
) -> EvalexprResult<Option<(F, F, F)>, F> {
	if ops.len() > 2 {
		if let (
			FlatOperator::PushConst { value },
			FlatOperator::PushConst { value: second },
			FlatOperator::PushConst { value: third },
		) = (&ops[ops.len() - 3], &ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Ok(Some((value.as_float()?, second.as_float()?, third.as_float()?)));
		}
	}

	Ok(None)
}
#[allow(clippy::type_complexity)]
fn get_last_2_if_const<F: EvalexprFloat>(
	ops: &[FlatOperator<F>],
) -> EvalexprResult<Option<(Value<F>, Value<F>)>, F> {
	if ops.len() > 1 {
		if let (FlatOperator::PushConst { value }, FlatOperator::PushConst { value: second }) =
			(&ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Ok(Some((value.clone(), second.clone())));
		}
	}

	Ok(None)
}
#[allow(clippy::type_complexity)]
fn get_last_3_if_const<F: EvalexprFloat>(
	ops: &[FlatOperator<F>],
) -> EvalexprResult<Option<(Value<F>, Value<F>, Value<F>)>, F> {
	if ops.len() > 2 {
		if let (
			FlatOperator::PushConst { value },
			FlatOperator::PushConst { value: second },
			FlatOperator::PushConst { value: third },
		) = (&ops[ops.len() - 3], &ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Ok(Some((value.clone(), second.clone(), third.clone())));
		}
	}

	Ok(None)
}
