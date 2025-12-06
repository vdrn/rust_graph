use core::num::NonZeroU32;

use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::error::EvalexprResultValue;
use crate::flat_node::eval::{self, access_index, eval_range, eval_range_with_step};
use crate::flat_node::subexpression_elemination::{get_n_previous_exprs, range_as_const};
use crate::flat_node::{ClosureNode, FlatOperator, MapOp};
use crate::{EvalexprError, EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, IStr, Value};

/// Inlines variables and folds the expression tree
pub fn inline_variables_and_fold<F: EvalexprFloat>(
	node: &FlatNode<F>, context: &mut HashMapContext<F>, skip_vars: &[IStr],
) -> EvalexprResult<FlatNode<F>, F> {
	// return Ok(node.clone());
	let mut new_ops = Vec::with_capacity(node.ops.len());
	new_ops.extend(node.ops.iter().take(node.num_local_var_ops as usize).cloned());

	let mut rerun_inlning = false;
	let mut popping_ops = SmallVec::<[SkippingOps; 4]>::new();
	for source_op in node.ops.iter().skip(node.num_local_var_ops as usize) {
		if let Some(popping_op) = popping_ops.last_mut() {
			// NOTE: check how If functions are compiled in compile.rs to understand this code
			match popping_op {
				SkippingOps::SkippingIfsTrueExpr(state) => {
					match state {
						SkippingIfsTrueExprState::P1PoppingEverythingUntilLabel { label, last_jump_label } => {
							// skip everything until we find label (while keeping track of last Jumps label)
							match source_op {
								FlatOperator::Label { id } => {
									if *id == *label {
										let last_jump_label = last_jump_label.unwrap();
										*state = SkippingIfsTrueExprState::P2PoppingLabel {
											label: last_jump_label,
										};
									}
								},
								FlatOperator::Jump { id: jumps_label, .. } => {
									*last_jump_label = Some(*jumps_label);
								},
								_ => {},
							}
							continue;
						},
						SkippingIfsTrueExprState::P2PoppingLabel { label } => {
							// find label L2 and skip it, then we're done
							if let FlatOperator::Label { id } = source_op {
								if *id == *label {
									// we're done
									popping_ops.pop().unwrap();
									continue;
								}
							}
						},
					}
				},
				SkippingOps::SkippingIfsFalseExpr(state) => {
					match state {
						SkippingIfsFalseExprState::P1FindingTargetlabel { label } => {
							// find target label, while adding eveything while we're finding it
							// once we find it, we pop the last op (which has to be Jump) and read its label
							if let FlatOperator::Label { id } = source_op {
								if *id == *label {
									let last_jump = new_ops.pop().unwrap();
									let FlatOperator::Jump { id: jumps_label, .. } = last_jump else {
										unreachable!()
									};
									*state = SkippingIfsFalseExprState::P2PopEverythingUntilLabel {
										label: jumps_label,
									};

									continue;
								}
							}
						},
						SkippingIfsFalseExprState::P2PopEverythingUntilLabel { label } => {
							// pop eveything until we find label
							if let FlatOperator::Label { id } = source_op {
								if *id == *label {
									// we're done
									popping_ops.pop().unwrap();
								}
							}
							continue;
						},
					}
				},
			}
		}

		match source_op {
			FlatOperator::ReadVar { identifier } => {
				if !skip_vars.contains(identifier) {
					if let Some(value) = context.get_value(*identifier) {
						new_ops.push(FlatOperator::PushConst { value: value.clone() });
					} else {
						new_ops.push(source_op.clone());
					}
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::Range => {
				if let Some((start, end)) = get_last_2_if_const(&new_ops) {
					new_ops.pop().unwrap();
					new_ops.pop().unwrap();
					let result = eval_range(start, end)?;

					new_ops.push(FlatOperator::PushConst { value: Value::Tuple(result) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::RangeWithStep => {
				if let Some((start, end, step)) = get_last_3_if_const(&new_ops) {
					new_ops.pop().unwrap();
					new_ops.pop().unwrap();
					new_ops.pop().unwrap();

					let result = eval_range_with_step(start, end, step)?;

					new_ops.push(FlatOperator::PushConst { value: Value::Tuple(result) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::Sum(closure) | FlatOperator::Product(closure) => {
				let is_sum = matches!(source_op, FlatOperator::Sum { .. });
				match closure.as_ref() {
					ClosureNode::Unprepared { expr, params } => {
						if let Some(tuple_param) = pop_last_if_const_tuple(&mut new_ops)? {
							if tuple_param.len() > 20 {
								new_ops.push(FlatOperator::PushConst {
									value: Value::Tuple(tuple_param.clone()),
								});
							} else {
								rerun_inlning = true;
								let param_len = tuple_param.len();

								let prev_arg_values = params
									.iter()
									.map(|v| context.get_value(*v).cloned())
									.collect::<SmallVec<[Option<Value<F>>; 4]>>();

								for value in tuple_param.into_iter().rev() {
									context.set_value(params[0], value)?;
									let mut inlined_expr =
										inline_variables_and_fold(expr, context, skip_vars)?;
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
								for (i, value) in prev_arg_values.into_iter().enumerate() {
									if let Some(value) = value {
										context.set_value(params[i], value)?;
									} else {
										context.remove_value(params[i])?;
									}
								}
								continue;
							}
						}
						let new_expr = inline_variables_and_fold(expr, context, skip_vars)?;
						if is_sum {
							new_ops.push(FlatOperator::Sum(Box::new(ClosureNode::Unprepared {
								expr:   new_expr,
								params: params.clone(),
							})));
						} else {
							new_ops.push(FlatOperator::Product(Box::new(ClosureNode::Unprepared {
								expr:   new_expr,
								params: params.clone(),
							})));
						}
					},
					ClosureNode::Prepared { .. } => {
						// might get optimized away later at function inlining
						new_ops.push(source_op.clone());
					},
				}
			},
			FlatOperator::Map(map_op) => {
				if let Some(tuple_param) = pop_last_if_const_as_tuple(&mut new_ops)? {
					let tuple_len = tuple_param.len();
					if tuple_len > 20 {
						new_ops.push(FlatOperator::PushConst { value: Value::Tuple(tuple_param) });
					} else {
						match map_op {
							MapOp::Closure(closure) => match closure.as_ref() {
								ClosureNode::Unprepared { expr, params } => {
									let prev_arg_values = params
										.iter()
										.map(|v| context.get_value(*v).cloned())
										.collect::<SmallVec<[Option<Value<F>>; 4]>>();

									for (i, value) in tuple_param.into_iter().enumerate() {
										context.set_value(params[0], value)?;
										match params.len() {
											1 => {},
											2 => {
												context
													.set_value(params[1], Value::Float(F::from_usize(i)))?;
											},
											3 => {
												context
													.set_value(params[1], Value::Float(F::from_usize(i)))?;
												context.set_value(
													params[2],
													Value::Float(F::from_usize(tuple_len)),
												)?;
											},
											_ => {
												unreachable!()
											},
										}

										let mut inlined_expr =
											inline_variables_and_fold(expr, context, skip_vars)?;
										new_ops.append(&mut inlined_expr.ops);
									}

									for (i, value) in prev_arg_values.into_iter().enumerate() {
										if let Some(value) = value {
											context.set_value(params[i], value)?;
										} else {
											context.remove_value(params[i])?;
										}
									}

									new_ops.push(FlatOperator::Tuple { len: tuple_len as u32 });
									rerun_inlning = true;
									continue;
								},
								ClosureNode::Prepared { .. } => {
									new_ops.push(FlatOperator::PushConst { value: Value::Tuple(tuple_param) });
								},
							},
							MapOp::Func { name } => {
								if let Some(expr_function) = context.expr_functions.get(name) {
									for (i, value) in tuple_param.into_iter().enumerate() {
										match expr_function.args.len() {
											1 => {
												new_ops.push(FlatOperator::PushConst { value });
											},
											2 => {
												new_ops.push(FlatOperator::PushConst { value });
												new_ops.push(FlatOperator::PushConst {
													value: Value::Float(F::from_usize(i)),
												});
											},
											3 => {
												new_ops.push(FlatOperator::PushConst { value });
												new_ops.push(FlatOperator::PushConst {
													value: Value::Float(F::from_usize(i)),
												});
												new_ops.push(FlatOperator::PushConst {
													value: Value::Float(F::from_usize(tuple_len)),
												});
											},
											l => {
												return Err(
													EvalexprError::wrong_function_argument_amount_range(
														l,
														1..=3,
													),
												);
											},
										}
										new_ops.push(FlatOperator::FunctionCall {
											identifier: *name,
											arg_num:    expr_function.args.len() as u32,
										});
									}
									new_ops.push(FlatOperator::Tuple { len: tuple_len as u32 });
									rerun_inlning = true;
									continue;
								} else {
									// fn not found, skip inlining
									new_ops.push(FlatOperator::PushConst { value: Value::Tuple(tuple_param) });
								}
							},
						}
					}
				}
				match map_op {
					MapOp::Closure(closure) => match closure.as_ref() {
						ClosureNode::Unprepared { expr, params } => {
							let mut new_skip_vars = params.clone();
							new_skip_vars.extend(skip_vars.iter().cloned());

							let new_expr = inline_variables_and_fold(expr, context, &new_skip_vars)?;
							new_ops.push(FlatOperator::Map(MapOp::Closure(Box::new(
								ClosureNode::Unprepared { params: params.clone(), expr: new_expr },
							))));
						},
						ClosureNode::Prepared { .. } => {
							new_ops.push(source_op.clone());
						},
					},
					MapOp::Func { name } => {
						new_ops.push(FlatOperator::Map(MapOp::Func { name: *name }));
					},
				}
			},
			FlatOperator::Add => {
				fold_binary_op_with_two_const_variants(
					&mut new_ops,
					source_op,
					|a, b| eval::add(a, b),
					|value| FlatOperator::AddConst { value },
					|value| FlatOperator::AddConst { value },
				)?;
			},
			FlatOperator::AddConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::add(base, value.clone()))?;
			},
			FlatOperator::Sub => {
				fold_binary_op_with_two_const_variants(
					&mut new_ops,
					source_op,
					|a, b| eval::sub(a, b),
					|value| FlatOperator::SubConst { value },
					|value| FlatOperator::ConstSub { value },
				)?;
			},
			FlatOperator::SubConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::sub(base, value.clone()))?;
			},
			FlatOperator::ConstSub { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |sub| eval::sub(value.clone(), sub))?;
			},
			FlatOperator::Mul => {
				fold_binary_op_with_two_const_variants(
					&mut new_ops,
					source_op,
					|a, b| eval::mul(a, b),
					|value| FlatOperator::MulConst { value },
					|value| FlatOperator::MulConst { value },
				)?;
			},
			FlatOperator::MulConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::mul(base, value.clone()))?;
			},
			FlatOperator::Div => {
				fold_binary_op_with_two_const_variants(
					&mut new_ops,
					source_op,
					|a, b| eval::div(a, b),
					|value| FlatOperator::DivConst { value },
					|value| FlatOperator::ConstDiv { value },
				)?;
			},
			FlatOperator::DivConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::div(base, value.clone()))?;
			},
			FlatOperator::ConstDiv { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |div| eval::div(value.clone(), div))?;
			},
			FlatOperator::Exp => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| eval::exp(a, b),
					|value| FlatOperator::ExpConst { value },
				)?;
			},
			FlatOperator::ExpConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::exp(base, value.clone()))?;
			},
			FlatOperator::Mod => {
				fold_binary_op_with_const_variant(
					&mut new_ops,
					source_op,
					|a, b| eval::mod_(a, b),
					|value| FlatOperator::ModConst { value },
				)?;
			},

			FlatOperator::ModConst { value } => {
				fold_binary_const_op(&mut new_ops, source_op, |base| eval::mod_(base, value.clone()))?;
			},
			FlatOperator::Neg => {
				fold_unary_op(&mut new_ops, source_op, |a| eval::neg(a))?;
			},
			FlatOperator::Square => {
				fold_unary_op(&mut new_ops, source_op, |a| eval::square(a))?;
			},
			FlatOperator::Cube => {
				fold_unary_op(&mut new_ops, source_op, |a| eval::cube(a))?;
			},
			FlatOperator::Sqrt => {
				fold_unary_op(&mut new_ops, source_op, eval::sqrt)?;
			},
			FlatOperator::Cbrt => {
				fold_unary_op(&mut new_ops, source_op, eval::cbrt)?;
			},
			FlatOperator::Abs => {
				fold_unary_op(&mut new_ops, source_op, eval::abs)?;
			},
			FlatOperator::Floor => {
				fold_unary_op(&mut new_ops, source_op, eval::floor)?;
			},
			FlatOperator::Round => {
				fold_unary_op(&mut new_ops, source_op, eval::round)?;
			},
			FlatOperator::Ceil => {
				fold_unary_op(&mut new_ops, source_op, eval::ceil)?;
			},
			FlatOperator::Ln => {
				fold_unary_op(&mut new_ops, source_op, eval::ln)?;
			},
			FlatOperator::Log2 => {
				fold_unary_op(&mut new_ops, source_op, eval::log2)?;
			},
			FlatOperator::Log10 => {
				fold_unary_op(&mut new_ops, source_op, eval::log10)?;
			},
			FlatOperator::ExpE => {
				fold_unary_op(&mut new_ops, source_op, eval::exp_e)?;
			},
			FlatOperator::Exp2 => {
				fold_unary_op(&mut new_ops, source_op, eval::exp_2)?;
			},
			FlatOperator::Cos => {
				fold_unary_op(&mut new_ops, source_op, eval::cos)?;
			},
			FlatOperator::Acos => {
				fold_unary_op(&mut new_ops, source_op, eval::acos)?;
			},
			FlatOperator::CosH => {
				fold_unary_op(&mut new_ops, source_op, eval::cosh)?;
			},
			FlatOperator::AcosH => {
				fold_unary_op(&mut new_ops, source_op, eval::acosh)?;
			},
			FlatOperator::Sin => {
				fold_unary_op(&mut new_ops, source_op, eval::sin)?;
			},
			FlatOperator::Asin => {
				fold_unary_op(&mut new_ops, source_op, eval::asin)?;
			},
			FlatOperator::SinH => {
				fold_unary_op(&mut new_ops, source_op, eval::sinh)?;
			},
			FlatOperator::AsinH => {
				fold_unary_op(&mut new_ops, source_op, eval::asinh)?;
			},
			FlatOperator::Tan => {
				fold_unary_op(&mut new_ops, source_op, eval::tan)?;
			},
			FlatOperator::Atan => {
				fold_unary_op(&mut new_ops, source_op, eval::atan)?;
			},
			FlatOperator::TanH => {
				fold_unary_op(&mut new_ops, source_op, eval::tanh)?;
			},
			FlatOperator::AtanH => {
				fold_unary_op(&mut new_ops, source_op, eval::atanh)?;
			},
			FlatOperator::Signum => {
				fold_unary_op(&mut new_ops, source_op, eval::signum)?;
			},
			FlatOperator::Factorial => {
				fold_unary_op(&mut new_ops, source_op, eval::factorial)?;
			},
			FlatOperator::Gcd => {
				fold_binary_op(&mut new_ops, source_op, eval::gcd)?;
			},
			FlatOperator::MulAdd => {
				// a*b + c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::mul_add, |a, b, _| {
					let mul = eval::mul(a, b)?;
					Ok(FlatOperator::AddConst { value: mul })
				})?;
			},
			FlatOperator::MulSub => {
				// a*b - c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::mul_sub, |a, b, _| {
					let mul = eval::mul(a, b)?;
					Ok(FlatOperator::ConstSub { value: mul })
				})?;
			},
			FlatOperator::DivAdd => {
				// a/b + c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::div_add, |a, b, _| {
					let div = eval::div(a, b)?;
					Ok(FlatOperator::AddConst { value: div })
				})?;
			},
			FlatOperator::DivSub => {
				// a/b - c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::div_sub, |a, b, _| {
					let div = eval::div(a, b)?;
					Ok(FlatOperator::ConstSub { value: div })
				})?;
			},
			FlatOperator::AddMul => {
				// (a+b) * c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::add_mul, |a, b, _| {
					let added = eval::add(a, b)?;
					Ok(FlatOperator::MulConst { value: added })
				})?;
			},
			FlatOperator::SubMul => {
				// (a-b) * c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::sub_mul, |a, b, _| {
					let sub = eval::sub(a, b)?;
					Ok(FlatOperator::MulConst { value: sub })
				})?;
			},
			FlatOperator::AddDiv => {
				// (a+b) / c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::add_div, |a, b, _| {
					let added = eval::add(a, b)?;
					Ok(FlatOperator::ConstDiv { value: added })
				})?;
			},
			FlatOperator::SubDiv => {
				// (a-b) / c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::sub_div, |a, b, _| {
					let sub = eval::sub(a, b)?;
					Ok(FlatOperator::ConstDiv { value: sub })
				})?;
			},
			FlatOperator::MulDiv => {
				// a*b / c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::mul_div, |a, b, _| {
					let mul = eval::mul(a, b)?;
					Ok(FlatOperator::ConstDiv { value: mul })
				})?;
			},
			FlatOperator::DivMul => {
				// a/b * c
				fold_ternary_op_with_partial(&mut new_ops, source_op, eval::div_mul, |a, b, _| {
					let div = eval::div(a, b)?;
					Ok(FlatOperator::MulConst { value: div })
				})?;
			},
			FlatOperator::Log => {
				fold_binary_op(&mut new_ops, source_op, eval::log)?;
			},
			FlatOperator::Atan2 => {
				fold_binary_op(&mut new_ops, source_op, eval::atan2)?;
			},
			FlatOperator::Hypot => {
				fold_binary_op(&mut new_ops, source_op, eval::hypot)?;
			},
			FlatOperator::Min => {
				fold_binary_op(&mut new_ops, source_op, eval::min)?;
			},
			FlatOperator::Max => {
				fold_binary_op(&mut new_ops, source_op, eval::max)?;
			},
			FlatOperator::Clamp => {
				fold_ternary_op(&mut new_ops, source_op, eval::clamp)?;
			},
			FlatOperator::AddN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, eval::add)?;
			},
			FlatOperator::SubN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, eval::sub)?;
			},

			FlatOperator::MulN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, eval::mul)?;
			},
			FlatOperator::DivN { n } => {
				fold_nary_op(&mut new_ops, *n as usize, source_op, eval::div)?;
			},

			FlatOperator::Gt => {
				fold_binary_op(&mut new_ops, source_op, eval::greater)?;
			},
			FlatOperator::Lt => {
				fold_binary_op(&mut new_ops, source_op, eval::less)?;
			},
			FlatOperator::Geq => {
				fold_binary_op(&mut new_ops, source_op, eval::greater_eq)?;
			},
			FlatOperator::Leq => {
				fold_binary_op(&mut new_ops, source_op, eval::less_eq)?;
			},
      FlatOperator::Eq => {
        fold_binary_op(&mut new_ops, source_op,|a,b| Ok(eval::eq(a,b)))?;
      },
      FlatOperator::Neq => {
        fold_binary_op(&mut new_ops, source_op,|a,b| Ok(eval::neq(a,b)))?;
      },
      FlatOperator::And => {
        fold_binary_op(&mut new_ops, source_op, eval::and)?;
      },
      FlatOperator::Or => {
        fold_binary_op(&mut new_ops, source_op, eval::or)?;
      },
      FlatOperator::Not => {
        fold_unary_op(&mut new_ops, source_op, eval::not)?;
      },

			FlatOperator::Tuple { len } => {
				if *len == 2 {
					if let Some((a, b)) = get_last_2_if_const(&new_ops) {
						new_ops.pop().unwrap();
						new_ops.pop().unwrap();
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
				ClosureNode::Unprepared { expr, params } => {
					let mut new_skip_vars = params.clone();
					new_skip_vars.extend(skip_vars.iter().cloned());

					let new_expr = inline_variables_and_fold(expr, context, &new_skip_vars)?;
					new_ops.push(FlatOperator::Integral(Box::new(ClosureNode::Unprepared {
						expr:   new_expr,
						params: params.clone(),
					})));
				},
				ClosureNode::Prepared { .. } => {
					new_ops.push(source_op.clone());
				},
			},
			FlatOperator::AccessX => {
				if let Some(last_const) = get_last_if_const_as_float2(&new_ops)? {
					new_ops.pop().unwrap();
					new_ops.push(FlatOperator::PushConst { value: Value::Float(last_const.0) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::AccessY => {
				if let Some(last_const) = get_last_if_const_as_float2(&new_ops)? {
					new_ops.pop().unwrap();
					new_ops.push(FlatOperator::PushConst { value: Value::Float(last_const.1) });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::AccessIndex { index } => {
				if let Some(last_const) = get_last_if_const(&new_ops) {
					new_ops.pop().unwrap();
					let value = access_index(last_const, *index)?;
					new_ops.push(FlatOperator::PushConst { value });
				} else {
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::Access => {
				if let Some((receiver, field)) = get_last_2_if_const(&new_ops) {
					new_ops.pop().unwrap();
					new_ops.pop().unwrap();
					new_ops.push(FlatOperator::PushConst { value: eval::access(receiver, field)? });
				} else {
					// Todo also compile to AccessIndex if just field is a const
					new_ops.push(source_op.clone());
				}
			},
			FlatOperator::JumpIfFalse { id, .. } => {
				if let Some(const_condition) = get_last_if_const(&new_ops) {
					new_ops.pop().unwrap();
					let cond = const_condition.as_boolean()?;
					if cond {
						popping_ops.push(SkippingOps::SkippingIfsFalseExpr(
							SkippingIfsFalseExprState::P1FindingTargetlabel { label: *id },
						));
					} else {
						popping_ops.push(SkippingOps::SkippingIfsTrueExpr(
							SkippingIfsTrueExprState::P1PoppingEverythingUntilLabel {
								label:           *id,
								last_jump_label: None,
							},
						));
					}
				} else {
					new_ops.push(source_op.clone());
				}
			},

			FlatOperator::Label { .. }
			| FlatOperator::Jump { .. }
			| FlatOperator::ReadLocalVar { .. }
			| FlatOperator::ReadParam { .. }
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

	let result = FlatNode {
		ops:               new_ops,
		num_local_vars:    node.num_local_vars,
		num_local_var_ops: node.num_local_var_ops,
	};
	if rerun_inlning {
		inline_variables_and_fold(&result, context, skip_vars)
	} else {
		Ok(result)
	}
}
/// TODO: we're bailing out if any of the operands is not a constant, but we could fold all
/// continous ranges of constants
fn fold_nary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, n: usize, source_op: &FlatOperator<F>,
	fold2: impl Fn(Value<F>, Value<F>) -> EvalexprResultValue<F>,
) -> EvalexprResult<(), F> {
	let mut result = None;
	for value in new_ops[new_ops.len() - n..].iter().rev() {
		if let FlatOperator::PushConst { value } = value {
			let value = value.clone();
			if let Some(r) = result.take() {
				result = Some(fold2(r, value)?);
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
		new_ops.push(FlatOperator::PushConst { value: result });
	} else {
		new_ops.push(source_op.clone());
	}
	Ok(())
}
fn fold_unary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold: impl FnOnce(Value<F>) -> EvalexprResultValue<F>,
) -> EvalexprResult<(), F> {
	if let Some(last_const) = get_last_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold(last_const)? });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold: impl FnOnce(Value<F>, Value<F>) -> EvalexprResultValue<F>,
) -> EvalexprResult<(), F> {
	if let Some((a, b)) = get_last_2_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold(a, b)? });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_op_with_const_variant<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold_2: impl FnOnce(Value<F>, Value<F>) -> EvalexprResultValue<F>,
	fold_1: impl FnOnce(Value<F>) -> FlatOperator<F>,
) -> EvalexprResult<(), F> {
	if let Some((a, b)) = get_last_2_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold_2(a, b)? });
	} else if let Some(last_const) = get_last_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.push(fold_1(last_const));
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_op_with_two_const_variants<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold_2: impl FnOnce(Value<F>, Value<F>) -> EvalexprResultValue<F>,
	fold_1_1: impl FnOnce(Value<F>) -> FlatOperator<F>, fold_1_2: impl FnOnce(Value<F>) -> FlatOperator<F>,
) -> EvalexprResult<(), F> {
	if let Some((a, b)) = get_last_2_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold_2(a, b)? });
	} else if let Some(last_const) = get_last_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.push(fold_1_1(last_const));
	} else if let Some((second_last_const, idx)) = get_second_last_if_const(new_ops) {
		// println!("State before {new_ops:?}");
		// println!("folding secondd last const {op:?} removing {idx}");
		new_ops.remove(idx);
		new_ops.push(fold_1_2(second_last_const));
		// println!("State after {new_ops:?}");
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_binary_const_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold: impl FnOnce(Value<F>) -> EvalexprResultValue<F>,
) -> EvalexprResult<(), F> {
	if let Some(last_const) = get_last_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold(last_const)? });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_ternary_op<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold_3: impl FnOnce(Value<F>, Value<F>, Value<F>) -> EvalexprResultValue<F>,
) -> EvalexprResult<(), F> {
	if let Some((a, b, c)) = get_last_3_if_const(new_ops) {
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold_3(a, b, c)? });
	} else {
		new_ops.push(op.clone());
	}
	Ok(())
}
fn fold_ternary_op_with_partial<F: EvalexprFloat>(
	new_ops: &mut Vec<FlatOperator<F>>, op: &FlatOperator<F>,
	fold_3: impl FnOnce(Value<F>, Value<F>, Value<F>) -> EvalexprResultValue<F>,
	fold_12: impl FnOnce(Value<F>, Value<F>, ()) -> EvalexprResult<FlatOperator<F>, F>,
) -> EvalexprResult<(), F> {
	let previous_three = get_n_previous_exprs(new_ops, new_ops.len() - 1, 3);
	let c = range_as_const(new_ops, previous_three[0]);
	let b = range_as_const(new_ops, previous_three[1]);
	let a = range_as_const(new_ops, previous_three[2]);
	if let (Some(a), Some(b), Some(c)) = (a.clone(), b.clone(), c) {
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.pop().unwrap();
		new_ops.push(FlatOperator::PushConst { value: fold_3(a, b, c)? });
	} else if let (Some(a), Some(b)) = (a, b) {
		new_ops.remove(previous_three[1].0);
		new_ops.remove(previous_three[2].0);
		new_ops.push(fold_12(a, b, ())?);
	} else {
		new_ops.push(op.clone());
	}

	Ok(())
}

fn get_last_if_const<F: EvalexprFloat>(ops: &[FlatOperator<F>]) -> Option<Value<F>> {
	let Some(last) = ops.last() else {
		panic!("Must have last op");
	};
	if let FlatOperator::PushConst { value } = last {
		return Some(value.clone());
	};

	None
}
fn get_last_if_const_as_float2<F: EvalexprFloat>(
	ops: &[FlatOperator<F>],
) -> EvalexprResult<Option<(F, F)>, F> {
	if let Some(FlatOperator::PushConst { value }) = ops.last() {
		return Ok(Some(value.as_float2()?));
	};

	Ok(None)
}
fn get_second_last_if_const<F: EvalexprFloat>(ops: &[FlatOperator<F>]) -> Option<(Value<F>, usize)> {
	let prev_ranges = get_n_previous_exprs(ops, ops.len() - 1, 2);
	assert!(prev_ranges.len() == 2);
	if let Some(value) = range_as_const(ops, prev_ranges[1]) {
		return Some((value, prev_ranges[1].0));
	}

	None
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
fn pop_last_if_const_as_tuple<F: EvalexprFloat>(
	ops: &mut Vec<FlatOperator<F>>,
) -> EvalexprResult<Option<ThinVec<Value<F>>>, F> {
	if let Some(FlatOperator::PushConst { .. }) = ops.last() {
		let Some(FlatOperator::PushConst { value }) = ops.pop() else {
			unreachable!()
		};
		use thin_vec::thin_vec;
		return match value {
			Value::Tuple(tuple) => {
				return Ok(Some(tuple));
			},
			Value::Float2(x, y) => {
				return Ok(Some(thin_vec![Value::Float(x), Value::Float(y)]));
			},
			Value::Float(_) => Ok(Some(thin_vec![value])),
			Value::Boolean(_) => Ok(Some(thin_vec![value])),
			Value::Empty => Ok(Some(thin_vec![])),
		};
	};

	Ok(None)
}

fn get_last_2_if_const<F: EvalexprFloat>(ops: &[FlatOperator<F>]) -> Option<(Value<F>, Value<F>)> {
	if ops.len() > 1 {
		if let (FlatOperator::PushConst { value }, FlatOperator::PushConst { value: second }) =
			(&ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Some((value.clone(), second.clone()));
		}
	}

	None
}
fn get_last_3_if_const<F: EvalexprFloat>(ops: &[FlatOperator<F>]) -> Option<(Value<F>, Value<F>, Value<F>)> {
	if ops.len() > 2 {
		if let (
			FlatOperator::PushConst { value },
			FlatOperator::PushConst { value: second },
			FlatOperator::PushConst { value: third },
		) = (&ops[ops.len() - 3], &ops[ops.len() - 2], &ops[ops.len() - 1])
		{
			return Some((value.clone(), second.clone(), third.clone()));
		}
	}

	None
}

enum SkippingIfsTrueExprState {
	P1PoppingEverythingUntilLabel { label: u64, last_jump_label: Option<u64> },
	P2PoppingLabel { label: u64 },
}
enum SkippingIfsFalseExprState {
	P1FindingTargetlabel { label: u64 },
	P2PopEverythingUntilLabel { label: u64 },
}
enum SkippingOps {
	SkippingIfsTrueExpr(SkippingIfsTrueExprState),
	SkippingIfsFalseExpr(SkippingIfsFalseExprState),
}
pub fn setup_jump_offsets<F: EvalexprFloat>(node: &mut FlatNode<F>) {
	for i in 0..node.ops.len() {
		match &node.ops[i] {
			FlatOperator::JumpIfFalse { id, .. } | FlatOperator::Jump { id, .. } => {
				let pos = node.ops[i..]
					.iter()
					.position(|op| match op {
						FlatOperator::Label { id: other_id } => id == other_id,
						_ => false,
					})
					.unwrap();
				match &mut node.ops[i] {
					FlatOperator::JumpIfFalse { offset, .. } | FlatOperator::Jump { offset, .. } => {
						*offset = Some(NonZeroU32::new(pos as u32).unwrap());
					},
					_ => unreachable!(),
				}
			},
			_ => {},
		}
	}
}
