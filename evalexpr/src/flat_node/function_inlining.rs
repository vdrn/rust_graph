use arrayvec::ArrayVec;
use smallvec::SmallVec;

use crate::error::expect_function_argument_amount;
use crate::flat_node::subexpression_elemination::{get_arg_ranges, get_n_previous_exprs};
use crate::flat_node::{FlatOperator, IntegralNode};
use crate::math::integrate;
use crate::{EvalexprFloat, EvalexprResult, ExpressionFunction, FlatNode, HashMapContext, IStr, Stack, Value};

pub fn inline_functions<F: EvalexprFloat>(
	node: &mut FlatNode<F>, context: &mut HashMapContext<F>,
) -> EvalexprResult<usize, F> {
	let mut cur_idx = node.num_local_var_ops as usize;
	let mut inlined_functions = 0;
	while cur_idx < node.ops.len() {
		let op = &node.ops[cur_idx];
		#[allow(clippy::single_match)]
		match op {
			FlatOperator::FunctionCall { identifier, arg_num } => {
				if let Some(expr_func) = context.expr_functions.get(identifier) {
					inline_function(node, &mut cur_idx, *identifier, expr_func.clone(), *arg_num)?;
					inlined_functions += 1;
				}
			},
			FlatOperator::ReadVar { identifier } => {
				if !context.variables.contains_key(identifier) {
					if let Some(expr_func) = context.expr_functions.get(identifier) {
						inline_function(node, &mut cur_idx, *identifier, expr_func.clone(), 0)?;
						inlined_functions += 1;
					}
				}
			},
			FlatOperator::Integral(int) => match int.as_ref() {
				IntegralNode::PreparedFunc { func, additional_args, .. } => {
					if additional_args.is_empty() {
						// println!("inlining integral closure");
						let arg_ranges = get_arg_ranges(&node.ops, cur_idx);
						assert_eq!(arg_ranges.len(), 2);
						let mut bounds = ArrayVec::<_, 2>::new();
						for (start, end) in arg_ranges {
							if start == end {
								if let FlatOperator::PushConst { value } = &node.ops[start] {
									bounds.push(value.clone());
								}
							}
						}
						if bounds.len() == 2 {
							let upper = bounds[0].as_float()?;
							let lower = bounds[1].as_float()?;
							let mut stack = Stack::<F>::with_capacity(1);
							stack.num_args = 1;
							stack.push(Value::Empty);
							let result = integrate::integrate(
								lower,
								upper,
								|x| {
									*stack.last_mut().unwrap() = Value::Float(x);
									func.unchecked_call(&mut stack, context)?.as_float()
								},
								&F::INTEGRATION_PRECISION,
							)?;
							// remove the bound args
							node.ops.drain(cur_idx - 2..cur_idx);
							cur_idx -= 2;

							// replace the integral op with the result
							node.ops[cur_idx] = FlatOperator::PushConst { value: Value::Float(result.value) };
						}
					}
				},
				_ => {},
			},
			_ => {},
		}
		cur_idx += 1;
	}

	Ok(inlined_functions)
}

// fn inline_integral_closure_params<F: EvalexprFloat>(
// 	node: &mut FlatNode<F>, cur_idx: &mut usize, fn_name: IStr, int:IntegralNode <F>, arg_num: u32,
// )-> EvalexprResult<(), F> {
//   let IntegralNode::PreparedFunc { func, variable,additional_args, .. } = int else {
//     return Ok(());
//   };
//   if !additional_args.is_empty(){
//     let parent_arg_ranges = get_arg_ranges(&node.ops, cur_idx);

//   }

// }

fn inline_function<F: EvalexprFloat>(
	node: &mut FlatNode<F>, cur_idx: &mut usize, fn_name: IStr, expr_func: ExpressionFunction<F>, arg_num: u32,
) -> EvalexprResult<(), F> {
	expect_function_argument_amount(arg_num as usize, expr_func.num_args())?;

	// println!("inlining {fn_name}");
	// println!("parent cur_idx {cur_idx} ops {:?}", node.ops);
	// println!("function ops {:?}", &expr_func.expr.ops);

	let arg_ranges = get_arg_ranges(&node.ops, *cur_idx);
	let const_args = arg_ranges
		.iter()
		.map(|(start, end)| {
			if start == end {
				if let FlatOperator::PushConst { value } = &node.ops[*start] {
					return Some(((*start, *end), value.clone()));
				}
			}
			None
		})
		.collect::<SmallVec<[Option<((usize, usize), Value<F>)>; 4]>>();

	let mut func_expr = expr_func.expr.clone();

	// inline const args into function call
	// println!("before inlining args const_args {:?}", const_args);
	// println!("before inlining args func_expr {:?}", &func_expr.ops);
	for (i, arg) in const_args.iter().enumerate() {
		if let Some((_, arg)) = arg {
			let inverse_idx = (i + 1) as u32;
			func_expr.iter_mut(&mut |op| match op {
				FlatOperator::ReadParam { inverse_index } => {
					if *inverse_index == inverse_idx {
						*op = FlatOperator::PushConst { value: arg.clone() };
					} else if *inverse_index > inverse_idx
						&& const_args[(*inverse_index - 1) as usize].is_none()
					{
						*inverse_index -= 1;
					}
				},
				FlatOperator::ReadParamNeg { inverse_index } => {
					if *inverse_index == inverse_idx {
						// TODO cannot use unwrap_or here
						*op = FlatOperator::PushConst {
							value: Value::Float(-arg.as_float().unwrap_or(F::ZERO)),
						};
					} else if *inverse_index > inverse_idx
						&& const_args[(*inverse_index - 1) as usize].is_none()
					{
						*inverse_index -= 1;
					}
				},
				FlatOperator::Integral(int) => match int.as_mut() {
					IntegralNode::UnpreparedExpr { expr, variable } => expr.iter_mut(&mut |op| match op {
						FlatOperator::ReadVar { identifier } => {
							let idx = expr_func.args.len() - i - 1;
							if *identifier == expr_func.args[idx] && *identifier != *variable {
								*op = FlatOperator::PushConst { value: arg.clone() };
							}
						},
						FlatOperator::ReadVarNeg { identifier } => {
							let idx = expr_func.args.len() - i - 1;
							if *identifier == expr_func.args[idx] && *identifier != *variable {
								*op = FlatOperator::PushConst {
									// TODO cannot use unwrap_or here
									value: Value::Float(-arg.as_float().unwrap_or(F::ZERO)),
								};
							}
						},
						_ => {},
					}),
					IntegralNode::PreparedFunc { func, additional_args, .. } => {
						additional_args.retain(|&idx| idx != inverse_idx);
						func.args.retain(|&arg| arg != expr_func.args[expr_func.args.len() - i - 1]);

						func.expr.iter_mut(&mut |op| match op {
							FlatOperator::ReadParam { inverse_index } => {
								if *inverse_index - 1 == inverse_idx {
									// TODO cannot use unwrap_or here
									*op = FlatOperator::PushConst { value: arg.clone() };
								} else if *inverse_index - 1 > inverse_idx
									&& const_args[(*inverse_index - 1) as usize].is_none()
								{
									*inverse_index -= 1;
								}
							},
							FlatOperator::ReadParamNeg { inverse_index } => {
								if *inverse_index - 1 == inverse_idx {
									// TODO cannot use unwrap_or here
									*op = FlatOperator::PushConst {
										value: Value::Float(-arg.as_float().unwrap_or(F::ZERO)),
									};
								} else if *inverse_index - 1 > inverse_idx
									&& const_args[(*inverse_index - 1) as usize].is_none()
								{
									*inverse_index -= 1;
								}
							},
							_ => {},
						});
					},
				},
				_ => {},
			});
		}
	}
	// now remove pushing of const args
	let mut num_inlined_args = 0;
	let mut num_removed_ops = 0;
	for ((start, end), _) in const_args.iter().flatten() {
		node.ops.drain(*start..=*end);
		num_removed_ops += (*start..=*end).count();
		num_inlined_args += 1;
	}
	*cur_idx -= num_removed_ops;
	let new_num_args = expr_func.num_args() - num_inlined_args;

	// run inlining again
	// if num_inlined_args > 0 {
	// 	func_expr = inline_variables_and_fold(&func_expr, context)?;
	// }
	// println!("inlined args {num_inlined_args} new num_args = {new_num_args}");
	// println!("parent after inlining args {:?}", &node.ops);
	// println!("inlining funciton after inilning args :{:?}", &func_expr.ops);
	//

	// find local vars
	let local_param_ranges = get_n_previous_exprs(
		&func_expr.ops,
		func_expr.num_local_var_ops.saturating_sub(1) as usize,
		func_expr.num_local_vars as usize,
	);
	let mut const_local_vars = local_param_ranges
		.iter()
		.map(|(start, end)| {
			if start == end {
				if let FlatOperator::PushConst { value } = &func_expr.ops[*start] {
					return Some(((*start, *end), value.clone()));
				}
			}
			None
		})
		.collect::<SmallVec<[Option<((usize, usize), Value<F>)>; 4]>>();
	// params are collected in reversse order
	const_local_vars.reverse();
	// inline const local vars
	for (i, param) in const_local_vars.iter().enumerate() {
		if let Some((_, param)) = param {
			func_expr.iter_mut(&mut |op| {
				if let FlatOperator::ReadLocalVar { idx } = op {
					if *idx == i as u32 {
						*op = FlatOperator::PushConst { value: param.clone() };
					} else if *idx > i as u32 {
						*idx -= 1;
					}
				}
			});
		}
	}
	// now remove inlined local vars
	let mut num_inlined_vars = 0;
	let mut num_removed_var_ops = 0;
	for ((start, end), _) in const_local_vars.iter().flatten() {
		func_expr.ops.drain(*start..=*end);
		num_removed_var_ops += end - start + 1;
		num_inlined_vars += 1;
	}
	func_expr.num_local_var_ops -= num_removed_var_ops as u32;
	func_expr.num_local_vars -= num_inlined_vars;
	// println!("Const inlined num vars {num_inlined_vars} ops: {num_removed_var_ops}");
	// println!("remaining ")
	// println!("current parent num local vars {}", node.num_local_vars);

	// run inlining again
	// if num_inlined_vars > 0 {
	// 	func_expr = inline_variables_and_fold(&func_expr, context)?;
	// }

	let shift_read_arg = node.num_local_vars;
	let shift_read_var = node.num_local_vars + func_expr.num_local_vars;

	// move arguments to parents local vars
	let arg_ranges = get_n_previous_exprs(&node.ops, *cur_idx - 1, new_num_args);
	let arg_range = arg_ranges.last().map(|(start, _)| *start).unwrap_or(0)
		..arg_ranges.first().map(|(_, end)| *end + 1).unwrap_or(0);
	let insertion_idx = node.num_local_var_ops as usize;
	// println!("current inlining funciton :{:?}", &func_expr.ops);
	// println!("parent before pushing args {:?}", &node.ops);
	// println!(
	// 	"pushing ARGS to parent local at {insertion_idx} {:?}",
	// 	&node.ops[arg_range.clone()]
	// );
	let args = node.ops.drain(arg_range.clone()).collect::<Vec<FlatOperator<F>>>();
	node.ops.splice(insertion_idx..insertion_idx, args);
	// println!("parent after pushing args {:?}", &node.ops);
	node.num_local_vars += new_num_args as u32;
	node.num_local_var_ops += arg_range.len() as u32;

	// update references to params and vars in the function being inlined
	func_expr.iter_mut(&mut |op| {
		if let FlatOperator::ReadParam { inverse_index } = op {
			*op = FlatOperator::ReadLocalVar { idx: shift_read_arg + (new_num_args as u32 - *inverse_index) };
		} else if let FlatOperator::ReadLocalVar { idx } = op {
			*idx += shift_read_var;
		}
	});

	// move inlinded function local vars to outer function local vars
	let insertion_idx = (node.num_local_var_ops) as usize;
	// println!(
	// 	"pushing vars to parent local at {insertion_idx} {:?}",
	// 	&func_expr.ops[0..func_expr.num_local_var_ops as usize]
	// );
	let vars = func_expr.ops.drain(0..func_expr.num_local_var_ops as usize);
	node.ops.splice(insertion_idx..insertion_idx, vars);
	node.num_local_vars += func_expr.num_local_vars;
	node.num_local_var_ops += func_expr.num_local_var_ops;

	// we've added the local vars from the inlined fn
	*cur_idx += func_expr.num_local_var_ops as usize;

	if let FlatOperator::FunctionCall { identifier, .. } = node.ops[*cur_idx] {
		if identifier != fn_name {
			panic!("Expected FunctionCall {fn_name} bug found {identifier}");
		}
	} else if let FlatOperator::ReadVar { identifier } = &node.ops[*cur_idx] {
		assert_eq!(*identifier, fn_name);
		assert_eq!(arg_num, 0);
	} else {
		panic!("Expected FunctionCall but got {:?}", node.ops[*cur_idx]);
	}

	// pop the function call and replace it with the inlined function
	let num_ops = func_expr.ops.len();
	node.ops.splice(*cur_idx..=*cur_idx, func_expr.ops);
	// println!("parent after inlining {:?}", node.ops);

	*cur_idx -= 1;
	*cur_idx += num_ops;
	Ok(())
}
