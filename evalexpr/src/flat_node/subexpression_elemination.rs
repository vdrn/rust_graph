use smallvec::SmallVec;

use crate::flat_node::{AdditionalArgs, ClosureNode, FlatOperator, MapOp};
use crate::{EvalexprFloat, FlatNode, HashMapContext, Value};

/// Returns `true` if any local variables were deduplicated at the end of this function.
/// This is a signal that you may want to re-run this function, as that may eliminate more
/// subexpressions.
pub fn eliminate_subexpressions<F: EvalexprFloat>(
	node: &mut FlatNode<F>, context: &HashMapContext<F>,
) -> bool {
	let mut local_vars: Vec<FlatOperator<F>> = Vec::new();

	remove_one_op_local_vars(node);

	// let mut local_vars_indices: Vec<u32> = Vec::new();
	let mut local_var_idx = node.num_local_vars;

	let mut cur_idx = node.num_local_var_ops as usize;
	while cur_idx < node.ops.len() {
		let op = &mut node.ops[cur_idx];
		let num_args = num_args(op);

		let should_eliminate = match op {
			FlatOperator::ReadVar { identifier } => {
				// function calls without brackets
				!context.variables.contains_key(identifier) && context.functions.contains_key(identifier)
			},
			FlatOperator::FunctionCall { .. } => {
				// calls with brackets
				true
			},
			_ => num_args > 0,
		};

		if should_eliminate {
			let start_idx = get_operator_range(&node.ops, cur_idx, 0).unwrap();
			let current_op_range = &node.ops[start_idx..=cur_idx];
			let mut found_matching_ranges: SmallVec<[(usize, usize); 5]> = SmallVec::new();

			// find matching ranges
			for i in (cur_idx + 1)..node.ops.len() {
				// range must start after current range ends
				if let Some(candidate_start) = get_operator_range(&node.ops, i, start_idx + 1) {
					// if candidate_start > cur_idx {
					let candidate_range = &node.ops[candidate_start..=i];
					if candidate_range == current_op_range {
						found_matching_ranges.push((candidate_start, i));
					}
				}
			}

			if !found_matching_ranges.is_empty() {
				// println!(
				//     "found matching ranges: {:?} {:?}",
				//     current_op_range,
				//     found_matching_ranges.len()
				// );
				let cur_len = cur_idx - start_idx;
				let cur_local_var_idx = local_var_idx;
				local_var_idx += 1;

				// println!(
				// 	"exracting {:?} from {:?} to local vars and replacing with index {cur_local_var_idx}",
				// 	&node.ops[start_idx..=cur_idx],
				// 	&node.ops
				// );

				let extracted = node
					.ops
					.splice(start_idx..=cur_idx, [FlatOperator::ReadLocalVar { idx: cur_local_var_idx }]);
				local_vars.extend(extracted);

				for (i, (start, end)) in found_matching_ranges.iter().enumerate() {
					let start = *start - (i + 1) * (cur_len);
					let end = *end - (i + 1) * (cur_len);
					node.ops.splice(start..=end, [FlatOperator::ReadLocalVar { idx: cur_local_var_idx }]);
				}
				cur_idx -= cur_len;
			}
		}

		cur_idx += 1;
	}

	if !local_vars.is_empty() {
		let prev_num_local_var_ops = node.num_local_var_ops;
		node.num_local_vars = local_var_idx;
		node.num_local_var_ops = prev_num_local_var_ops + local_vars.len() as u32;

		let insert_place = prev_num_local_var_ops as usize;
		node.ops.splice(insert_place..insert_place, local_vars);
	}

	// false
	let mut deduplicated = deduplicate_local_vars(node);
	deduplicated |= inline_single_ref_local_vars(node);
	deduplicated
}
/// Inlining functions can produce local variables that are just ReadParam/ReadVar.
/// Those are pointless, so we replace reads of those local vars with direct ReadParam/ReadVar,
/// and then remove those local variables.
fn remove_one_op_local_vars<F: EvalexprFloat>(node: &mut FlatNode<F>) {
	let local_var_ranges = get_n_previous_exprs(
		&node.ops,
		node.num_local_var_ops.saturating_sub(1) as usize,
		node.num_local_vars as usize,
	);
	let total_num_local_vars = node.num_local_vars as usize;
	assert!(local_var_ranges.len() == total_num_local_vars);

	for (i, (start, end)) in local_var_ranges.into_iter().enumerate() {
		if start != end {
			continue;
		}
		if !matches!(&node.ops[start], FlatOperator::ReadParam { .. } | FlatOperator::ReadVar { .. }) {
			continue;
		}

		let local_var_op = node.ops[start].clone();
		let local_var_idx = total_num_local_vars - i - 1;
		let mut closures_need_local_var = false;
		// NOTE: we cannot remove the local var if the closures use it
		{
			let mut check_closure = |closure: &mut ClosureNode<F>| {
				match closure {
					ClosureNode::Unprepared { .. } => {},
					ClosureNode::Prepared { additional_args, .. } => {
						if let AdditionalArgs::LocalVarIndices(local_var_indices) = additional_args {
							if local_var_indices.contains(&(local_var_idx as u32)) {
								closures_need_local_var = true;
								return true;
							}
						}
					},
				}
				false
			};
			for op in node.iter_mut_top_level_ops2() {
				match op {
					FlatOperator::If { true_expr, false_expr } => {
						if check_closure(true_expr) {
							break;
						}
						if let Some(false_expr) = false_expr {
							if check_closure(false_expr) {
								break;
							}
						}
					},
					FlatOperator::Integral(closure)
					| FlatOperator::Product(closure)
					| FlatOperator::Sum(closure)
					| FlatOperator::Map(MapOp::Closure(closure)) => {
						if check_closure(closure) {
							break;
						}
					},
					_ => {},
				}
			}
		}
		if closures_need_local_var {
			continue;
		}
		node.iter_mut_top_level_ops(&mut |op| {
			let FlatOperator::ReadLocalVar { idx } = op else {
				return;
			};

			match local_var_op {
				FlatOperator::ReadParam { inverse_index } => {
					if *idx == local_var_idx as u32 {
						*op = FlatOperator::ReadParam { inverse_index };
					} else if *idx > local_var_idx as u32 {
						*idx -= 1;
					}
				},
				FlatOperator::ReadVar { identifier } => {
					if *idx == local_var_idx as u32 {
						*op = FlatOperator::ReadVar { identifier };
					} else if *idx > local_var_idx as u32 {
						*idx -= 1;
					}
				},
				_ => {
					unreachable!()
				},
			};
		});
		// ranges are in reverse order, so we can remove the op without affecting further start/end
		// indices
		node.ops.remove(start);
		node.num_local_vars -= 1;
		node.num_local_var_ops -= 1;
	}
}

fn deduplicate_local_vars<F: EvalexprFloat>(node: &mut FlatNode<F>) -> bool {
	let mut local_var_ranges = get_n_previous_exprs(
		&node.ops,
		node.num_local_var_ops.saturating_sub(1) as usize,
		node.num_local_vars as usize,
	);
	local_var_ranges.reverse();
	// println!("local var ranges: {local_var_ranges:?}");

	let mut deduplicated = false;
	let mut i = 0;
	while i < local_var_ranges.len().saturating_sub(1) {
		let (start, end) = local_var_ranges[i];
		let mut j = i + 1;
		while j < local_var_ranges.len() {
			let (start2, end2) = local_var_ranges[j];
			if node.ops[start..=end] == node.ops[start2..=end2] {
				// println!(
				// 	"deduplicating local vars: source {i} to keep: {start}..={end} local_to_remove {j} range \
				// 	 {start2}..={end2} all ops : {node:?}"
				// );
				let update_cloure = |closure: &mut ClosureNode<F>| match closure {
					ClosureNode::Unprepared { .. } => {},
					ClosureNode::Prepared { additional_args, .. } => {
						if let AdditionalArgs::LocalVarIndices(local_var_indices) = additional_args {
							for idx in local_var_indices.iter_mut() {
								if *idx == j as u32 {
									*idx = i as u32;
								} else if *idx > j as u32 {
									*idx -= 1;
								}
							}
							// TODO: we can also check if the local_var_indices have duplicates now, and
							// deduplicate ReadParams in closure body as well. (need to be careful on how to
							// translate the indices to ReadParam's inverse index, as they depend on number of
							// regular closure's params!)
						}
					},
				};
				for op in node.ops[end2 + 1..].iter_mut() {
					match op {
						FlatOperator::ReadLocalVar { idx } => {
							if *idx == j as u32 {
								*idx = i as u32;
							} else if *idx > j as u32 {
								*idx -= 1;
							}
						},
						FlatOperator::Map(MapOp::Closure(closure))
						| FlatOperator::Integral(closure)
						| FlatOperator::Product(closure)
						| FlatOperator::Sum(closure) => {
							update_cloure(closure);
						},
						FlatOperator::If { true_expr, false_expr } => {
							update_cloure(true_expr);
							if let Some(false_expr) = false_expr {
								update_cloure(false_expr);
							}
						},
						_ => {},
					}
				}
				node.ops.drain(start2..=end2);
				node.num_local_vars -= 1;
				let num_removed_ops = end2 - start2 + 1;
				node.num_local_var_ops -= num_removed_ops as u32;
				deduplicated = true;
				for further_range in local_var_ranges[j..].iter_mut() {
					*further_range = (further_range.0 - num_removed_ops, further_range.1 - num_removed_ops);
				}
				local_var_ranges.remove(j);

				// println!("parent after removing local var source{i} var{j}: {node:?}");
			} else {
				j += 1;
			}
		}
		i += 1;
	}

	deduplicated
}
/// Removes local variables that are only referenced once.
/// This can happen after inlining or other subexpression eliminations.
fn inline_single_ref_local_vars<F: EvalexprFloat>(node: &mut FlatNode<F>) -> bool {
	let mut local_var_ranges = get_n_previous_exprs(
		&node.ops,
		node.num_local_var_ops.saturating_sub(1) as usize,
		node.num_local_vars as usize,
	);
	local_var_ranges.reverse();
	// println!("local var ranges: {local_var_ranges:?}");

	let mut inlined = false;
	let mut i = 0;
	'outer: while i < local_var_ranges.len() {
		let (start, end) = local_var_ranges[i];
		// println!("inlining local var {i} with staret {start} ..= {end} node {node:?}");
		let num_ops_in_lv = end - start + 1;
		// let mut j = i + 1;
		let mut ref_pos = None;
		for op_i in end + 1..node.ops.len() {
			let check_closure_references_local_var = |closure: &ClosureNode<F>| {
				match closure {
					ClosureNode::Unprepared { .. } => {},
					ClosureNode::Prepared { additional_args, .. } => {
						if let AdditionalArgs::LocalVarIndices(local_var_indices) = additional_args {
							if local_var_indices.contains(&(i as u32)) {
								return true;
							}
						}
					},
				}
				false
			};

			match &node.ops[op_i] {
				FlatOperator::ReadLocalVar { idx } => {
					if *idx == i as u32 {
						if ref_pos.is_some() {
							// multiple references, nothing to do
							i += 1;
							continue 'outer;
						} else {
							ref_pos = Some(op_i);
						}
					}
				},
				FlatOperator::Map(MapOp::Closure(closure))
				| FlatOperator::Integral(closure)
				| FlatOperator::Product(closure)
				| FlatOperator::Sum(closure) => {
					if check_closure_references_local_var(closure) {
						i += 1;
						continue 'outer;
					}
				},
				FlatOperator::If { true_expr, false_expr } => {
					if check_closure_references_local_var(true_expr) {
						i += 1;
						continue 'outer;
					}
					if let Some(false_expr) = false_expr {
						if check_closure_references_local_var(false_expr) {
							i += 1;
							continue 'outer;
						}
					}
				},
				_ => {},
			}
		}
		if let Some(ref_pos) = ref_pos {
			if let Some(target_local_var_i) =
				local_var_ranges.iter().position(|(start, end)| (*start..=*end).contains(&ref_pos))
			{
				// ref is within local vars

				// since we just moved one local var into the other, total number of local ops is just 1
				// less: ReadLocalVar in 2nd local var
				node.num_local_var_ops -= 1;

				for further_range in local_var_ranges[i + 1..target_local_var_i].iter_mut() {
					*further_range = (further_range.0 - num_ops_in_lv, further_range.1 - num_ops_in_lv);
				}
				// firsst we enlarge the range for target local var
				// -1 because will will remove another 1 in the loop below
				local_var_ranges[target_local_var_i].0 -= num_ops_in_lv - 1;

				for further_range in local_var_ranges[target_local_var_i..].iter_mut() {
					*further_range = (further_range.0 - 1, further_range.1 - 1);
				}
			} else {
				// ref is after local vars

				node.num_local_var_ops -= num_ops_in_lv as u32;
				for further_range in local_var_ranges[i + 1..].iter_mut() {
					*further_range = (further_range.0 - num_ops_in_lv, further_range.1 - num_ops_in_lv);
				}
			}
			node.num_local_vars -= 1;

			let mut drained_ops = Some(node.ops.drain(start..=end).collect::<Vec<_>>());

			let mut op_i = 0;
			while op_i < node.ops.len() {
				// for op_i in 0..node.ops.len() {
				if let FlatOperator::ReadLocalVar { idx } = &mut node.ops[op_i] {
					if *idx == i as u32 {
						node.ops.splice(
							op_i..=op_i,
							drained_ops.take().expect("we're only doing this if there is just 1 ref"),
						);
					} else if *idx > i as u32 {
						*idx -= 1;
					}
				}
				op_i += 1;
			}
			local_var_ranges.remove(i);
			// println!("inlined local var {i} ");
			inlined = true;
		} else {
			i += 1;
		}
	}

	inlined
}

fn num_args<F: EvalexprFloat>(op: &FlatOperator<F>) -> usize {
	match op {
		FlatOperator::PushConst { .. }
		| FlatOperator::ReadVar { .. }
		| FlatOperator::ReadLocalVar { .. }
		| FlatOperator::ReadParam { .. }
		| FlatOperator::WriteVar { .. } => 0,

		FlatOperator::Neg
		| FlatOperator::Not
		| FlatOperator::MulConst { .. }
		| FlatOperator::AddConst { .. }
		| FlatOperator::DivConst { .. }
		| FlatOperator::ConstDiv { .. }
		| FlatOperator::SubConst { .. }
		| FlatOperator::ConstSub { .. }
		| FlatOperator::ExpConst { .. }
		| FlatOperator::ModConst { .. } => 1,

		FlatOperator::Add
		| FlatOperator::Sub
		| FlatOperator::Mul
		| FlatOperator::Div
		| FlatOperator::Mod
		| FlatOperator::Exp
		| FlatOperator::Eq
		| FlatOperator::Neq
		| FlatOperator::Gt
		| FlatOperator::Lt
		| FlatOperator::Geq
		| FlatOperator::Leq
		| FlatOperator::And
		| FlatOperator::Or
		| FlatOperator::Assign
		| FlatOperator::AddAssign
		| FlatOperator::SubAssign
		| FlatOperator::MulAssign
		| FlatOperator::DivAssign
		| FlatOperator::ModAssign
		| FlatOperator::ExpAssign
		| FlatOperator::AndAssign
		| FlatOperator::OrAssign => 2,

		FlatOperator::Tuple { len } | FlatOperator::Chain { len } => *len as usize,
		FlatOperator::FunctionCall { arg_num, .. } => *arg_num as usize,
		FlatOperator::AddN { n }
		| FlatOperator::SubN { n }
		| FlatOperator::MulN { n }
		| FlatOperator::DivN { n } => *n as usize,

		FlatOperator::MulAdd
		| FlatOperator::MulSub
		| FlatOperator::DivAdd
		| FlatOperator::DivSub
		| FlatOperator::AddMul
		| FlatOperator::SubMul
		| FlatOperator::AddDiv
		| FlatOperator::SubDiv
		| FlatOperator::MulDiv
		| FlatOperator::DivMul => 3,

		FlatOperator::Square
		| FlatOperator::Cube
		| FlatOperator::Sqrt
		| FlatOperator::Cbrt
		| FlatOperator::Abs
		| FlatOperator::Floor
		| FlatOperator::Round
		| FlatOperator::Ceil
		| FlatOperator::Ln
		| FlatOperator::Log2
		| FlatOperator::Log10
		| FlatOperator::ExpE
		| FlatOperator::Exp2
		| FlatOperator::Cos
		| FlatOperator::Acos
		| FlatOperator::CosH
		| FlatOperator::AcosH
		| FlatOperator::Sin
		| FlatOperator::Asin
		| FlatOperator::SinH
		| FlatOperator::AsinH
		| FlatOperator::Tan
		| FlatOperator::Atan
		| FlatOperator::TanH
		| FlatOperator::AtanH
		| FlatOperator::Signum
		| FlatOperator::Factorial => 1,

		FlatOperator::Min
		| FlatOperator::Max
		| FlatOperator::Log
		| FlatOperator::Atan2
		| FlatOperator::Hypot
		| FlatOperator::Gcd => 2,

		FlatOperator::Range => 2,
		FlatOperator::RangeWithStep => 3,

		FlatOperator::Sum { .. } | FlatOperator::Product { .. } => 1,
		FlatOperator::Integral(_) => 2,

		FlatOperator::Clamp => 3,
		FlatOperator::AccessX | FlatOperator::AccessY => 1,
		FlatOperator::AccessIndex { .. } => 1,
		FlatOperator::Access => 2,
		FlatOperator::Map(_) => 1,
		FlatOperator::If { .. } => 1,
	}
}
fn get_operator_range<F: EvalexprFloat>(
	ops: &[FlatOperator<F>], index: usize, min_start: usize,
) -> Option<usize> {
	let mut args_needed = num_args(&ops[index]);
	if args_needed == 0 {
		return Some(index);
	}

	let mut current = index;

	while args_needed > 0 {
		current -= 1;

		if current < min_start {
			return None;
		}

		let num_args = num_args(&ops[current]);

		if num_args == 0 {
			args_needed -= 1;
		} else {
			args_needed += num_args - 1;
		}
	}

	Some(current)
}
pub fn get_arg_ranges<F: EvalexprFloat>(
	ops: &[FlatOperator<F>], op_index: usize,
) -> SmallVec<[(usize, usize); 5]> {
	let num_args = num_args(&ops[op_index]);
	if num_args == 0 {
		return SmallVec::new();
	}
	get_n_previous_exprs(ops, op_index - 1, num_args)
}
pub fn get_n_previous_exprs<F: EvalexprFloat>(
	ops: &[FlatOperator<F>], start_idx: usize, n: usize,
) -> SmallVec<[(usize, usize); 5]> {
	use smallvec::smallvec;
	let mut result = smallvec![];

	let mut cur_arg_idx = start_idx;
	for i in 0..n {
		let start = get_operator_range(ops, cur_arg_idx, 0).unwrap();
		result.push((start, cur_arg_idx));
		cur_arg_idx = match start.checked_sub(1) {
			Some(idx) => idx,
			None => {
				assert!(i == n - 1);
				0
			},
		};
	}

	result
}
pub fn range_as_const<F: EvalexprFloat>(ops: &[FlatOperator<F>], range: (usize, usize)) -> Option<Value<F>> {
	if range.0 == range.1 {
		if let FlatOperator::PushConst { value } = &ops[range.0] {
			return Some(value.clone());
		}
	}
	None
}
