use smallvec::SmallVec;

use crate::flat_node::{FlatOperator, IntegralNode};
use crate::{EvalexprFloat, FlatNode, HashMapContext};

pub fn eliminate_subexpressions<F: EvalexprFloat>(node: &mut FlatNode<F>, context: &HashMapContext<F>) {
	let mut local_vars: Vec<FlatOperator<F>> = Vec::new();
	// let mut local_vars_indices: Vec<u32> = Vec::new();
	let mut local_var_idx = node.num_local_vars;

	let mut cur_idx = node.num_local_var_ops as usize;
	while cur_idx < node.ops.len() {
		let op = &node.ops[cur_idx];
		let num_args = num_args(op);
		let mut should_eliminate = num_args > 0;

		// functions with 0 args need to be considered too
		match op {
			FlatOperator::ReadVar { identifier } | FlatOperator::ReadVarNeg { identifier } => {
				if !context.variables.contains_key(identifier) && context.functions.contains_key(identifier) {
					// function calls without brackets
					should_eliminate = true;
				}
			},
			FlatOperator::FunctionCall { .. } => {
				// calls with brackets
				should_eliminate = true;
			},
			_ => {},
		}

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

				println!(
					"exracting {:?} from {:?} to local vars and replacing with index {cur_local_var_idx}",
					&node.ops[start_idx..=cur_idx],
					&node.ops
				);

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
}

fn num_args<F: EvalexprFloat>(op: &FlatOperator<F>) -> usize {
	match op {
		FlatOperator::PushConst { .. }
		| FlatOperator::ReadVar { .. }
		| FlatOperator::ReadVarNeg { .. }
		| FlatOperator::ReadLocalVar { .. }
		| FlatOperator::ReadParam { .. }
		| FlatOperator::ReadParamNeg { .. }
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
		FlatOperator::Integral(int) => 2,

		FlatOperator::Clamp => 3,
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
) -> SmallVec<[(usize, usize); 3]> {
	let num_args = num_args(&ops[op_index]);
	if num_args == 0 {
		return SmallVec::new();
	}
	get_n_previous_exprs(ops, op_index - 1, num_args)
}
pub fn get_n_previous_exprs<F: EvalexprFloat>(
	ops: &[FlatOperator<F>], start_idx: usize, n: usize,
) -> SmallVec<[(usize, usize); 3]> {
	use smallvec::smallvec;
	let mut result = smallvec![];

	let mut cur_arg_idx = start_idx;
	for _ in 0..n {
		let start = get_operator_range(ops, cur_arg_idx, 0).unwrap();
		result.push((start, cur_arg_idx));
		cur_arg_idx = start - 1;
	}

	result
}
