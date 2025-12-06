use core::num::NonZeroU32;

use crate::error::EvalexprResultValue;
use crate::{
	EmptyType, EvalexprError, EvalexprFloat, EvalexprResult, ExpressionFunction, HashMapContext, IStr, TupleType, Value, EMPTY_VALUE
};
mod compile;
pub(crate) mod eval;
mod function_inlining;
mod subexpression_elemination;
mod variable_inlining;

use eval::{eval_flat_node, eval_flat_node_mut};
pub(crate) use function_inlining::inline_functions;
use smallvec::SmallVec;
use variable_inlining::inline_variables_and_fold;
pub(crate) use variable_inlining::setup_jump_offsets;

pub use compile::compile_to_flat;
pub use eval::Stack;

/// Optimizes a FlatNode by inlining variables and eliminating subexpressions
/// Returns optimized FlatNode without modifying the original
pub fn optimize_flat_node<F: EvalexprFloat>(
	node: &FlatNode<F>, context: &mut HashMapContext<F>,
) -> EvalexprResult<FlatNode<F>, F> {
	let mut inlined = inline_variables_and_fold(node, context, &[])?;
	if function_inlining::inline_functions(&mut inlined, context)? > 0 {
		inlined = inline_variables_and_fold(&inlined, context, &[])?;
	};
	if subexpression_elemination::eliminate_subexpressions(&mut inlined, context) {
		// run again since we deduplicated local vars. could be a loop maybe
		subexpression_elemination::eliminate_subexpressions(&mut inlined, context);
	}
	Ok(inlined)
}

#[cold]
pub fn cold() {}

const _: () = assert!(std::mem::size_of::<FlatOperator<f64>>() == 48);
#[derive(Debug, Clone, PartialEq)]
// NOTE: while repr(C) costs us 8 bytes, it generates much nicer match in `eval_priv`
#[repr(C)]
#[allow(missing_docs)]
pub enum FlatOperator<F: EvalexprFloat> {
	// Arithmetic operators
	Add,
	Sub,
	Mul,
	Div,
	Mod,
	Exp,
	Neg,

	// Comparison operators
	Eq,
	Neq,
	Gt,
	Lt,
	Geq,
	Leq,

	// Logical operators
	And,
	Or,
	Not,

	// Assignment operators (may not be used in immutable context)
	Assign,
	AddAssign,
	SubAssign,
	MulAssign,
	DivAssign,
	ModAssign,
	ExpAssign,
	AndAssign,
	OrAssign,

	// Variable-length operators with explicit length
	/// Construct tuple from top `len` stack values
	Tuple {
		len: u32,
	},
	/// Execute `len` expressions, keep only the last result
	Chain {
		len: u32,
	},
	/// Call function with `len` arguments from stack
	FunctionCall {
		identifier: IStr,
		arg_num:    u32,
	},

	// Constants and variables
	/// Push constant value onto stack
	PushConst {
		value: Value<F>,
	},
	/// Read variable and push onto stack
	ReadVar {
		identifier: IStr,
	},
	/// Write to variable (pops value from stack)
	WriteVar {
		identifier: IStr,
	},

	// N-ary operations
	AddN {
		n: u32,
	},
	SubN {
		n: u32,
	},
	MulN {
		n: u32,
	},
	DivN {
		n: u32,
	},

	// Fused operations
	/// a*b + c
	MulAdd,
	/// a*b - c
	MulSub,
	/// a/b + c
	DivAdd,
	/// a/b - c
	DivSub,
	/// (a+b) * c
	AddMul,
	/// (a-b) * c
	SubMul,
	/// (a+b) / c
	AddDiv,
	/// (a-b) / c
	SubDiv,
	/// (a * b / c)
	MulDiv,
	/// (a / b * c)
	DivMul,
	/// (x * C or C * x)
	MulConst {
		value: Value<F>,
	},
	///(x + C or C + x)
	AddConst {
		value: Value<F>,
	},
	///(x/C)
	DivConst {
		value: Value<F>,
	},
	///(C/x)
	ConstDiv {
		value: Value<F>,
	},
	///(x - C)
	SubConst {
		value: Value<F>,
	},
	/// (C - x)
	ConstSub {
		value: Value<F>,
	},
	/// x^C
	ExpConst {
		value: Value<F>,
	},
	/// x%C
	ModConst {
		value: Value<F>,
	},
	// Specialized math operations
	Square,
	Cube,

	Sqrt,
	Cbrt,
	Abs,

	Floor,
	Round,
	Ceil,
	Ln,
	Log, // 2 params
	Log2,
	Log10,
	ExpE,
	Exp2,

	Cos,
	Acos,
	CosH,
	AcosH,

	Sin,
	Asin,
	SinH,
	AsinH,

	Tan,
	Atan,
	TanH,
	AtanH,
	Atan2, //2 params
	Hypot, // 2 params
	Signum,

	Min,   // 2 params
	Max,   // 2 params
	Clamp, // 3 params

	Factorial,
	Gcd,

	Range,
	RangeWithStep,
	/// ∑
	Sum(Box<ClosureNode<F>>),
	/// ∏
	Product(Box<ClosureNode<F>>),
	Integral(Box<ClosureNode<F>>),

	ReadLocalVar {
		idx: u32,
	},
	ReadParam {
		/// for arguments           (a,b,c)
		/// inverse indices will be (3,2,1)
		inverse_index: u32,
	},
	AccessX,
	AccessY,
	AccessIndex {
		index: u32,
	},
	Access,
	Map(MapOp<F>),

	Label {
		id: u64,
	},
	JumpIfFalse {
		id:     u64,
		offset: Option<NonZeroU32>,
	},
	Jump {
		id:     u64,
		offset: Option<NonZeroU32>,
	},
}

#[derive(Debug, Clone, PartialEq)]
pub enum MapOp<F: EvalexprFloat> {
	Closure(Box<ClosureNode<F>>),
	Func { name: IStr },
}
// #[derive(Debug, Clone, PartialEq)]
// pub enum IntegralNode<F: EvalexprFloat> {
// 	UnpreparedExpr {
// 		expr:     FlatNode<F>,
// 		variable: IStr,
// 	},
// 	PreparedFunc {
// 		func:            ExpressionFunction<F>,
// 		variable:        IStr,
// 		/// inverse_indices from outer scope
// 		additional_args: Vec<u32>,
// 	},
// }

#[derive(Debug, Clone, PartialEq)]
pub enum AdditionalArgs {
	InverseIndices(SmallVec<[u32; 4]>),
	LocalVarIndices(SmallVec<[u32; 4]>),
}
impl AdditionalArgs {
	pub fn len(&self) -> usize {
		match self {
			AdditionalArgs::InverseIndices(args) => args.len(),
			AdditionalArgs::LocalVarIndices(args) => args.len(),
		}
	}
	#[inline(always)]
	pub fn push_values_to_stack<F: EvalexprFloat>(&self, stack: &mut Stack<F>, base_stack_idx: usize) {
		match self {
			AdditionalArgs::InverseIndices(args) => {
				for arg in args.iter() {
					let value = stack.get_unchecked(base_stack_idx - *arg as usize);
					stack.push(value.clone());
				}
			},
			AdditionalArgs::LocalVarIndices(args) => {
				for arg in args.iter() {
					let value = stack.get_unchecked(base_stack_idx + *arg as usize);
					stack.push(value.clone());
				}
			},
		}
	}
}
#[derive(Debug, Clone, PartialEq)]
pub enum ClosureNode<F: EvalexprFloat> {
	Unprepared {
		expr:   FlatNode<F>,
		params: SmallVec<[IStr; 4]>,
	},
	Prepared {
		func:            ExpressionFunction<F>,
		params:          SmallVec<[IStr; 4]>,
		/// initialy inverse_indices for params from outer scope
		/// after function inlining, these are parents local var indices!
		additional_args: AdditionalArgs,
	},
}

/// Flat compiled node - linear sequence of operations
#[derive(Debug, Clone, PartialEq)]
pub struct FlatNode<F: EvalexprFloat> {
	ops:               Vec<FlatOperator<F>>,
	num_local_vars:    u32,
	num_local_var_ops: u32,
}

impl<F: EvalexprFloat> FlatNode<F> {
	pub(crate) fn ops_len(&self) -> usize { self.ops.len() }
	/// Returns the constant value of this node it it only contains a single PushConst operator.
	pub fn as_constant(&self) -> Option<Value<F>> {
		if self.ops.len() == 1 {
			if let FlatOperator::PushConst { value } = &self.ops[0] {
				return Some(value.clone());
			}
		}
		None
	}
	/// Evaluates the operator tree rooted at this node with empty context.
	pub fn eval(&self) -> EvalexprResultValue<F> {
		let context = HashMapContext::<F>::new();
		let mut stack = Stack::new();
		eval_flat_node(self, &mut stack, &context, &[])
	}
	/// Evaluates the operator tree rooted at this node with the given context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_with_context(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResultValue<F> {
		eval_flat_node(self, stack, context, &[])
	}
	/// Evaluates the operator tree rooted at this node with the given mutable context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_with_context_mut(
		&self, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
	) -> EvalexprResultValue<F> {
		eval_flat_node_mut(self, stack, context)
	}

	/// Evaluates the operator tree rooted at this node into a float with an the given context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_float_with_context(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResult<F, F> {
		match self.eval_with_context(stack, context) {
			Ok(Value::Float(float)) => Ok(float),
			Ok(value) => Err(EvalexprError::expected_float(value)),
			Err(error) => Err(error),
		}
	}
	/// Evaluates the operator tree rooted at this node with the given context and override vars
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_with_context_and_override(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, Value<F>)],
	) -> EvalexprResultValue<F> {
		eval_flat_node(self, stack, context, override_vars)
	}

	/// Evaluates the operator tree rooted at this node into a float with an the given context.
	/// If the result of the expression is an integer, it is silently converted into a float.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_float_with_context_and_override(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, Value<F>)],
	) -> EvalexprResult<F, F> {
		match self.eval_with_context_and_override(stack, context, override_vars) {
			Ok(Value::Float(float)) => Ok(float),
			Ok(value) => Err(EvalexprError::expected_float(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into a boolean with an the given context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_boolean_with_context(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResult<bool, F> {
		match self.eval_with_context(stack, context) {
			Ok(Value::Boolean(boolean)) => Ok(boolean),
			Ok(value) => Err(EvalexprError::expected_boolean(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into a tuple with an the given context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_tuple_with_context(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResult<TupleType<F>, F> {
		match self.eval_with_context(stack, context) {
			Ok(Value::Tuple(tuple)) => Ok(tuple),
			Ok(value) => Err(EvalexprError::expected_tuple(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into an empty value with an the given context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_empty_with_context(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResult<EmptyType, F> {
		match self.eval_with_context(stack, context) {
			Ok(Value::Empty) => Ok(EMPTY_VALUE),
			Ok(value) => Err(EvalexprError::expected_empty(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into a float with an the given mutable context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_float_with_context_mut(
		&self, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
	) -> EvalexprResult<F, F> {
		match self.eval_with_context_mut(stack, context) {
			Ok(Value::Float(float)) => Ok(float),
			Ok(value) => Err(EvalexprError::expected_float(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into a boolean with an the given mutable context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_boolean_with_context_mut(
		&self, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
	) -> EvalexprResult<bool, F> {
		match self.eval_with_context_mut(stack, context) {
			Ok(Value::Boolean(boolean)) => Ok(boolean),
			Ok(value) => Err(EvalexprError::expected_boolean(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into a tuple with an the given mutable context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_tuple_with_context_mut(
		&self, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
	) -> EvalexprResult<TupleType<F>, F> {
		match self.eval_with_context_mut(stack, context) {
			Ok(Value::Tuple(tuple)) => Ok(tuple),
			Ok(value) => Err(EvalexprError::expected_tuple(value)),
			Err(error) => Err(error),
		}
	}

	/// Evaluates the operator tree rooted at this node into an empty value with an the given mutable context.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_empty_with_context_mut(
		&self, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
	) -> EvalexprResult<EmptyType, F> {
		match self.eval_with_context_mut(stack, context) {
			Ok(Value::Empty) => Ok(EMPTY_VALUE),
			Ok(value) => Err(EvalexprError::expected_empty(value)),
			Err(error) => Err(error),
		}
	}
	/// Evaluates the operator tree rooted at this node into a float.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_float(&self) -> EvalexprResult<F, F> {
		let mut stack = Stack::new();
		self.eval_float_with_context_mut(&mut stack, &mut HashMapContext::new())
	}
	/// Evaluates the operator tree rooted at this node into a boolean.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_boolean(&self) -> EvalexprResult<bool, F> {
		let mut stack = Stack::new();
		self.eval_boolean_with_context_mut(&mut stack, &mut HashMapContext::new())
	}

	/// Evaluates the operator tree rooted at this node into a tuple.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_tuple(&self) -> EvalexprResult<TupleType<F>, F> {
		let mut stack = Stack::new();
		self.eval_tuple_with_context_mut(&mut stack, &mut HashMapContext::new())
	}

	/// Evaluates the operator tree rooted at this node into an empty value.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_empty(&self) -> EvalexprResult<EmptyType, F> {
		let mut stack = Stack::new();
		self.eval_empty_with_context_mut(&mut stack, &mut HashMapContext::new())
	}

	/// Returns an iterator over all nodes in this tree.
	pub fn iter(&self) -> impl Iterator<Item = &FlatOperator<F>> {
		let mut ops = Vec::new();
		for op in self.ops.iter() {
			match op {
				FlatOperator::Map(MapOp::Closure(closure))
				| FlatOperator::Integral(closure)
				| FlatOperator::Product(closure)
				| FlatOperator::Sum(closure) => match closure.as_ref() {
					ClosureNode::Unprepared { expr, params } => {
						ops.extend(expr.iter().filter(|op| match op {
							FlatOperator::ReadVar { identifier } => !params.contains(identifier),
							_ => true,
						}));
					},
					ClosureNode::Prepared { func, params, .. } => {
						ops.extend(func.expr.ops.iter().filter(|op| match op {
							FlatOperator::ReadVar { identifier } => !params.contains(identifier),
							_ => true,
						}));
					},
				},
				op => {
					ops.push(op);
				},
			}
		}
		ops.into_iter()
	}
	pub(crate) fn iter_mut_top_level_ops(&mut self, f: &mut dyn FnMut(&mut FlatOperator<F>)) {
		self.ops.iter_mut().for_each(f);
	}
	pub(crate) fn iter_mut_top_level_ops2(&mut self) -> impl Iterator<Item = &mut FlatOperator<F>> {
		self.ops.iter_mut()
	}
	// pub(crate) fn iter_mut_closure_vars(&mut self, f: &mut dyn FnMut(&mut FlatOperator<F>)) {
	// 	for op in self.ops.iter_mut() {
	// 		f(op);
	// 		let mut closure_iter = |closure: &mut ClosureNode<F>| match closure {
	// 			ClosureNode::Unprepared { expr, params } => expr.iter_mut_top_level_ops(f),
	// 			ClosureNode::Prepared { func, params, .. } => {
	// 				func.expr.iter_mut_top_level_ops(f);
	// 			},
	// 		};
	// 		match op {
	// 			FlatOperator::Map(MapOp::Closure(closure))
	// 			| FlatOperator::Integral(closure)
	// 			| FlatOperator::Product(closure)
	// 			| FlatOperator::Sum(closure) => {
	// 				closure_iter(closure);
	// 			},
	// 			FlatOperator::If { true_expr, false_expr } => {
	// 				closure_iter(true_expr);
	// 				if let Some(false_expr) = false_expr {
	// 					closure_iter(false_expr);
	// 				}
	// 			},
	// 			_ => {},
	// 		}
	// 	}
	// }
	/// Returns an iterator over all identifiers in this expression.
	/// Each occurrence of an identifier is returned separately.
	pub fn iter_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier }
			| FlatOperator::WriteVar { identifier }
			| FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
			FlatOperator::Map(MapOp::Func { name }) => Some(name.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	pub fn iter_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier } | FlatOperator::WriteVar { identifier } => {
				Some(identifier.to_str())
			},
			FlatOperator::Map(MapOp::Func { name }) => Some(name.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all read variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	pub fn iter_read_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier } => Some(identifier.to_str()),
			FlatOperator::Map(MapOp::Func { name }) => Some(name.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all write variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	pub fn iter_write_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::WriteVar { identifier } => Some(identifier.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all function identifiers in this expression.
	/// Each occurrence of a function identifier is returned separately.
	pub fn iter_function_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
			FlatOperator::Map(MapOp::Func { name }) => Some(name.to_str()),
			_ => None,
		})
	}
}
