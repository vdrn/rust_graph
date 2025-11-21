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
use variable_inlining::inline_variables_and_fold;

pub use compile::compile_to_flat;
pub use eval::Stack;
/// Optimizes a FlatNode by inlining variables and eliminating subexpressions
/// Returns optimized FlatNode without modifying the original
pub fn optimize_flat_node<F: EvalexprFloat>(
	node: &FlatNode<F>, context: &mut HashMapContext<F>,
) -> EvalexprResult<FlatNode<F>, F> {
	let mut inlined = inline_variables_and_fold(node, context)?;
	if function_inlining::inline_functions(&mut inlined, context)? > 0 {
		inlined = inline_variables_and_fold(&inlined, context)?;
	};
	subexpression_elemination::eliminate_subexpressions(&mut inlined, context);
	// println!("optimized {:?}", inlined);
	Ok(inlined)
}

#[cold]
pub fn cold() {}
// pub fn unlikely(x: bool) -> bool {
//     if x {
//         cold()
//     }
//     x
// }

const _: () = assert!(std::mem::size_of::<FlatOperator<f64>>() == 48);
#[derive(Debug, Clone, PartialEq)]
// NOTE: while repr(C) costs us 8 bytes, it generates much nicer match in `eval_priv`
#[repr(C)]
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
	/// Reads a variable and pused `-value` to the stack
	ReadVarNeg {
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
		value: F,
	},
	///(x + C or C + x)
	AddConst {
		value: F,
	},
	///(x/C)
	DivConst {
		value: F,
	},
	///(C/x)
	ConstDiv {
		value: F,
	},
	///(x - C)
	SubConst {
		value: F,
	},
	/// (C - x)
	ConstSub {
		value: F,
	},
	/// x^C
	ExpConst {
		value: F,
	},
	/// x%C
	ModConst {
		value: F,
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

	Range,
	RangeWithStep,
	/// ∑
	Sum {
		variable: IStr,
		expr:     Box<FlatNode<F>>,
	},
	/// ∏
	Product {
		variable: IStr,
		expr:     Box<FlatNode<F>>,
	},
	Integral(Box<IntegralNode<F>>),

	ReadLocalVar {
		idx: u32,
	},
	ReadParam {
		/// for arguments           (a,b,c)
		/// inverse indices will be (3,2,1)
		inverse_index: u32,
	},
	ReadParamNeg {
		/// for arguments           (a,b,c)
		/// inverse indices will be (3,2,1)
		inverse_index: u32,
	},
}
#[derive(Debug, Clone, PartialEq)]
pub enum IntegralNode<F: EvalexprFloat> {
	UnpreparedExpr {
		expr:     FlatNode<F>,
		variable: IStr,
	},
	PreparedFunc {
		func:            ExpressionFunction<F>,
		variable:        IStr,
		/// inverse_indices from outer scope
		additional_args: Vec<u32>,
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
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, F)],
	) -> EvalexprResultValue<F> {
		eval_flat_node(self, stack, context, override_vars)
	}

	/// Evaluates the operator tree rooted at this node into a float with an the given context.
	/// If the result of the expression is an integer, it is silently converted into a float.
	///
	/// Fails, if one of the operators in the expression tree fails.
	pub fn eval_float_with_context_and_override(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, F)],
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
				FlatOperator::Product { expr, variable, .. } | FlatOperator::Sum { expr, variable, .. } => {
					ops.extend(expr.iter().filter(|op| match op {
						FlatOperator::ReadVar { identifier } => identifier != variable,
						_ => true,
					}));
				},
				FlatOperator::Integral(int) => match int.as_ref() {
					IntegralNode::UnpreparedExpr { expr, variable } => {
						ops.extend(expr.iter().filter(|op| match op {
							FlatOperator::ReadVar { identifier } => identifier != variable,
							_ => true,
						}));
					},
					IntegralNode::PreparedFunc { func, variable, .. } => {
						ops.extend(func.expr.ops.iter().filter(|op| match op {
							FlatOperator::ReadVar { identifier } => identifier != variable,
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
	pub(crate) fn iter_mut(&mut self, f: &mut dyn FnMut(&mut FlatOperator<F>)) {
		for op in self.ops.iter_mut() {
			match op {
				FlatOperator::Product { expr, .. } | FlatOperator::Sum { expr, .. } => {
					expr.iter_mut(f);
				},
				op => {
					f(op);
				},
			}
		}
	}
	/// Returns an iterator over all identifiers in this expression.
	/// Each occurrence of an identifier is returned separately.
	///
	/// # Examples
	///
	/// ```rust
	/// use evalexpr::*;
	///
	/// let tree = build_operator_tree::<DefaultNumericTypes>("a + b + c * f()").unwrap(); // Do proper error handling here
	/// let mut iter = tree.iter_identifiers();
	/// assert_eq!(iter.next(), Some("a"));
	/// assert_eq!(iter.next(), Some("b"));
	/// assert_eq!(iter.next(), Some("c"));
	/// assert_eq!(iter.next(), Some("f"));
	/// assert_eq!(iter.next(), None);
	/// ```
	pub fn iter_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier }
			| FlatOperator::ReadVarNeg { identifier }
			| FlatOperator::WriteVar { identifier }
			| FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	///
	/// # Examples
	///
	/// ```rust
	/// use evalexpr::*;
	///
	/// let tree = build_operator_tree::<DefaultNumericTypes>("a + f(b + c)").unwrap(); // Do proper error handling here
	/// let mut iter = tree.iter_variable_identifiers();
	/// assert_eq!(iter.next(), Some("a"));
	/// assert_eq!(iter.next(), Some("b"));
	/// assert_eq!(iter.next(), Some("c"));
	/// assert_eq!(iter.next(), None);
	/// ```
	pub fn iter_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier }
			| FlatOperator::ReadVarNeg { identifier }
			| FlatOperator::WriteVar { identifier } => Some(identifier.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all read variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	///
	/// # Examples
	///
	/// ```rust
	/// use evalexpr::*;
	///
	/// let tree = build_operator_tree::<DefaultNumericTypes>("d = a + f(b + c)").unwrap(); // Do proper error handling here
	/// let mut iter = tree.iter_read_variable_identifiers();
	/// assert_eq!(iter.next(), Some("a"));
	/// assert_eq!(iter.next(), Some("b"));
	/// assert_eq!(iter.next(), Some("c"));
	/// assert_eq!(iter.next(), None);
	/// ```
	pub fn iter_read_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::ReadVar { identifier } => Some(identifier.to_str()),
			FlatOperator::ReadVarNeg { identifier } => Some(identifier.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all write variable identifiers in this expression.
	/// Each occurrence of a variable identifier is returned separately.
	///
	/// # Examples
	///
	/// ```rust
	/// use evalexpr::*;
	///
	/// let tree = build_operator_tree::<DefaultNumericTypes>("d = a + f(b + c)").unwrap(); // Do proper error handling here
	/// let mut iter = tree.iter_write_variable_identifiers();
	/// assert_eq!(iter.next(), Some("d"));
	/// assert_eq!(iter.next(), None);
	/// ```
	pub fn iter_write_variable_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::WriteVar { identifier } => Some(identifier.to_str()),
			_ => None,
		})
	}
	/// Returns an iterator over all function identifiers in this expression.
	/// Each occurrence of a function identifier is returned separately.
	///
	/// # Examples
	///
	/// ```rust
	/// use evalexpr::*;
	///
	/// let tree = build_operator_tree::<DefaultNumericTypes>("a + f(b + c)").unwrap(); // Do proper error handling here
	/// let mut iter = tree.iter_function_identifiers();
	/// assert_eq!(iter.next(), Some("f"));
	/// assert_eq!(iter.next(), None);
	/// ```
	pub fn iter_function_identifiers(&self) -> impl Iterator<Item = &str> {
		self.iter().filter_map(|node| match node {
			FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
			_ => None,
		})
	}
}
