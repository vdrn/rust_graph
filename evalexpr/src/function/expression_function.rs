use smallvec::SmallVec;

use crate::error::{expect_function_argument_amount, EvalexprResultValue};
use crate::flat_node::{inline_functions, FlatOperator, IntegralNode};
use crate::{EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, IStr, Stack, Value};

#[derive(Clone, PartialEq, Debug)]
/// Struct that represents expression function
pub struct ExpressionFunction<F: EvalexprFloat> {
	pub(crate) expr: FlatNode<F>,
	pub(crate) args: Vec<IStr>,
}
impl<F: EvalexprFloat> ExpressionFunction<F> {
	/// Creates new ExpressionFunction from FlatNode and arguments
	pub fn new(
		mut expr: FlatNode<F>, args: &[IStr], context: &mut Option<&mut HashMapContext<F>>,
	) -> EvalexprResult<Self, F> {
		let mut has_integrals = false;
		let mut error = None;
		expr.iter_mut(&mut |op| match op {
			FlatOperator::ReadVar { identifier } => {
				if let Some(idx) = args.iter().position(|e| e == identifier) {
					*op = FlatOperator::ReadParam { inverse_index: (args.len() - idx) as u32 };
				}
			},
			FlatOperator::Integral(int) => match int.as_ref() {
				IntegralNode::UnpreparedExpr { expr, variable } => {
					has_integrals = true;
					let mut arg_names = SmallVec::<[IStr; 4]>::new();
					let mut additional_arg_indices = Vec::with_capacity(args.len());
					for (i, arg) in args.iter().enumerate() {
						if arg != variable {
							for internal_var in expr.iter_variable_identifiers() {
								if internal_var == arg.to_str()
									&& arg_names.iter().all(|e| e.to_str() != internal_var)
								{
									arg_names.push(*arg);
									additional_arg_indices.push((args.len() - i) as u32);
								}
							}
							// arg_names.push(*arg);
							// additional_arg_indices.push((args.len() - i) as u32);
						}
					}
					arg_names.push(*variable);

					let func = match ExpressionFunction::new(expr.clone(), &arg_names, context) {
						Ok(func) => func,
						Err(e) => {
							error = Some(e);
							return;
						},
					};
					*op = FlatOperator::Integral(Box::new(IntegralNode::PreparedFunc {
						func,
						variable: *variable,
						additional_args: additional_arg_indices,
					}));
				},
				IntegralNode::PreparedFunc { .. } => {
					has_integrals = true;
				},
			},
			_ => {},
		});
		if let Some(error) = error {
			return Err(error);
		}
		if has_integrals {
			if let Some(context) = context {
				inline_functions(&mut expr, context)?;
			}
		}

		Ok(Self { expr, args: args.to_vec() })
	}
  /// number of top level ops
  pub fn ops_len(&self) -> usize { self.expr.ops_len() }
	/// Returns the constant value of this node it it only contains a single PushConst operator.
	pub fn as_constant(&self) -> Option<Value<F>> { self.expr.as_constant() }

	/// Returns the arguments this expression function takes.
	pub fn args(&self) -> &[IStr] { &self.args }
	/// Returns the number of arguments this expression function takes.
	pub fn num_args(&self) -> usize { self.args.len() }
	pub(crate) fn unchecked_call(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>,
	) -> EvalexprResultValue<F> {
		expect_function_argument_amount(stack.num_args(), self.num_args())?;

		stack.function_called()?;
		let value = self.expr.eval_with_context(stack, context);
		stack.function_returned();
		value
	}
	///. Calls the expression function with the given arguments.
	pub fn call(
		&self, stack: &mut Stack<F>, context: &HashMapContext<F>, args: &[Value<F>],
	) -> EvalexprResultValue<F> {
		stack.push_args(args);
		let value = self.unchecked_call(stack, context);
		stack.pop_args();
		value
	}
}
