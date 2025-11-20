use crate::error::{expect_function_argument_amount, EvalexprResultValue};
use crate::flat_node::{FlatOperator, IntegralNode};
use crate::{EvalexprFloat, FlatNode, HashMapContext, IStr, Stack, Value};

#[derive(Clone, PartialEq, Debug)]
/// Struct that represents expression function
pub struct ExpressionFunction<F: EvalexprFloat> {
	pub(crate) expr:     FlatNode<F>,
	pub(crate) num_args: u32,
}
impl<F: EvalexprFloat> ExpressionFunction<F> {
	/// Creates new ExpressionFunction from FlatNode and arguments
	pub fn new(mut expr: FlatNode<F>, args: &[IStr]) -> Self {
		expr.iter_mut(&mut |op| match op {
			FlatOperator::ReadVar { identifier } => {
				if let Some(idx) = args.iter().position(|e| e == identifier) {
					*op = FlatOperator::ReadParam { inverse_index: (args.len() - idx) as u32 };
				}
			},
			FlatOperator::ReadVarNeg { identifier } => {
				if let Some(idx) = args.iter().position(|e| e == identifier) {
					*op = FlatOperator::ReadParamNeg { inverse_index: (args.len() - idx) as u32 };
				}
			},
			FlatOperator::Integral(int) => match int.as_ref() {
				IntegralNode::UnpreparedExpr { expr, variable } => {
					let mut int_args = args.to_vec();
					int_args.push(*variable);
					*op = FlatOperator::Integral(Box::new(IntegralNode::PreparedFunc {
						func:            ExpressionFunction::new(expr.clone(), &int_args),
						additional_args: (0..args.len()).map(|i| (args.len() - i) as u32).collect(),
					}));
				},
				IntegralNode::PreparedFunc { .. } => {},
			},
			_ => {},
		});

		Self { expr, num_args: args.len() as u32 }
	}
	/// Returns the constant value of this node it it only contains a single PushConst operator.
	pub fn as_constant(&self) -> Option<Value<F>> { self.expr.as_constant() }

	/// Returns the number of arguments this expression function takes.
	pub fn num_args(&self) -> usize { self.num_args as usize }
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
		let value = self.unchecked_call(stack, context)?;
		stack.pop_args();
		Ok(value)
	}
}
