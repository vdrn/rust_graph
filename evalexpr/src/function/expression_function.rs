use crate::{EvalexprFloat, FlatNode, HashMapContext, IStr, Stack, Value, error::{EvalexprResultValue, expect_function_argument_amount}, flat_node::FlatOperator};


#[derive(Clone, Debug)]
/// Struct that represents expression function
pub struct ExpressionFunction<F: EvalexprFloat> {
    expr: FlatNode<F>,
    args: Box<[IStr]>,
}
impl<F: EvalexprFloat> ExpressionFunction<F> {
    /// Creates new ExpressionFunction from FlatNode and arguments
    pub fn new(mut expr: FlatNode<F>, args: Vec<IStr>) -> Self {
        expr.iter_mut(&mut |op| {
            if let FlatOperator::ReadVar { identifier } | FlatOperator::ReadVarNeg { identifier } =
                op
            {
                if let Some(idx) = args.iter().position(|e| e == identifier) {
                    *op = FlatOperator::ReadParam {
                        inverse_index: (args.len() - idx) as u32,
                    };
                }
            }
        });

        Self {
            expr,
            args: args.into_boxed_slice(),
        }
    }
    /// Returns the constant value of this node it it only contains a single PushConst operator.
    pub fn as_constant(&self) -> Option<Value<F>> {
        self.expr.as_constant()
    }

    /// Returns the number of arguments this expression function takes.
    pub fn arg_num(&self) -> usize {
        self.args.len()
    }
    pub(crate) fn unchecked_call(
        &self,
        stack: &mut Stack<F>,
        context: &HashMapContext<F>,
    ) -> EvalexprResultValue<F> {
        expect_function_argument_amount(stack.num_args(), self.args.len())?;

        stack.function_called()?;
        let value = self.expr.eval_with_context(stack, context);
        stack.function_returned();
        value
    }
    ///. Calls the expression function with the given arguments.
    pub fn call(
        &self,
        stack: &mut Stack<F>,
        context: &HashMapContext<F>,
        args: &[Value<F>],
    ) -> EvalexprResultValue<F> {
        stack.push_args(args);
        let value = self.unchecked_call(stack, context)?;
        stack.pop_args();
        Ok(value)
    }
}
