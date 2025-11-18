use std::fmt;

use crate::{
    error::EvalexprResultValue, value::numeric_types::default_numeric_types::DefaultNumericTypes,
    EvalexprFloat, HashMapContext, Stack, Value,
};

pub(crate) mod builtin;

/// A helper trait to enable cloning through `Fn` trait objects.
trait ClonableFn<F: EvalexprFloat = DefaultNumericTypes>
where
    Self: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
    Self: Send + Sync + 'static,
{
    fn dyn_clone(&self) -> Box<dyn ClonableFn<F>>;
}

impl<FN, F: EvalexprFloat> ClonableFn<F> for FN
where
    FN: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
    FN: Send + Sync + 'static,
    FN: Clone,
{
    fn dyn_clone(&self) -> Box<dyn ClonableFn<F>> {
        Box::new(self.clone()) as _
    }
}

/// A user-defined function.
/// Functions can be used in expressions by storing them in a `Context`.
///
/// # Examples
///
/// ```rust
/// use evalexpr::*;
///
/// let mut context = HashMapContext::<DefaultNumericTypes>::new();
/// context.set_function("id".into(), Function::new(|argument| {
///     Ok(argument.clone())
/// })).unwrap(); // Do proper error handling here
/// assert_eq!(eval_with_context("id(4)", &context), Ok(Value::from_float(4.0)));
/// ```
pub struct Function<F: EvalexprFloat> {
    function: Box<dyn ClonableFn<F>>,
}

impl<F: EvalexprFloat> Clone for Function<F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function.dyn_clone(),
        }
    }
}

impl<F: EvalexprFloat + 'static> Function<F> {
    /// Creates a user-defined function.
    ///
    /// The `function` is boxed for storage.
    pub fn new<FN>(function: FN) -> Self
    where
        FN: Fn(&mut Stack<F>, &HashMapContext<F>) -> EvalexprResultValue<F>,
        FN: Send + Sync + 'static,
        FN: Clone,
    {
        Self {
            function: Box::new(function) as _,
        }
    }

    pub(crate) fn call(
        &self,
        stack: &mut Stack<F>,
        context: &HashMapContext<F>,
        arguments: &[Value<F>],
    ) -> EvalexprResultValue<F> {
        stack.push_args(arguments);

        let value = self.unchecked_call(stack, context);

        stack.pop_args();
        value
    }
    pub(crate) fn unchecked_call(
        &self,
        stack: &mut Stack<F>,
        context: &HashMapContext<F>,
    ) -> EvalexprResultValue<F> {
        stack.function_called()?;
        let value = (self.function)(stack, context);
        stack.function_returned();
        value
    }
}

impl<F: EvalexprFloat> fmt::Debug for Function<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Function {{ [...] }}")
    }
}
impl<F: EvalexprFloat> fmt::Display for Function<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Function {{ [...] }}")
    }
}

/// A trait to ensure a type is `Send` and `Sync`.
/// If implemented for a type, the crate will not compile if the type is not `Send` and `Sync`.
#[allow(dead_code)]
#[doc(hidden)]
trait IsSendAndSync: Send + Sync {}

impl<F: EvalexprFloat> IsSendAndSync for Function<F> {}
