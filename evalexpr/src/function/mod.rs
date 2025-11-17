use std::fmt;

use crate::{
    error::EvalexprResultValue, value::numeric_types::default_numeric_types::DefaultNumericTypes,
    Context, EvalexprFloat, Stack,
};

pub(crate) mod builtin;

/// A helper trait to enable cloning through `Fn` trait objects.
trait ClonableFn<
    C: Context<NumericTypes = NumericTypes>,
    NumericTypes: EvalexprFloat = DefaultNumericTypes,
> where
    Self: Fn(&mut Stack<NumericTypes>, &C) -> EvalexprResultValue<NumericTypes>,
    Self: Send + Sync + 'static,
{
    fn dyn_clone(&self) -> Box<dyn ClonableFn<C, NumericTypes>>;
}

impl<F, NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>>
    ClonableFn<C, NumericTypes> for F
where
    F: Fn(&mut Stack<NumericTypes>, &C) -> EvalexprResultValue<NumericTypes>,
    F: Send + Sync + 'static,
    F: Clone,
{
    fn dyn_clone(&self) -> Box<dyn ClonableFn<C, NumericTypes>> {
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
pub struct Function<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>> {
    function: Box<dyn ClonableFn<C, NumericTypes>>,
}

impl<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes> + 'static> Clone
    for Function<NumericTypes, C>
{
    fn clone(&self) -> Self {
        Self {
            function: self.function.dyn_clone(),
        }
    }
}

impl<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>>
    Function<NumericTypes, C>
{
    /// Creates a user-defined function.
    ///
    /// The `function` is boxed for storage.
    pub fn new<F>(function: F) -> Self
    where
        F: Fn(&mut Stack<NumericTypes>, &C) -> EvalexprResultValue<NumericTypes>,
        F: Send + Sync + 'static,
        F: Clone,
    {
        Self {
            function: Box::new(function) as _,
        }
    }

    pub(crate) fn call(
        &self,
        stack: &mut Stack<NumericTypes>,
        context: &C,
    ) -> EvalexprResultValue<NumericTypes> {
        (self.function)(stack, context)
    }
}

impl<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>> fmt::Debug
    for Function<NumericTypes, C>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Function {{ [...] }}")
    }
}
impl<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>> fmt::Display
    for Function<NumericTypes, C>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "Function {{ [...] }}")
    }
}

/// A trait to ensure a type is `Send` and `Sync`.
/// If implemented for a type, the crate will not compile if the type is not `Send` and `Sync`.
#[allow(dead_code)]
#[doc(hidden)]
trait IsSendAndSync: Send + Sync {}

impl<NumericTypes: EvalexprFloat, C: Context<NumericTypes = NumericTypes>> IsSendAndSync
    for Function<NumericTypes, C>
{
}
