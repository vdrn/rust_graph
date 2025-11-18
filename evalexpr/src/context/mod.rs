//! A context defines methods to retrieve variable values and call functions for literals in an expression tree.
//! If mutable, it also allows to assign to variables.
//!
//! This crate implements two basic variants, the `EmptyContext`, that returns `None` for each identifier and cannot be manipulated, and the `HashMapContext`, that stores its mappings in hash maps.
//! The HashMapContext is type-safe and returns an error if the user tries to assign a value of a different type than before to an identifier.

use std::{iter, marker::PhantomData};

use crate::{
    error::EvalexprResultValue,
    EvalexprFloat,
    function::Function,
    value::{
        numeric_types::{default_numeric_types::DefaultNumericTypes },
        value_type::ValueType,
        Value,
    },
    EvalexprError, EvalexprResult, IStr, IStrMap, Stack,
};

mod predefined;

pub(crate) trait UncheckedCallFunction<NumericTypes: EvalexprFloat> {
    fn unchecked_call_function(
        &self,
        stack: &mut Stack<NumericTypes>,
        identifier: IStr,
    ) -> EvalexprResultValue<NumericTypes>;
}
/// An immutable context.
#[allow(private_bounds)]
pub trait Context: UncheckedCallFunction<Self::NumericTypes> {
    /// The numeric types used for evaluation.
    type NumericTypes: EvalexprFloat;

    /// Returns the value that is linked to the given identifier.
    fn get_value(&self, identifier: IStr) -> Option<&Value<Self::NumericTypes>>;

    /// Calls the function that is linked to the given identifier with the given argument.
    /// If no function with the given identifier is found, this method returns `EvalexprError::FunctionIdentifierNotFound`.
    fn call_function(
        &self,
        stack: &mut Stack<Self::NumericTypes>,
        context: &Self,
        identifier: IStr,
        argument: &[Value<Self::NumericTypes>],
    ) -> EvalexprResultValue<Self::NumericTypes>;

    /// Checks if builtin functions are disabled.
    fn are_builtin_functions_disabled(&self) -> bool;

    /// Disables builtin functions if `disabled` is `true`, and enables them otherwise.
    /// If the context does not support enabling or disabling builtin functions, an error is returned.
    fn set_builtin_functions_disabled(
        &mut self,
        disabled: bool,
    ) -> EvalexprResult<(), Self::NumericTypes>;
}

/// A context that allows to assign to variables.
pub trait ContextWithMutableVariables: Context {
    /// Sets the variable with the given identifier to the given value.
    fn set_value(
        &mut self,
        _identifier: IStr,
        _value: Value<Self::NumericTypes>,
    ) -> EvalexprResult<(), Self::NumericTypes> {
        Err(EvalexprError::ContextNotMutable)
    }

    /// Removes the variable with the given identifier from the context.
    fn remove_value(
        &mut self,
        _identifier: IStr,
    ) -> EvalexprResult<Option<Value<Self::NumericTypes>>, Self::NumericTypes> {
        Err(EvalexprError::ContextNotMutable)
    }
}

/// A context that allows to assign to function identifiers.
pub trait ContextWithMutableFunctions: Context {
    /// Sets the function with the given identifier to the given function.
    fn set_function(
        &mut self,
        _identifier: IStr,
        _function: Function<Self::NumericTypes, Self>,
    ) -> EvalexprResult<(), Self::NumericTypes>
    where
        Self: Sized,
    {
        Err(EvalexprError::ContextNotMutable)
    }
}

/// A context that allows to iterate over its variable names with their values.
pub trait IterateVariablesContext: Context {
    /// The iterator type for iterating over variable name-value pairs.
    type VariableIterator<'a>: Iterator<Item = (IStr, Value<Self::NumericTypes>)>
    where
        Self: 'a;
    /// The iterator type for iterating over variable names.
    type VariableNameIterator<'a>: Iterator<Item = IStr>
    where
        Self: 'a;

    /// Returns an iterator over pairs of variable names and values.
    fn iter_variables(&self) -> Self::VariableIterator<'_>;

    /// Returns an iterator over variable names.
    fn iter_variable_names(&self) -> Self::VariableNameIterator<'_>;
}

/*/// A context that allows to retrieve functions programmatically.
pub trait GetFunctionContext: Context {
    /// Returns the function that is linked to the given identifier.
    ///
    /// This might not be possible for all functions, as some might be hard-coded.
    /// In this case, a special error variant should be returned (Not yet implemented).
    fn get_function(&self, identifier: &str) -> Option<&Function>;
}*/

/// A context that returns `None` for each identifier.
/// Builtin functions are disabled and cannot be enabled.
#[derive(Debug)]
pub struct EmptyContext<NumericTypes>(PhantomData<NumericTypes>);

impl<NumericTypes: EvalexprFloat> UncheckedCallFunction<NumericTypes>
    for EmptyContext<NumericTypes>
{
    fn unchecked_call_function(
        &self,
        _stack: &mut Stack<NumericTypes>,
        _identifier: IStr,
    ) -> EvalexprResultValue<NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            "EmptyContext".to_string(),
        ))
    }
}
impl<NumericTypes: EvalexprFloat> Context for EmptyContext<NumericTypes> {
    type NumericTypes = NumericTypes;

    fn get_value(&self, _identifier: IStr) -> Option<&Value<Self::NumericTypes>> {
        None
    }

    fn call_function(
        &self,
        _stack: &mut Stack<Self::NumericTypes>,
        _context: &Self,
        identifier: IStr,
        _argument: &[Value<Self::NumericTypes>],
    ) -> EvalexprResultValue<Self::NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            identifier.to_string(),
        ))
    }

    /// Builtin functions are always disabled for `EmptyContext`.
    fn are_builtin_functions_disabled(&self) -> bool {
        true
    }

    /// Builtin functions can't be enabled for `EmptyContext`.
    fn set_builtin_functions_disabled(
        &mut self,
        disabled: bool,
    ) -> EvalexprResult<(), Self::NumericTypes> {
        if disabled {
            Ok(())
        } else {
            Err(EvalexprError::BuiltinFunctionsCannotBeEnabled)
        }
    }
}

impl<NumericTypes: EvalexprFloat> IterateVariablesContext for EmptyContext<NumericTypes> {
    type VariableIterator<'a>
        = iter::Empty<(IStr, Value<Self::NumericTypes>)>
    where
        Self: 'a;
    type VariableNameIterator<'a>
        = iter::Empty<IStr>
    where
        Self: 'a;

    fn iter_variables(&self) -> Self::VariableIterator<'_> {
        iter::empty()
    }

    fn iter_variable_names(&self) -> Self::VariableNameIterator<'_> {
        iter::empty()
    }
}

impl<NumericTypes> Default for EmptyContext<NumericTypes> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
impl<NumericTypes: EvalexprFloat> UncheckedCallFunction<NumericTypes>
    for EmptyContextWithBuiltinFunctions<NumericTypes>
{
    fn unchecked_call_function(
        &self,
        _stack: &mut Stack<NumericTypes>,
        _identifier: IStr,
    ) -> EvalexprResultValue<NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            "EmptyContextWithBuiltinFunctions".to_string(),
        ))
    }
}

/// A context that returns `None` for each identifier.
/// Builtin functions are enabled and cannot be disabled.
#[derive(Debug)]
pub struct EmptyContextWithBuiltinFunctions<NumericTypes>(PhantomData<NumericTypes>);

impl<NumericTypes: EvalexprFloat> Context
    for EmptyContextWithBuiltinFunctions<NumericTypes>
{
    type NumericTypes = NumericTypes;

    fn get_value(&self, _identifier: IStr) -> Option<&Value<Self::NumericTypes>> {
        None
    }

    fn call_function(
        &self,
        _stack: &mut Stack<Self::NumericTypes>,
        _context: &Self,
        identifier: IStr,
        _argument: &[Value<Self::NumericTypes>],
    ) -> EvalexprResultValue<Self::NumericTypes> {
        Err(EvalexprError::FunctionIdentifierNotFound(
            identifier.to_string(),
        ))
    }

    /// Builtin functions are always enabled for EmptyContextWithBuiltinFunctions.
    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    /// Builtin functions can't be disabled for EmptyContextWithBuiltinFunctions.
    fn set_builtin_functions_disabled(
        &mut self,
        disabled: bool,
    ) -> EvalexprResult<(), Self::NumericTypes> {
        if disabled {
            Err(EvalexprError::BuiltinFunctionsCannotBeDisabled)
        } else {
            Ok(())
        }
    }
}

impl<NumericTypes: EvalexprFloat> IterateVariablesContext
    for EmptyContextWithBuiltinFunctions<NumericTypes>
{
    type VariableIterator<'a>
        = iter::Empty<(IStr, Value<Self::NumericTypes>)>
    where
        Self: 'a;
    type VariableNameIterator<'a>
        = iter::Empty<IStr>
    where
        Self: 'a;

    fn iter_variables(&self) -> Self::VariableIterator<'_> {
        iter::empty()
    }

    fn iter_variable_names(&self) -> Self::VariableNameIterator<'_> {
        iter::empty()
    }
}

impl<NumericTypes> Default for EmptyContextWithBuiltinFunctions<NumericTypes> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

/// A context that stores its mappings in hash maps.
///
/// *Value and function mappings are stored independently, meaning that there can be a function and a value with the same identifier.*
///
/// This context is type-safe, meaning that an identifier that is assigned a value of some type once cannot be assigned a value of another type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HashMapContext<NumericTypes: EvalexprFloat = DefaultNumericTypes> {
    pub(crate ) variables: IStrMap<Value<NumericTypes>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate ) functions: IStrMap<Function<NumericTypes, HashMapContext<NumericTypes>>>,

    /// True if builtin functions are disabled.
    without_builtin_functions: bool,
}

impl<NumericTypes: EvalexprFloat> HashMapContext<NumericTypes> {
    /// Constructs a `HashMapContext` with no mappings.
    pub fn new() -> Self {
        Default::default()
    }

    /// Removes all variables from the context.
    /// This allows to reuse the context without allocating a new HashMap.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use evalexpr::*;
    ///
    /// let mut context = HashMapContext::<DefaultNumericTypes>::new();
    /// context.set_value("abc".into(), "def".into()).unwrap();
    /// assert_eq!(context.get_value("abc"), Some(&("def".into())));
    /// context.clear_variables();
    /// assert_eq!(context.get_value("abc"), None);
    /// ```
    pub fn clear_variables(&mut self) {
        self.variables.clear()
    }

    /// Removes all functions from the context.
    /// This allows to reuse the context without allocating a new HashMap.
    pub fn clear_functions(&mut self) {
        self.functions.clear()
    }

    /// Removes all variables and functions from the context.
    /// This allows to reuse the context without allocating a new HashMap.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use evalexpr::*;
    ///
    /// let mut context = HashMapContext::<DefaultNumericTypes>::new();
    /// context.set_value("abc".into(), "def".into()).unwrap();
    /// assert_eq!(context.get_value("abc"), Some(&("def".into())));
    /// context.clear();
    /// assert_eq!(context.get_value("abc"), None);
    /// ```
    pub fn clear(&mut self) {
        self.clear_variables();
        self.clear_functions();
    }
}
impl<NumericTypes: EvalexprFloat> UncheckedCallFunction<NumericTypes>
    for HashMapContext<NumericTypes>
{
    fn unchecked_call_function(
        &self,
        stack: &mut Stack<NumericTypes>,
        identifier: IStr,
    ) -> EvalexprResultValue<NumericTypes> {
        if let Some(function) = self.functions.get(&identifier) {
            stack.function_called()?;
            let value = function.call(stack, self);
            stack.function_returned();
            value
        } else {
            Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            ))
        }
    }
}

impl<NumericTypes: EvalexprFloat> Context for HashMapContext<NumericTypes> {
    type NumericTypes = NumericTypes;

    fn get_value(&self, identifier: IStr) -> Option<&Value<Self::NumericTypes>> {
        self.variables.get(&identifier)
    }

    fn call_function(
        &self,
        stack: &mut Stack<Self::NumericTypes>,
        context: &Self,
        identifier: IStr,
        argument: &[Value<Self::NumericTypes>],
    ) -> EvalexprResultValue<Self::NumericTypes> {
        if let Some(function) = self.functions.get(&identifier) {
            stack.push_args(argument);
            stack.function_called()?;
            let value = function.call(stack, context);
            stack.function_returned();
            stack.pop_args();
            value
        } else {
            Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            ))
        }
    }

    fn are_builtin_functions_disabled(&self) -> bool {
        self.without_builtin_functions
    }

    fn set_builtin_functions_disabled(
        &mut self,
        disabled: bool,
    ) -> EvalexprResult<(), NumericTypes> {
        self.without_builtin_functions = disabled;
        Ok(())
    }
}

impl<NumericTypes: EvalexprFloat> ContextWithMutableVariables
    for HashMapContext<NumericTypes>
{
    fn set_value(
        &mut self,
        identifier: IStr,
        value: Value<Self::NumericTypes>,
    ) -> EvalexprResult<(), NumericTypes> {
        if let Some(existing_value) = self.variables.get_mut(&identifier) {
            if ValueType::from(&existing_value) == ValueType::from(&value) {
                *existing_value = value;
                return Ok(());
            } else {
                return Err(EvalexprError::expected_type(existing_value, value));
            }
        }

        // Implicit else, because `self.variables` and `identifier` are not unborrowed in else
        self.variables.insert(identifier, value);
        Ok(())
    }

    fn remove_value(
        &mut self,
        identifier: IStr,
    ) -> EvalexprResult<Option<Value<Self::NumericTypes>>, Self::NumericTypes> {
        // Removes a value from the `self.variables`, returning the value at the key if the key was previously in the map.
        Ok(self.variables.remove(&identifier))
    }
}

impl<NumericTypes: EvalexprFloat> ContextWithMutableFunctions
    for HashMapContext<NumericTypes>
{
    fn set_function(
        &mut self,
        identifier: IStr,
        function: Function<NumericTypes, Self>,
    ) -> EvalexprResult<(), Self::NumericTypes> {
        self.functions.insert(identifier, function);
        Ok(())
    }
}

impl<NumericTypes: EvalexprFloat> IterateVariablesContext for HashMapContext<NumericTypes> {
    type VariableIterator<'a>
        = std::iter::Map<
        std::collections::hash_map::Iter<'a, IStr, Value<NumericTypes>>,
        fn((&IStr, &Value<NumericTypes>)) -> (IStr, Value<NumericTypes>),
    >
    where
        Self: 'a;
    type VariableNameIterator<'a>
        = std::iter::Cloned<std::collections::hash_map::Keys<'a, IStr, Value<NumericTypes>>>
    where
        Self: 'a;

    fn iter_variables(&self) -> Self::VariableIterator<'_> {
        self.variables
            .iter()
            .map(|(string, value)| (*string, value.clone()))
    }

    fn iter_variable_names(&self) -> Self::VariableNameIterator<'_> {
        self.variables.keys().cloned()
    }
}

impl<NumericTypes: EvalexprFloat> Default for HashMapContext<NumericTypes> {
    fn default() -> Self {
        Self {
            variables: Default::default(),
            functions: Default::default(),
            without_builtin_functions: false,
        }
    }
}

/// This macro provides a convenient syntax for creating a static context.
///
/// # Examples
///
/// ```rust
/// use evalexpr::*;
///
/// let ctx: HashMapContext<DefaultNumericTypes> = context_map! {
///     "x" => float 8,
///     "f" => Function::new(|_| Ok(Value::from_float(42.0)))
/// }.unwrap(); // Do proper error handling here
///
/// assert_eq!(eval_with_context("x + f()", &ctx), Ok(Value::from_float(50.0)));
/// ```
#[macro_export]
macro_rules! context_map {
    // Termination (allow missing comma at the end of the argument list)
    ( ($ctx:expr) $k:expr => Function::new($($v:tt)*) ) =>
        { $crate::context_map!(($ctx) $k => Function::new($($v)*),) };
    ( ($ctx:expr) $k:expr => int $v:expr ) =>
        { $crate::context_map!(($ctx) $k => int $v,)  };
    ( ($ctx:expr) $k:expr => float $v:expr ) =>
        { $crate::context_map!(($ctx) $k => float $v,)  };
    ( ($ctx:expr) $k:expr => $v:expr ) =>
        { $crate::context_map!(($ctx) $k => $v,)  };
    // Termination
    ( ($ctx:expr) ) => { Ok(()) };

    // The user has to specify a literal 'Function::new' in order to create a function
    ( ($ctx:expr) $k:expr => Function::new($($v:tt)*) , $($tt:tt)*) => {{
        $crate::ContextWithMutableFunctions::set_function($ctx, $k.into(), $crate::Function::new($($v)*))
            .and($crate::context_map!(($ctx) $($tt)*))
    }};
    // // add an integer value, and chain the eventual error with the ones in the next values
    // ( ($ctx:expr) $k:expr => int $v:expr , $($tt:tt)*) => {{
    //     $crate::ContextWithMutableVariables::set_value($ctx, $k.into(), $crate::Value::from_int($v.into()))
    //         .and($crate::context_map!(($ctx) $($tt)*))
    // }};
    // add a float value, and chain the eventual error with the ones in the next values
    ( ($ctx:expr) $k:expr => float $v:expr , $($tt:tt)*) => {{
        $crate::ContextWithMutableVariables::set_value($ctx, $crate::istr($k), $crate::Value::from_float($v.into()))
            .and($crate::context_map!(($ctx) $($tt)*))
    }};
    // add a value, and chain the eventual error with the ones in the next values
    ( ($ctx:expr) $k:expr => $v:expr , $($tt:tt)*) => {{
        $crate::ContextWithMutableVariables::set_value($ctx, $k.into(), $v.into())
            .and($crate::context_map!(($ctx) $($tt)*))
    }};

    // Create a context, then recurse to add the values in it
    ( $($tt:tt)* ) => {{
        let mut context = $crate::HashMapContext::new();
        $crate::context_map!((&mut context) $($tt)*)
            .map(|_| context)
    }};
}
