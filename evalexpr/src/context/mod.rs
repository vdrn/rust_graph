//! A context defines methods to retrieve variable values and call functions for literals in an expression tree.
//! If mutable, it also allows to assign to variables.
//!
//! This crate implements two basic variants, the `EmptyContext`, that returns `None` for each identifier and cannot be manipulated, and the `HashMapContext`, that stores its mappings in hash maps.
//! The HashMapContext is type-safe and returns an error if the user tries to assign a value of a different type than before to an identifier.

use crate::{
    error::EvalexprResultValue,
    function::{expression_function::ExpressionFunction, rust_function::RustFunction},
    value::{
        numeric_types::default_numeric_types::DefaultNumericTypes, value_type::ValueType, Value,
    },
    EvalexprError, EvalexprFloat, EvalexprResult, IStr, IStrMap, Stack,
};

//mod predefined;

/// A context that stores its mappings in hash maps.
///
/// *Value and function mappings are stored independently, meaning that there can be a function and a value with the same identifier.*
///
/// This context is type-safe, meaning that an identifier that is assigned a value of some type once cannot be assigned a value of another type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HashMapContext<F: EvalexprFloat = DefaultNumericTypes> {
    pub(crate) variables: IStrMap<Value<F>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) functions: IStrMap<RustFunction<F>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) expr_functions: IStrMap<ExpressionFunction<F>>,
}

impl<F: EvalexprFloat> HashMapContext<F> {
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
    pub fn clear_rust_functions(&mut self) {
        self.functions.clear()
    }

    /// Removes all expression functions from the context.
    /// This allows to reuse the context without allocating a new HashMap.
    pub fn clear_expression_functions(&mut self) {
        self.expr_functions.clear()
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
        self.clear_rust_functions();
        self.clear_expression_functions();
    }

    /// Returns the value that is linked to the given identifier.
    pub fn get_value(&self, identifier: IStr) -> Option<&Value<F>> {
        self.variables.get(&identifier)
    }

    /// Sets the function with the given identifier to the given function.
    pub fn set_function(&mut self, identifier: IStr, function: RustFunction<F>) {
        self.functions.insert(identifier, function);
    }

    /// Calls the function that is linked to the given identifier with the given argument.
    /// If no function with the given identifier is found, this method returns `EvalexprError::FunctionIdentifierNotFound`.
    pub fn call_function(
        &self,
        stack: &mut Stack<F>,
        context: &Self,
        identifier: IStr,
        argument: &[Value<F>],
    ) -> EvalexprResultValue<F> {
        if let Some(function) = self.functions.get(&identifier) {
            function.call(stack, context, argument)
        } else {
            Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            ))
        }
    }

    /// Sets the expression function with the given identifier to the given function.
    pub fn set_expression_function(&mut self, identifier: IStr, expr: ExpressionFunction<F>) {
        self.expr_functions.insert(identifier, expr);
    }
    // pub(crate) fn unchecked_call_expression_function(
    //     &self,
    //     stack: &mut Stack<F>,
    //     identifier: IStr,
    // ) -> EvalexprResultValue<F> {
    //     if let Some(function) = self.expr_functions.get(&identifier) {
    //         function.unchecked_call(stack, self)
    //     } else {
    //         Err(EvalexprError::FunctionIdentifierNotFound(
    //             identifier.to_string(),
    //         ))
    //     }
    // }

    /// Calls the expression function with the given identifier with the given argument.
    pub fn call_expression_function(
        &self,
        stack: &mut Stack<F>,
        identifier: IStr,
        args: &[Value<F>],
    ) -> EvalexprResultValue<F> {
        if let Some(function) = self.expr_functions.get(&identifier) {
            function.call(stack, self, args)
        } else {
            Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            ))
        }
    }

    /// Sets the variable with the given identifier to the given value.
    pub fn set_value(&mut self, identifier: IStr, value: Value<F>) -> EvalexprResult<(), F> {
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

    /// Removes the variable with the given identifier from the context.
    pub fn remove_value(&mut self, identifier: IStr) -> EvalexprResult<Option<Value<F>>, F> {
        // Removes a value from the `self.variables`, returning the value at the key if the key was previously in the map.
        Ok(self.variables.remove(&identifier))
    }

    /// Returns an iterator over pairs of variable names and values.
    pub fn iter_variables(&self) -> impl Iterator<Item = (&IStr, &Value<F>)> {
        self.variables.iter()
    }

    /// Returns an iterator over variable names.
    pub fn iter_variable_names(&self) -> impl Iterator<Item = IStr> + '_ {
        self.variables.keys().cloned()
    }
}

impl<NumericTypes: EvalexprFloat> Default for HashMapContext<NumericTypes> {
    fn default() -> Self {
        Self {
            variables: Default::default(),
            functions: Default::default(),
            expr_functions: Default::default(),
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
        $crate::HashMapContext::set_function($ctx, $k.into(), $crate::Function::new($($v)*))
            .and($crate::context_map!(($ctx) $($tt)*))
    }};
    // // add an integer value, and chain the eventual error with the ones in the next values
    // ( ($ctx:expr) $k:expr => int $v:expr , $($tt:tt)*) => {{
    //     $crate::ContextWithMutableVariables::set_value($ctx, $k.into(), $crate::Value::from_int($v.into()))
    //         .and($crate::context_map!(($ctx) $($tt)*))
    // }};
    // add a float value, and chain the eventual error with the ones in the next values
    ( ($ctx:expr) $k:expr => float $v:expr , $($tt:tt)*) => {{
        $crate::HashMapContext::set_value($ctx, $crate::istr($k), $crate::Value::from_float($v.into()))
            .and($crate::context_map!(($ctx) $($tt)*))
    }};
    // add a value, and chain the eventual error with the ones in the next values
    ( ($ctx:expr) $k:expr => $v:expr , $($tt:tt)*) => {{
        $crate::HashMapContext::set_value($ctx, $k.into(), $v.into())
            .and($crate::context_map!(($ctx) $($tt)*))
    }};

    // Create a context, then recurse to add the values in it
    ( $($tt:tt)* ) => {{
        let mut context = $crate::HashMapContext::new();
        $crate::context_map!((&mut context) $($tt)*)
            .map(|_| context)
    }};
}
