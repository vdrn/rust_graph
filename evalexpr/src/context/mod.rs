//! A context defines methods to retrieve variable values and call functions for literals in an expression
//! tree. If mutable, it also allows to assign to variables.
//!
//! This crate implements two basic variants, the `EmptyContext`, that returns `None` for each identifier and
//! cannot be manipulated, and the `HashMapContext`, that stores its mappings in hash maps. The HashMapContext
//! is type-safe and returns an error if the user tries to assign a value of a different type than before to
//! an identifier.

use crate::error::EvalexprResultValue;
use crate::function::expression_function::ExpressionFunction;
use crate::function::rust_function::RustFunction;
use crate::value::numeric_types::default_numeric_types::DefaultNumericTypes;
use crate::value::value_type::ValueType;
use crate::value::Value;
use crate::{EvalexprError, EvalexprFloat, EvalexprResult, IStr, IStrMap, Stack};

//mod predefined;

/// A context that stores its mappings in hash maps.
///
/// *Value and function mappings are stored independently, meaning that there can be a function and a value
/// with the same identifier.*
///
/// This context is type-safe, meaning that an identifier that is assigned a value of some type once cannot be
/// assigned a value of another type.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HashMapContext<F: EvalexprFloat = DefaultNumericTypes> {
	pub(crate) variables:      IStrMap<Value<F>>,
	#[cfg_attr(feature = "serde", serde(skip))]
	pub(crate) functions:      IStrMap<RustFunction<F>>,
	#[cfg_attr(feature = "serde", serde(skip))]
	pub(crate) expr_functions: IStrMap<ExpressionFunction<F>>,
}

impl<F: EvalexprFloat> HashMapContext<F> {
  /// Compares two contexts for equality
  /// Since we cannot compare closures for equality, RustFunctions are not compared, but their
  /// names and order of insertion are.
	pub fn contexts_almost_equal(c1: &HashMapContext<F>, c2: &HashMapContext<F>) -> bool {
		let mut equal = c1.variables == c2.variables
			&& c1.expr_functions == c2.expr_functions
			&& c1.functions.len() == c2.functions.len();
		for (k1, k2) in c1.functions.keys().zip(c2.functions.keys()) {
			equal &= k1 == k2;
		}
		equal
	}
	/// Constructs a `HashMapContext` with no mappings.
	pub fn new() -> Self { Default::default() }

	/// Removes all variables from the context.
	/// This allows to reuse the context without allocating a new HashMap.
	///
	pub fn clear_variables(&mut self) { self.variables.clear() }

	/// Removes all functions from the context.
	/// This allows to reuse the context without allocating a new HashMap.
	pub fn clear_rust_functions(&mut self) { self.functions.clear() }

	/// Removes all expression functions from the context.
	/// This allows to reuse the context without allocating a new HashMap.
	pub fn clear_expression_functions(&mut self) { self.expr_functions.clear() }

	/// Removes all variables and functions from the context.
	/// This allows to reuse the context without allocating a new HashMap.
	///
	pub fn clear(&mut self) {
		self.clear_variables();
		self.clear_rust_functions();
		self.clear_expression_functions();
	}

	/// Returns the value that is linked to the given identifier.
	pub fn get_value(&self, identifier: IStr) -> Option<&Value<F>> { self.variables.get(&identifier) }

	/// Sets the function with the given identifier to the given function.
	pub fn set_function(&mut self, identifier: IStr, function: RustFunction<F>) {
		self.functions.insert(identifier, function);
	}

	/// Calls the function that is linked to the given identifier with the given argument.
	/// If no function with the given identifier is found, this method returns
	/// `EvalexprError::FunctionIdentifierNotFound`.
	pub fn call_function(
		&self, stack: &mut Stack<F>, context: &Self, identifier: IStr, argument: &[Value<F>],
	) -> EvalexprResultValue<F> {
		if let Some(function) = self.functions.get(&identifier) {
			function.call(stack, context, argument)
		} else {
			Err(EvalexprError::FunctionIdentifierNotFound(identifier.to_string()))
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
		&self, stack: &mut Stack<F>, identifier: IStr, args: &[Value<F>],
	) -> EvalexprResultValue<F> {
		if let Some(function) = self.expr_functions.get(&identifier) {
			function.call(stack, self, args)
		} else {
			Err(EvalexprError::FunctionIdentifierNotFound(identifier.to_string()))
		}
	}

	/// does constext contain a value for the given identifier
	pub fn has_value(&self, identifier: IStr) -> bool {
		self.variables.contains_key(&identifier)
			|| self.expr_functions.contains_key(&identifier)
			|| self.functions.contains_key(&identifier)
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
		// Removes a value from the `self.variables`, returning the value at the key if the key was previously
		// in the map.
		Ok(self.variables.remove(&identifier))
	}

	/// Returns an iterator over pairs of variable names and values.
	pub fn iter_variables(&self) -> impl Iterator<Item = (&IStr, &Value<F>)> { self.variables.iter() }

	/// Returns an iterator over variable names.
	pub fn iter_variable_names(&self) -> impl Iterator<Item = IStr> + '_ { self.variables.keys().cloned() }
}

impl<NumericTypes: EvalexprFloat> Default for HashMapContext<NumericTypes> {
	fn default() -> Self {
		Self {
			variables:      Default::default(),
			functions:      Default::default(),
			expr_functions: Default::default(),
		}
	}
}

/// This macro provides a convenient syntax for creating a static context.
///
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
