use crate::error::EvalexprResultValue;
use crate::flat_node::{compile_to_flat, Stack};
use crate::value::numeric_types::default_numeric_types::DefaultNumericTypes;
use crate::value::TupleType;
use crate::{
	optimize_flat_node, token, tree, EmptyType, EvalexprError, EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, Node, Value, EMPTY_VALUE
};

/// Evaluate the given expression string.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval(string: &str) -> EvalexprResultValue {
	eval_with_context_mut(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}
/// Used for testing, optimizations are not worth it for one off evaluations.
pub fn eval_optimized(string: &str) -> EvalexprResultValue {
	eval_optimized_with_context(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}

/// Evaluate the given expression string with the given context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_with_context<F: EvalexprFloat>(
	string: &str, context: &HashMapContext<F>,
) -> EvalexprResultValue<F> {
	let node = tree::tokens_to_operator_tree(token::tokenize(string)?)?;
	let compiled_node: FlatNode<F> = compile_to_flat(node)?;
	let mut stack = Stack::new();

	compiled_node.eval_with_context(&mut stack, context)
}

/// Evaluate the given expression string with the given mutable context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_with_context_mut<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResultValue<F> {
	let node = tree::tokens_to_operator_tree(token::tokenize(string)?)?;
	let compiled_node = compile_to_flat(node)?;

	compiled_node.eval_with_context_mut(&mut Stack::new(), context)
}

/// Parse the string and build optimized flat node with context, then evaluate it.
/// Only useful for testing, optimizing is not worth it for one off evaluations.
/// If you want performance, you should use `build_optimized_flat_node` instead!
/// Or `build_flat_node` with `optimize_flat_node` combination.
pub fn eval_optimized_with_context<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResultValue<F> {
	let optimized = build_optimized_flat_node(string, context)?;

	optimized.eval_with_context(&mut Stack::new(), context)
}
/// Build the flat node for the given string
pub fn build_flat_node<F: EvalexprFloat>(string: &str) -> EvalexprResult<FlatNode<F>, F> {
	let node = tree::tokens_to_operator_tree(token::tokenize(string)?)?;
	compile_to_flat(node)
}

/// Builds optimized flat node with inlining variabless from context
/// Takes mutable reference to context, but after returning the context will have saame values as
/// before the call
pub fn build_optimized_flat_node<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResult<FlatNode<F>, F> {
	let node = tree::tokens_to_operator_tree(token::tokenize(string)?)?;

	let flat_node = compile_to_flat(node)?;
	optimize_flat_node(&flat_node, context)
}
/// Build the operator tree for the given expression string.
pub fn build_ast<F: EvalexprFloat>(string: &str) -> EvalexprResult<Node<F>, F> {
	tree::tokens_to_operator_tree(token::tokenize(string)?)
}

/// Build the flat node for the given Ast
pub fn build_flat_node_from_ast<F: EvalexprFloat>(ast: Node<F>) -> EvalexprResult<FlatNode<F>, F> {
	compile_to_flat(ast)
}

/// Evaluate the given expression string into a float.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_float(string: &str) -> EvalexprResult<DefaultNumericTypes> {
	eval_float_with_context_mut(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}

/// Evaluate the given expression string into a boolean.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_boolean(string: &str) -> EvalexprResult<bool> {
	eval_boolean_with_context_mut(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}

/// Evaluate the given expression string into a tuple.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_tuple(string: &str) -> EvalexprResult<TupleType> {
	eval_tuple_with_context_mut(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}

/// Evaluate the given expression string into an empty value.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_empty(string: &str) -> EvalexprResult<EmptyType> {
	eval_empty_with_context_mut(string, &mut HashMapContext::<DefaultNumericTypes>::new())
}

/// Evaluate the given expression string into a float with the given context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_float_with_context<F: EvalexprFloat>(
	string: &str, context: &HashMapContext<F>,
) -> EvalexprResult<F, F> {
	match eval_with_context(string, context) {
		Ok(Value::Float(float)) => Ok(float),
		Ok(value) => Err(EvalexprError::expected_float(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into a boolean with the given context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_boolean_with_context<F: EvalexprFloat>(
	string: &str, context: &HashMapContext<F>,
) -> EvalexprResult<bool, F> {
	match eval_with_context(string, context) {
		Ok(Value::Boolean(boolean)) => Ok(boolean),
		Ok(value) => Err(EvalexprError::expected_boolean(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into a tuple with the given context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_tuple_with_context<F: EvalexprFloat>(
	string: &str, context: &HashMapContext<F>,
) -> EvalexprResult<TupleType<F>, F> {
	match eval_with_context(string, context) {
		Ok(Value::Tuple(tuple)) => Ok(tuple),
		Ok(value) => Err(EvalexprError::expected_tuple(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into an empty value with the given context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_empty_with_context<F: EvalexprFloat>(
	string: &str, context: &HashMapContext<F>,
) -> EvalexprResult<EmptyType, F> {
	match eval_with_context(string, context) {
		Ok(Value::Empty) => Ok(EMPTY_VALUE),
		Ok(value) => Err(EvalexprError::expected_empty(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into a float with the given mutable context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_float_with_context_mut<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResult<F, F> {
	match eval_with_context_mut(string, context) {
		Ok(Value::Float(float)) => Ok(float),
		Ok(value) => Err(EvalexprError::expected_float(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into a boolean with the given mutable context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_boolean_with_context_mut<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResult<bool, F> {
	match eval_with_context_mut(string, context) {
		Ok(Value::Boolean(boolean)) => Ok(boolean),
		Ok(value) => Err(EvalexprError::expected_boolean(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into a tuple with the given mutable context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_tuple_with_context_mut<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResult<TupleType<F>, F> {
	match eval_with_context_mut(string, context) {
		Ok(Value::Tuple(tuple)) => Ok(tuple),
		Ok(value) => Err(EvalexprError::expected_tuple(value)),
		Err(error) => Err(error),
	}
}

/// Evaluate the given expression string into an empty value with the given mutable context.
///
/// *See the [crate doc](index.html) for more examples and explanations of the expression format.*
pub fn eval_empty_with_context_mut<F: EvalexprFloat>(
	string: &str, context: &mut HashMapContext<F>,
) -> EvalexprResult<EmptyType, F> {
	match eval_with_context_mut(string, context) {
		Ok(Value::Empty) => Ok(EMPTY_VALUE),
		Ok(value) => Err(EvalexprError::expected_empty(value)),
		Err(error) => Err(error),
	}
}
