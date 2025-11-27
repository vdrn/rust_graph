use std::mem;

use crate::error::{EvalexprError, EvalexprResult};
use crate::token::Token;
use crate::value::numeric_types::default_numeric_types::DefaultNumericTypes;
use crate::value::numeric_types::EvalexprFloat;
use crate::value::Value;
use crate::{istr, IStr};

// Exclude display module from coverage, as it prints not well-defined prefix notation.
#[cfg(not(tarpaulin_include))]
mod display;
mod iter;

/// An enum that represents operators in the operator tree.
#[derive(Debug, PartialEq, Clone)]
pub enum Operator<NumericTypes: EvalexprFloat = DefaultNumericTypes> {
	/// A root node in the operator tree.
	/// The whole expression is stored under a root node, as well as each subexpression surrounded by
	/// parentheses.
	RootNode,

	/// A binary addition operator.
	Add,
	/// A binary subtraction operator.
	Sub,
	/// A unary negation operator.
	Neg,
	/// A binary multiplication operator.
	Mul,
	/// A binary division operator.
	Div,
	/// A binary modulo operator.
	Mod,
	/// A binary exponentiation operator.
	Exp,

	/// A binary equality comparator.
	Eq,
	/// A binary inequality comparator.
	Neq,
	/// A binary greater-than comparator.
	Gt,
	/// A binary lower-than comparator.
	Lt,
	/// A binary greater-than-or-equal comparator.
	Geq,
	/// A binary lower-than-or-equal comparator.
	Leq,
	/// A binary logical and operator.
	And,
	/// A binary logical or operator.
	Or,
	/// A binary logical not operator.
	Not,

	/// A binary assignment operator.
	Assign,
	/// A binary add-assign operator.
	AddAssign,
	/// A binary subtract-assign operator.
	SubAssign,
	/// A binary multiply-assign operator.
	MulAssign,
	/// A binary divide-assign operator.
	DivAssign,
	/// A binary modulo-assign operator.
	ModAssign,
	/// A binary exponentiate-assign operator.
	ExpAssign,
	/// A binary and-assign operator.
	AndAssign,
	/// A binary or-assign operator.
	OrAssign,

	/// An n-ary tuple constructor.
	Tuple,
	/// An n-ary subexpression chain.
	Chain,
	/// Range operator
	Range,

	/// A constant value.
	Const {
		/** The value of the constant. */
		value: Value<NumericTypes>,
	},
	/// A write to a variable identifier.
	VariableIdentifierWrite {
		/// The identifier of the variable.
		identifier: IStr,
	},
	/// A read from a variable identifier.
	VariableIdentifierRead {
		/// The identifier of the variable.
		identifier: IStr,
	},
	/// A function identifier.
	FunctionIdentifier {
		/// The identifier of the function.
		identifier: IStr,
	},
	/// A unary postfix factorial operator.
	Factorial,
	/// Dot Access
	DotAccess {
		/// The identifier of the field
		identifier: IStr,
	},
}
impl<NumericTypes: EvalexprFloat> core::fmt::Display for Operator<NumericTypes> {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
		use Operator::*;
		match self {
			RootNode => Ok(()),
			Add => write!(f, "+"),
			Sub => write!(f, "-"),
			Neg => write!(f, "-"),
			Mul => write!(f, "*"),
			Div => write!(f, "/"),
			Mod => write!(f, "%"),
			Exp => write!(f, "^"),

			Eq => write!(f, "=="),
			Neq => write!(f, "!="),
			Gt => write!(f, ">"),
			Lt => write!(f, "<"),
			Geq => write!(f, ">="),
			Leq => write!(f, "<="),
			And => write!(f, "&&"),
			Or => write!(f, "||"),
			Not => write!(f, "!"),
			Factorial => write!(f, "!"),

			Assign => write!(f, " = "),
			AddAssign => write!(f, " += "),
			SubAssign => write!(f, " -= "),
			MulAssign => write!(f, " *= "),
			DivAssign => write!(f, " /= "),
			ModAssign => write!(f, " %= "),
			ExpAssign => write!(f, " ^= "),
			AndAssign => write!(f, " &&= "),
			OrAssign => write!(f, " ||= "),

			Tuple => write!(f, ", "),
			Chain => write!(f, "; "),
			Range => write!(f, ".."),

			Const { value } => write!(f, "{}", value),
			VariableIdentifierWrite { identifier } | VariableIdentifierRead { identifier } => {
				write!(f, "{}", identifier)
			},
			FunctionIdentifier { identifier } => write!(f, "{}", identifier),
			DotAccess { identifier } => write!(f, ".{}", identifier),
		}
	}
}

impl<NumericTypes: EvalexprFloat> Operator<NumericTypes> {
	pub(crate) fn value(value: Value<NumericTypes>) -> Self { Operator::Const { value } }

	pub(crate) fn variable_identifier_write(identifier: &str) -> Self {
		Operator::VariableIdentifierWrite { identifier: istr(identifier) }
	}

	pub(crate) fn variable_identifier_read(identifier: &str) -> Self {
		Operator::VariableIdentifierRead { identifier: istr(identifier) }
	}

	pub(crate) fn function_identifier(identifier: &str) -> Self {
		Operator::FunctionIdentifier { identifier: istr(identifier) }
	}

	/// Returns the precedence of the operator.
	/// A high precedence means that the operator has priority to be deeper in the tree.
	pub(crate) const fn precedence(&self) -> i32 {
		use Operator::*;
		match self {
			RootNode => 200,

			Add | Sub => 95,
			Neg => 110,
			Mul | Div | Mod => 100,
			Exp => 120,
			Factorial => 125,

			Eq | Neq | Gt | Lt | Geq | Leq => 80,
			And => 75,
			Or => 70,
			Not => 110,

			Assign | AddAssign | SubAssign | MulAssign | DivAssign | ModAssign | ExpAssign | AndAssign
			| OrAssign => 50,

			Tuple => 40,
			Range => 20,
			Chain => 0,

			Const { .. } => 200,
			VariableIdentifierWrite { .. } | VariableIdentifierRead { .. } => 200,
			FunctionIdentifier { .. } => 190,
			DotAccess { .. } => 220,
		}
	}

	/// Returns true if chains of operators with the same precedence as this one should be evaluated
	/// left-to-right, and false if they should be evaluated right-to-left.
	/// Left-to-right chaining has priority if operators with different order but same precedence are chained.
	pub(crate) const fn is_left_to_right(&self) -> bool {
		!matches!(self, Operator::Assign | Operator::FunctionIdentifier { .. })
	}

	/// Returns true if chains of this operator should be flattened into one operator with many arguments.
	pub(crate) const fn is_sequence(&self) -> bool { matches!(self, Operator::Tuple | Operator::Chain) }

	/// True if this operator is a leaf, meaning it accepts no arguments.
	// Make this a const fn as soon as whatever is missing gets stable (issue #57563)
	pub(crate) fn is_leaf(&self) -> bool { self.max_argument_amount() == Some(0) }

	pub(crate) fn is_postfix(&self) -> bool {
		matches!(self, Operator::Factorial | Operator::DotAccess { .. })
	}

	/// Returns the maximum amount of arguments required by this operator.
	pub(crate) const fn max_argument_amount(&self) -> Option<usize> {
		use Operator::*;
		match self {
			Add | Sub | Mul | Div | Mod | Exp | Eq | Neq | Gt | Lt | Geq | Leq | And | Or | Assign
			| AddAssign | SubAssign | MulAssign | DivAssign | ModAssign | ExpAssign | AndAssign | OrAssign
			| Range => Some(2),
			Tuple | Chain => None,
			Not | Neg | Factorial | RootNode | DotAccess { .. } => Some(1),
			Const { .. } => Some(0),
			VariableIdentifierWrite { .. } | VariableIdentifierRead { .. } => Some(0),
			FunctionIdentifier { .. } => Some(99),
		}
	}

	/// Returns true if this operator is unary, i.e. it requires exactly one argument.
	pub(crate) fn is_unary(&self) -> bool {
		self.max_argument_amount() == Some(1) && *self != Operator::RootNode
	}
}

/// A node in the operator tree.
/// The operator tree is created by the crate-level `build_operator_tree` method.
/// It can be evaluated for a given context with the `Node::eval` method.
///
/// The advantage of constructing the operator tree separately from the actual evaluation is that it can be
/// evaluated arbitrarily often with different contexts.
///
/// # Examples
///
/// ```rust
/// use evalexpr::*;
///
/// let mut context = HashMapContext::<DefaultNumericTypes>::new();
/// context.set_value("alpha".into(), Value::from_float(2.0)).unwrap(); // Do proper error handling here
/// let node = build_operator_tree("1 + alpha").unwrap(); // Do proper error handling here
/// assert_eq!(node.eval_with_context(&context), Ok(Value::from_float(3.0)));
/// ```
#[derive(Debug, PartialEq, Clone)]
pub struct Node<NumericTypes: EvalexprFloat = DefaultNumericTypes> {
	pub(crate) operator: Operator<NumericTypes>,
	pub(crate) children: Vec<Node<NumericTypes>>,
}

impl<NumericTypes: EvalexprFloat> Node<NumericTypes> {
	fn new(operator: Operator<NumericTypes>) -> Self { Self { children: Vec::new(), operator } }

	fn root_node() -> Self { Self::new(Operator::RootNode) }

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
		self.iter().filter_map(|node| match node.operator() {
			Operator::VariableIdentifierWrite { identifier }
			| Operator::VariableIdentifierRead { identifier }
			| Operator::FunctionIdentifier { identifier } => Some(identifier.to_str()),
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
		self.iter().filter_map(|node| match node.operator() {
			Operator::VariableIdentifierWrite { identifier }
			| Operator::VariableIdentifierRead { identifier } => Some(identifier.to_str()),
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
		self.iter().filter_map(|node| match node.operator() {
			Operator::VariableIdentifierRead { identifier } => Some(identifier.to_str()),
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
		self.iter().filter_map(|node| match node.operator() {
			Operator::VariableIdentifierWrite { identifier } => Some(identifier.to_str()),
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
		self.iter().filter_map(|node| match node.operator() {
			Operator::FunctionIdentifier { identifier } => Some(identifier.to_str()),
			_ => None,
		})
	}

	/// Returns the children of this node as a slice.
	pub fn children(&self) -> &[Node<NumericTypes>] { &self.children }

	/// Returns the operator associated with this node.
	pub fn operator(&self) -> &Operator<NumericTypes> { &self.operator }

	/// Returns a mutable reference to the vector containing the children of this node.
	///
	/// WARNING: Writing to this might have unexpected results, as some operators require certain amounts and
	/// types of arguments.
	pub fn children_mut(&mut self) -> &mut Vec<Node<NumericTypes>> { &mut self.children }

	/// Returns a mutable reference to the operator associated with this node.
	///
	/// WARNING: Writing to this might have unexpected results, as some operators require different amounts
	/// and types of arguments.
	pub fn operator_mut(&mut self) -> &mut Operator<NumericTypes> { &mut self.operator }

	fn has_enough_children(&self) -> bool {
		Some(self.children().len()) == self.operator().max_argument_amount()
	}

	fn has_too_many_children(&self) -> bool {
		if let Some(max_argument_amount) = self.operator().max_argument_amount() {
			self.children().len() > max_argument_amount
		} else {
			false
		}
	}
}
impl<NumericTypes: EvalexprFloat> Node<NumericTypes> {
	fn insert_back_prioritized(
		&mut self, mut node: Node<NumericTypes>, is_root_node: bool,
	) -> EvalexprResult<(), NumericTypes> {
		// println!(
		//     "Inserting {:?} into {:?}, is_root_node = {is_root_node}",
		//     node, self,
		// );
		if matches!(self.operator(), Operator::FunctionIdentifier { .. })
			&& node.operator() == &Operator::RootNode
		{
			if let Some(child) = node.children.pop() {
				if child.operator() == &Operator::Tuple {
					// println!("pushing tuple children into function");
					for mut child in child.children {
						if child.operator() == &Operator::RootNode && !child.children.is_empty() {
							self.children.append(&mut child.children);
						} else {
							self.children.push(child);
						}
					}
					return Ok(());
				} else {
					self.children.push(child);
					return Ok(());
				}
			}
		}
		// println!("Self is {:?}", self);
		if self.operator().precedence() < node.operator().precedence() || node.operator().is_unary() || is_root_node
            // Right-to-left chaining
            || (self.operator().precedence() == node.operator().precedence() && !self.operator().is_left_to_right() && !node.operator().is_left_to_right())
		{
			if self.operator().is_leaf() {
				// println!("insertback  appeand to leaf 0");
				Err(EvalexprError::AppendedToLeafNode)
			} else if self.has_enough_children() {
				// Unwrap cannot fail because is_leaf being false and has_enough_children being true implies
				// that the operator wants and has at least one child
				let last_child_operator = self.children.last().unwrap().operator();

				if last_child_operator.precedence()
                    < node.operator().precedence() || node.operator().is_unary()
                    // Right-to-left chaining
                    || (last_child_operator.precedence()
                    == node.operator().precedence() && !last_child_operator.is_left_to_right() && !node.operator().is_left_to_right())
				{
					// println!(
					//     "Recursing into {:?}",
					//     self.children.last().unwrap().operator()
					// );
					// Unwrap cannot fail because is_leaf being false and has_enough_children being true
					// implies that the operator wants and has at least one child
					self.children.last_mut().unwrap().insert_back_prioritized(node, false)
				} else {
					// println!("Rotating");
					if node.operator().is_leaf() {
						// println!("insertback  appeand to leaf 1");

						return Err(EvalexprError::AppendedToLeafNode);
					}

					// Unwrap cannot fail because is_leaf being false and has_enough_children being true
					// implies that the operator wants and has at least one child
					let last_child = self.children.pop().unwrap();
					// Root nodes have at most one child
					// TODO I am not sure if this is the correct error
					if self.operator() == &Operator::RootNode && !self.children().is_empty() {
						return Err(EvalexprError::MissingOperatorOutsideOfBrace);
					}
					// Do not insert root nodes into root nodes.
					// TODO I am not sure if this is the correct error
					if self.operator() == &Operator::RootNode && node.operator() == &Operator::RootNode {
						return Err(EvalexprError::MissingOperatorOutsideOfBrace);
					}
					self.children.push(node);
					let node = self.children.last_mut().unwrap();

					// Root nodes have at most one child
					// TODO I am not sure if this is the correct error
					if node.operator() == &Operator::RootNode && !node.children().is_empty() {
						return Err(EvalexprError::MissingOperatorOutsideOfBrace);
					}
					// Do not insert root nodes into root nodes.
					// TODO I am not sure if this is the correct error
					if node.operator() == &Operator::RootNode && last_child.operator() == &Operator::RootNode {
						return Err(EvalexprError::MissingOperatorOutsideOfBrace);
					}
					if last_child.operator() == &Operator::RootNode {
						node.children.extend(last_child.children);
					} else {
						node.children.push(last_child);
					}
					Ok(())
				}
			} else {
				// println!("Inserting as specified");
				// if node.operator() == &Operator::RootNode {
				//     self.children.extend(node.children);
				// } else {
				self.children.push(node);
				// }
				Ok(())
			}
		} else {
			Err(EvalexprError::PrecedenceViolation)
		}
	}
}

fn collapse_root_stack_to<NumericTypes: EvalexprFloat>(
	root_stack: &mut Vec<Node<NumericTypes>>, mut root: Node<NumericTypes>, collapse_goal: &Node<NumericTypes>,
) -> EvalexprResult<Node<NumericTypes>, NumericTypes> {
	loop {
		if let Some(mut potential_higher_root) = root_stack.pop() {
			// TODO I'm not sure about this >, as I have no example for different sequence operators with the
			// same precedence
			if potential_higher_root.operator().precedence() > collapse_goal.operator().precedence() {
				potential_higher_root.children.push(root);
				root = potential_higher_root;
			} else {
				root_stack.push(potential_higher_root);
				break;
			}
		} else {
			// This is the only way the topmost root node could have been removed
			return Err(EvalexprError::UnmatchedRBrace);
		}
	}

	Ok(root)
}

fn collapse_all_sequences<NumericTypes: EvalexprFloat>(
	root_stack: &mut Vec<Node<NumericTypes>>,
) -> EvalexprResult<(), NumericTypes> {
	// println!("Collapsing all sequences");
	// println!("Initial root stack is: {:?}", root_stack);
	let mut root = if let Some(root) = root_stack.pop() {
		root
	} else {
		return Err(EvalexprError::UnmatchedRBrace);
	};

	loop {
		// println!("Root is: {:?}", root);
		if root.operator() == &Operator::RootNode {
			// This should fire if parsing something like `4(5)`
			if root.has_too_many_children() {
				return Err(EvalexprError::MissingOperatorOutsideOfBrace);
			}

			root_stack.push(root);
			break;
		}

		if let Some(mut potential_higher_root) = root_stack.pop() {
			if root.operator().is_sequence() {
				potential_higher_root.children.push(root);
				root = potential_higher_root;
			} else {
				// This should fire if parsing something like `4(5)`
				if root.has_too_many_children() {
					return Err(EvalexprError::MissingOperatorOutsideOfBrace);
				}

				root_stack.push(potential_higher_root);
				root_stack.push(root);
				break;
			}
		} else {
			// This is the only way the topmost root node could have been removed
			return Err(EvalexprError::UnmatchedRBrace);
		}
	}

	// println!("Root stack after collapsing all sequences is: {:?}", root_stack);
	Ok(())
}

// fn insert_postfix_operator<NumericTypes: EvalexprFloat>(
// 	root: &mut Node<NumericTypes>, mut postfix_op: Node<NumericTypes>, after_rbrace: bool,
// ) -> EvalexprResult<(), NumericTypes> {
// 	// println!(
// 	//     "Inserting postfix root {:?} postfix op {:?}, after_rbrace = {after_rbrace}",
// 	// root, postfix_op,
// 	// );
// 	// find rightmost position based on precedence
// 	if root.operator() == &Operator::RootNode {
// 		if let Some(child) = root.children.last_mut() {
// 			insert_postfix_operator(child, postfix_op, after_rbrace)
// 		} else {
// 			// println!("postfix appeand to leaf 0");
// 			Err(EvalexprError::AppendedToLeafNode)
// 		}
// 	} else if root.operator().is_sequence() && after_rbrace {
// 		// if we just closed a brace and we're at a sequence operator,
// 		// wrap the entire sequence
// 		let old_root = std::mem::replace(root, postfix_op);
// 		root.children.push(old_root);
// 		Ok(())
// 	} else if root.operator().is_leaf() {
// 		// wrap the leaf
// 		let old_root = std::mem::replace(root, postfix_op);
// 		root.children.push(old_root);
// 		Ok(())
// 	} else if root.operator().precedence() < postfix_op.operator().precedence() {
// 		// cur oper has lower precedence, so bind to the right operand
// 		if let Some(last_child) = root.children.last_mut() {
// 			// if last child is a RootNode and we just had rbrace, wrap it entirely
// 			if after_rbrace && last_child.operator() == &Operator::RootNode {
// 				let nested_root = root.children.pop().unwrap();
// 				postfix_op.children.push(nested_root);
// 				root.children.push(postfix_op);
// 				Ok(())
// 			} else {
// 				insert_postfix_operator(last_child, postfix_op, false)
// 			}
// 		} else {
// 			// println!("postfix appeand to leaf 1");
// 			Err(EvalexprError::AppendedToLeafNode)
// 		}
// 	} else {
// 		// current op has higher or equal precedence, wrap it
// 		let old_root = std::mem::replace(root, postfix_op);
// 		root.children.push(old_root);
// 		Ok(())
// 	}
// }
fn insert_postfix_operator<NumericTypes: EvalexprFloat>(
	root: &mut Node<NumericTypes>, mut postfix_op: Node<NumericTypes>, after_rbrace: bool,
) -> EvalexprResult<(), NumericTypes> {
	// find rightmost position based on precedence
	if root.operator() == &Operator::RootNode {
		if let Some(child) = root.children.last_mut() {
			insert_postfix_operator(child, postfix_op, after_rbrace)
		} else {
			Err(EvalexprError::AppendedToLeafNode)
		}
	} else if after_rbrace {
		// If we just closed a brace, wrap the entire current subtree
		// This handles cases like ((1,2) * (3,4)).x
		let old_root = std::mem::replace(root, postfix_op);
		root.children.push(old_root);
		Ok(())
	} else if root.operator().is_leaf() {
		// wrap the leaf
		let old_root = std::mem::replace(root, postfix_op);
		root.children.push(old_root);
		Ok(())
	} else if root.operator().precedence() < postfix_op.operator().precedence() {
		// cur oper has lower precedence, so bind to the right operand
		if let Some(last_child) = root.children.last_mut() {
			insert_postfix_operator(last_child, postfix_op, false)
		} else {
			Err(EvalexprError::AppendedToLeafNode)
		}
	} else {
		// current op has higher or equal precedence, wrap it
		let old_root = std::mem::replace(root, postfix_op);
		root.children.push(old_root);
		Ok(())
	}
}
pub(crate) fn tokens_to_operator_tree<NumericTypes: EvalexprFloat>(
	tokens: Vec<Token<NumericTypes>>,
) -> EvalexprResult<Node<NumericTypes>, NumericTypes> {
	let mut root_stack = vec![Node::root_node()];
	let mut last_token_is_rightsided_value = false;
	let mut last_token_is_rbrace = false;
	let mut token_iter = tokens.iter().peekable();
	// println!("TOKENS TO OPERATOR TREE: {tokens:?}");

	while let Some(token) = token_iter.next().cloned() {
		let next = token_iter.peek().cloned();

		let node = match token.clone() {
			Token::DotAccess(identifier) => {
				Some(Node::new(Operator::DotAccess { identifier: istr(&identifier) }))
			},
			Token::Plus => Some(Node::new(Operator::Add)),
			Token::Minus => {
				if last_token_is_rightsided_value {
					Some(Node::new(Operator::Sub))
				} else {
					Some(Node::new(Operator::Neg))
				}
			},
			Token::Star => Some(Node::new(Operator::Mul)),
			Token::Slash => Some(Node::new(Operator::Div)),
			Token::Percent => Some(Node::new(Operator::Mod)),
			Token::Hat => Some(Node::new(Operator::Exp)),

			Token::Eq => Some(Node::new(Operator::Eq)),
			Token::Neq => Some(Node::new(Operator::Neq)),
			Token::Gt => Some(Node::new(Operator::Gt)),
			Token::Lt => Some(Node::new(Operator::Lt)),
			Token::Geq => Some(Node::new(Operator::Geq)),
			Token::Leq => Some(Node::new(Operator::Leq)),
			Token::And => Some(Node::new(Operator::And)),
			Token::Or => Some(Node::new(Operator::Or)),

			Token::Not => {
				if last_token_is_rightsided_value {
					Some(Node::new(Operator::Factorial))
				} else {
					Some(Node::new(Operator::Not))
				}
			},
			Token::LBrace => {
				root_stack.push(Node::root_node());
				None
			},
			Token::RBrace => {
				if root_stack.len() <= 1 {
					return Err(EvalexprError::UnmatchedRBrace);
				} else {
					collapse_all_sequences(&mut root_stack)?;
					// last_token_is_rbrace = true;
					root_stack.pop()
				}
			},

			Token::Assign => Some(Node::new(Operator::Assign)),
			Token::PlusAssign => Some(Node::new(Operator::AddAssign)),
			Token::MinusAssign => Some(Node::new(Operator::SubAssign)),
			Token::StarAssign => Some(Node::new(Operator::MulAssign)),
			Token::SlashAssign => Some(Node::new(Operator::DivAssign)),
			Token::PercentAssign => Some(Node::new(Operator::ModAssign)),
			Token::HatAssign => Some(Node::new(Operator::ExpAssign)),
			Token::AndAssign => Some(Node::new(Operator::AndAssign)),
			Token::OrAssign => Some(Node::new(Operator::OrAssign)),

			Token::Comma => Some(Node::new(Operator::Tuple)),
			Token::Semicolon => Some(Node::new(Operator::Chain)),
			Token::Range => Some(Node::new(Operator::Range)),

			Token::Identifier(identifier) => {
				let mut result = Some(Node::new(Operator::variable_identifier_read(&identifier)));
				if let Some(next) = next {
					if next.is_assignment() {
						result = Some(Node::new(Operator::variable_identifier_write(&identifier)));
					} else if next.is_leftsided_value() {
						result = Some(Node::new(Operator::function_identifier(&identifier)));
					}
				}
				result
			},
			Token::Float(float) => Some(Node::new(Operator::value(Value::Float(float)))),
			Token::Int(int) => Some(Node::new(Operator::value(Value::Float(int)))),
			Token::Boolean(boolean) => Some(Node::new(Operator::value(Value::Boolean(boolean)))),
			Token::String(_) => {
				return Err(EvalexprError::CustomMessage("Strings are not supported yet".to_string()));
				// Some(Node::new(Operator::value(Value::String(string)))),
			},
		};

		if let Some(mut node) = node {

			// Need to pop and then repush here, because Rust 1.33.0 cannot release the mutable borrow of
			// root_stack before the end of this complete if-statement
			if let Some(mut root) = root_stack.pop() {
				if node.operator().is_sequence() {
					// println!("Found a sequence operator");
					// println!("Stack before sequence operation: {:?}, {:?}", root_stack, root);
					// If root.operator() and node.operator() are of the same variant, ...
					if mem::discriminant(root.operator()) == mem::discriminant(node.operator()) {
						// ... we create a new root node for the next expression in the sequence
						root.children.push(Node::root_node());
						root_stack.push(root);
					} else if root.operator() == &Operator::RootNode {
						// If the current root is an actual root node, we start a new sequence
						node.children.push(root);
						node.children.push(Node::root_node());
						root_stack.push(Node::root_node());
						root_stack.push(node);
					} else {
						// Otherwise, we combine the sequences based on their precedences
						// TODO I'm not sure about this <, as I have no example for different sequence
						// operators with the same precedence
						if root.operator().precedence() < node.operator().precedence() {
							// If the new sequence has a higher precedence, it is part of the last element of
							// the current root sequence
							if let Some(last_root_child) = root.children.pop() {
								node.children.push(last_root_child);
								node.children.push(Node::root_node());
								root_stack.push(root);
								root_stack.push(node);
							} else {
								// Once a sequence has been pushed on top of the stack, it also gets a child
								unreachable!()
							}
						} else {
							// If the new sequence doesn't have a higher precedence, then all sequences with a
							// higher precedence are collapsed below this one
							root = collapse_root_stack_to(&mut root_stack, root, &node)?;
							node.children.push(root);
							root_stack.push(node);
						}
					}
					// println!("Stack after sequence operation: {:?}", root_stack);
				} else if node.operator().is_postfix() {
					if root.operator().is_sequence() {
						if let Some(mut last_root_child) = root.children.pop() {
							insert_postfix_operator(&mut last_root_child, node, last_token_is_rbrace)?;
							root.children.push(last_root_child);
							root_stack.push(root);
						} else {
							unreachable!()
						}
					} else {
						insert_postfix_operator(&mut root, node, last_token_is_rbrace)?;
						root_stack.push(root);
					}
				} else if root.operator().is_sequence() {
					if let Some(mut last_root_child) = root.children.pop() {
						last_root_child.insert_back_prioritized(node, true)?;
						root.children.push(last_root_child);
						root_stack.push(root);
					} else {
						// Once a sequence has been pushed on top of the stack, it also gets a child
						unreachable!()
					}
				} else {
					root.insert_back_prioritized(node, true)?;
					root_stack.push(root);
				}
			} else {
				return Err(EvalexprError::UnmatchedRBrace);
			}
		}

		last_token_is_rightsided_value = token.is_rightsided_value();
		last_token_is_rbrace = matches!(token, Token::RBrace);
	
	}

	// In the end, all sequences are implicitly terminated
	collapse_all_sequences(&mut root_stack)?;

	if root_stack.len() > 1 {
		Err(EvalexprError::UnmatchedLBrace)
	} else if let Some(root) = root_stack.pop() {
		// println!("ROOT: {root:#?}");
		Ok(root)
	} else {
		Err(EvalexprError::UnmatchedRBrace)
	}
}
