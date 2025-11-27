use core::ops::RangeBounds;

use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::error::EvalexprResultValue;
use crate::flat_node::{cold, FlatOperator, IntegralNode};
use crate::function::rust_function::builtin_function;
use crate::math::integrate;
use crate::{EvalexprError, EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, IStr, Value};

#[inline(always)]
pub fn eval_flat_node<F: EvalexprFloat>(
	node: &FlatNode<F>, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, F)],
) -> EvalexprResultValue<F> {
	let stack_size = stack.len();
	let prev_num_args = stack.num_args;
	match eval_priv_inner(node, stack, context, override_vars) {
		Ok(value) => {
			debug_assert_eq!(stack_size, stack.len());
			stack.num_args = prev_num_args;
			Ok(value)
		},
		Err(err) => {
			stack.truncate(stack_size);
			stack.num_args = prev_num_args;
			Err(err)
		},
	}
}
/// Evaluates the operator with the given arguments and mutable context.
pub fn eval_flat_node_mut<F: EvalexprFloat>(
	_node: &FlatNode<F>, _stack: &mut Stack<F>, _context: &mut HashMapContext<F>,
) -> EvalexprResultValue<F> {
	todo!()
	// use crate::operator::Operator::*;
	// match self {
	//     Assign => {
	//         expect_operator_argument_amount(arguments.len(), 2)?;
	//         let target = arguments[0].as_str()?;
	//         context.set_value(target, arguments[1].clone())?;

	//         Ok(Value::Empty)
	//     },
	//     AddAssign | SubAssign | MulAssign | DivAssign | ModAssign | ExpAssign | AndAssign
	//     | OrAssign => {
	//         expect_operator_argument_amount(arguments.len(), 2)?;

	//         let target = arguments[0].as_str()?;
	//         let left_value = Operator::VariableIdentifierRead {
	//             identifier: target.to_string(),
	//         }
	//         .eval(&Vec::new(), context)?;
	//         let arguments = vec![left_value, arguments[1].clone()];

	//         let result = match self {
	//             AddAssign => Operator::Add.eval(&arguments, context),
	//             SubAssign => Operator::Sub.eval(&arguments, context),
	//             MulAssign => Operator::Mul.eval(&arguments, context),
	//             DivAssign => Operator::Div.eval(&arguments, context),
	//             ModAssign => Operator::Mod.eval(&arguments, context),
	//             ExpAssign => Operator::Exp.eval(&arguments, context),
	//             AndAssign => Operator::And.eval(&arguments, context),
	//             OrAssign => Operator::Or.eval(&arguments, context),
	//             _ => unreachable!(
	//                 "Forgot to add a match arm for an assign operation: {}",
	//                 self
	//             ),
	//         }?;
	//         context.set_value(target, result)?;

	//         Ok(Value::Empty)
	//     },
	//     _ => self.eval(arguments, context),
	// }
}

/// Stack type
#[derive(Default)]
pub struct Stack<T: EvalexprFloat, const MAX_FUNCTION_NESTING: usize = 512> {
	stack:               Vec<Value<T>>,
	function_nesting:    usize,
	pub(crate) num_args: usize,
}
impl<T: EvalexprFloat, const MAX_FUNCTION_NESTING: usize> Stack<T, MAX_FUNCTION_NESTING> {
	/// Create new stack
	pub fn new() -> Self { Self { stack: Vec::new(), function_nesting: 0, num_args: 0 } }
	/// Create new stack with capacity
	pub fn with_capacity(capacity: usize) -> Self {
		Self { stack: Vec::with_capacity(capacity), function_nesting: 0, num_args: 0 }
	}
	#[inline(always)]
	pub(crate) fn push(&mut self, value: Value<T>) { self.stack.push(value); }
	/// Returns true if the stack is empty
	pub fn is_empty(&self) -> bool { self.stack.is_empty() }
	fn pop(&mut self) -> Option<Value<T>> { self.stack.pop() }
	pub(crate) fn last_mut(&mut self) -> Option<&mut Value<T>> { self.stack.last_mut() }
	pub(crate) fn function_called(&mut self) -> EvalexprResult<(), T> {
		if self.function_nesting > MAX_FUNCTION_NESTING {
			return Err(EvalexprError::StackOverflow);
		}
		self.function_nesting += 1;
		Ok(())
	}
	pub(crate) fn function_returned(&mut self) { self.function_nesting -= 1; }

	#[inline(always)]
	pub(crate) fn push_args(&mut self, arg: &[Value<T>]) {
		self.num_args = arg.len();
		// for arg in arg {
		//     self.stack.push(arg.clone());
		// }
		self.stack.extend(arg.iter().cloned());
	}
	#[inline(always)]
	pub(crate) fn pop_args(&mut self) {
		while self.num_args > 0 {
			self.pop_unchecked();
			self.num_args -= 1;
		}
	}
	/// Returns the number of arguments on the stack.
	pub fn num_args(&self) -> usize { self.num_args }
	/// Returns the number of arguments on the stack
	#[inline(always)]
	pub fn get_arg(&self, index: usize) -> Option<&Value<T>> {
		if index >= self.num_args {
			cold();
			return None;
		}
		let arg_i = self.stack.len() - self.num_args + index;
		debug_assert!(arg_i < self.stack.len());
		Some(unsafe { self.stack.get_unchecked(arg_i) })
	}
	fn get_unchecked(&self, index: usize) -> &Value<T> {
		assert!(index < self.stack.len(), "index {} out of bounds {}", index, self.stack.len());

		unsafe { self.stack.get_unchecked(index) }
	}
	// fn get(&self, index: usize) -> Option<&Value<T>> {
	//     self.stack.get(index)
	// }

	/// Returns the number of elements on the stack
	pub fn len(&self) -> usize { self.stack.len() }
	fn truncate(&mut self, len: usize) { self.stack.truncate(len); }
	fn drain(&mut self, range: impl RangeBounds<usize>) -> std::vec::Drain<'_, Value<T>> {
		self.stack.drain(range)
	}
	fn pop_unchecked(&mut self) -> Value<T> {
		assert!(!self.is_empty());
		unsafe { self.stack.pop().unwrap_unchecked() }
	}
}
// // /// helper for stack
// pub struct StackVec<T: EvalexprNumericTypes> {
//     start: *mut Value<T>,
//     end: *mut Value<T>,
//     cap_end: *mut Value<T>,
// }
// impl<T: EvalexprNumericTypes> Default for StackVec<T> {
//     fn default() -> Self {
//         Self::new()
//     }
// }
// unsafe impl<T: EvalexprNumericTypes> Send for StackVec<T> {}
// unsafe impl<T: EvalexprNumericTypes> Sync for StackVec<T> {}
// impl<T: EvalexprNumericTypes> StackVec<T> {
//     /// create new stack
//     pub fn new() -> Self {
//         let stack: Vec<Value<T>> = Vec::new();
//         Self::from_vec(stack)
//     }

//     /// create new stack with capacity
//     pub fn with_capacity(capacity: usize) -> Self {
//         let stack: Vec<Value<T>> = Vec::with_capacity(capacity);
//         Self::from_vec(stack)
//     }
//     fn from_vec(vec: Vec<Value<T>>) -> Self {
//         let (start, len, cap) = vec.into_raw_parts();
//         let end = unsafe { start.add(len) };
//         let cap_end = unsafe { start.add(cap) };
//         Self {
//             start,
//             end,
//             cap_end,
//         }
//     }
//     /// get the length of the stack
//     pub fn len(&self) -> usize {
//         unsafe { self.end.offset_from(self.start) as usize }
//     }
//     fn len_and_cap(&self) -> (usize, usize) {
//         let len = self.end as usize - self.start as usize;
//         let cap = self.cap_end as usize - self.start as usize;
//         (len, cap)
//     }
//     unsafe fn get_unchecked(&self, index: usize) -> &Value<T> {
//         debug_assert!(index < self.len());
//         let ptr = unsafe { self.start.add(index) };
//         &*ptr
//     }

//     fn push(&mut self, value: Value<T>) {
//         unsafe {
//             if self.end == self.cap_end {
//                 cold();
//                 let (len, cap) = self.len_and_cap();
//                 let mut vec = Vec::from_raw_parts(self.start, len, cap);
//                 vec.push(value);
//                 *self = Self::from_vec(vec);
//             } else {
//                 self.end.write(value);
//                 self.end = self.end.add(1);
//             }
//         }
//     }
//     /// Returns true if the stack is empty
//     pub fn is_empty(&self) -> bool {
//         self.start == self.end
//     }
//     fn pop(&mut self) -> Option<Value<T>> {
//         unsafe {
//             if self.is_empty() {
//                 cold();
//                 None
//             } else {
//                 self.end = self.end.sub(1);
//                 let value = self.end.read();
//                 Some(value)
//             }
//         }
//     }
// }

#[inline(always)]
fn eval_priv_inner<F: EvalexprFloat>(
	node: &FlatNode<F>, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, F)],
) -> EvalexprResultValue<F> {
	let base_index = stack.len();
	for op in &node.ops {
		match op {
			// Binary arithmetic operators
			FlatOperator::Add => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a + b));
			},
			FlatOperator::Sub => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a - b));
			},
			FlatOperator::Mul => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a * b));
			},
			FlatOperator::Div => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a / b));
			},
			FlatOperator::Mod => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a % b));
			},
			FlatOperator::Exp => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a.pow(&b)));
			},
			FlatOperator::Neg => {
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(-a));
			},

			// Comparison operators
			FlatOperator::Eq => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(Value::Boolean(a == b));
			},
			FlatOperator::Neq => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(Value::Boolean(a != b));
			},
			FlatOperator::Gt => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Boolean(a > b));
			},
			FlatOperator::Lt => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Boolean(a < b));
			},
			FlatOperator::Geq => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Boolean(a >= b));
			},
			FlatOperator::Leq => {
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Boolean(a <= b));
			},

			// Logical operators
			FlatOperator::And => {
				let b = stack.pop_unchecked().as_boolean()?;
				let a = stack.pop_unchecked().as_boolean()?;
				stack.push(Value::Boolean(a && b));
			},
			FlatOperator::Or => {
				let b = stack.pop_unchecked().as_boolean()?;
				let a = stack.pop_unchecked().as_boolean()?;
				stack.push(Value::Boolean(a || b));
			},
			FlatOperator::Not => {
				let a = stack.pop_unchecked().as_boolean()?;
				stack.push(Value::Boolean(!a));
			},

			// Assignment operators
			FlatOperator::Assign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::AddAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::SubAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::MulAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::DivAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::ModAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::ExpAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::AndAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},
			FlatOperator::OrAssign => {
				return Err(EvalexprError::ContextNotMutable);
			},

			// Variable-length operators
			FlatOperator::Tuple { len } => {
				// Special case: 2-element tuple with floats becomes Float2
				if *len == 2 {
					let b = stack.pop_unchecked();
					let a = stack.pop_unchecked();

					if a.is_float() && b.is_float() {
						stack.push(Value::Float2(a.as_float()?, b.as_float()?));
					} else {
						use thin_vec::thin_vec;
						stack.push(Value::Tuple(thin_vec![a, b]));
					}
				} else {
					// todo!()
					let start_idx = stack.len() - *len as usize;
					let values = stack.drain(start_idx..).collect();
					stack.push(Value::Tuple(values));
				}
			},
			FlatOperator::Chain { len } => {
				debug_assert!(*len > 0, "Chain with 0 length should be caught at compile time");

				todo!()
				// // Keep only the last value, discard the rest
				// let start_idx = stack.len() - *len as usize;
				// stack.drain(start_idx..stack.len() - 1);
			},
			FlatOperator::FunctionCall { identifier, arg_num } => {
				let prev_num_args = stack.num_args;
				stack.num_args = *arg_num as usize;
				let result = if let Some(expr_function) = context.expr_functions.get(identifier) {
					expr_function.unchecked_call(stack, context)?
				} else if let Some(function) = context.functions.get(identifier) {
					function.unchecked_call(stack, context)?
				} else if let Some(builtin_function) = builtin_function(identifier) {
					builtin_function.unchecked_call(stack, context)?
				} else {
					return Err(EvalexprError::FunctionIdentifierNotFound(identifier.to_string()));
				};

				stack.pop_args();
				stack.num_args = prev_num_args;
				stack.push(result);
			},

			// Constants and variables
			FlatOperator::PushConst { value } => {
				stack.push(value.clone());
			},
			FlatOperator::ReadVar { identifier } => {
				let value = read_var(identifier, stack, context, override_vars)?;
				stack.push(value);
			},
			FlatOperator::WriteVar { .. } => {
				// Not implemented in immutable context
				return Err(EvalexprError::ContextNotMutable);
			},

			// N-ary operations
			FlatOperator::AddN { n } => {
				let n = *n as usize;
				let mut sum = stack.pop_unchecked().as_float()?;
				for _ in 0..n - 1 {
					sum = sum + stack.pop_unchecked().as_float()?;
				}
				stack.push(Value::Float(sum));
			},
			FlatOperator::SubN { n } => {
				let n = *n as usize;
				let mut diff = stack.pop_unchecked().as_float()?;
				for _ in 0..n - 1 {
					diff = diff - stack.pop_unchecked().as_float()?;
				}
				stack.push(Value::Float(diff));
			},
			FlatOperator::MulN { n } => {
				let n = *n as usize;
				let mut product = stack.pop_unchecked().as_float()?;
				for _ in 0..n - 1 {
					product = product * stack.pop_unchecked().as_float()?;
				}
				stack.push(Value::Float(product));
			},
			FlatOperator::DivN { n } => {
				let n = *n as usize;
				let mut res = stack.pop_unchecked().as_float()?;
				for _ in 0..n - 1 {
					res = res / stack.pop_unchecked().as_float()?;
				}
				stack.push(Value::Float(res));
			},

			// Fused operations
			FlatOperator::MulAdd => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a * b + c));
			},
			FlatOperator::MulSub => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a * b - c));
			},
			FlatOperator::DivAdd => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a / b + c));
			},
			FlatOperator::DivSub => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a / b - c));
			},
			FlatOperator::AddMul => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float((a + b) * c));
			},
			FlatOperator::SubMul => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float((a - b) * c));
			},
			FlatOperator::AddDiv => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float((a + b) / c));
			},
			FlatOperator::SubDiv => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float((a - b) / c));
			},
			FlatOperator::MulDiv => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a * b / c));
			},
			FlatOperator::DivMul => {
				let c = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				let a = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a / b * c));
			},
			FlatOperator::MulConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x * *value));
			},
			FlatOperator::AddConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x + *value));
			},
			FlatOperator::DivConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x / *value));
			},
			FlatOperator::ConstDiv { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(*value / x));
			},
			FlatOperator::SubConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x - *value));
			},
			FlatOperator::ConstSub { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(*value - x));
			},
			FlatOperator::ExpConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.pow(value)));
			},
			FlatOperator::ModConst { value } => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x % *value));
			},

			// Specialized math operations
			FlatOperator::Square => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x * x));
			},
			FlatOperator::Cube => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x * x * x));
			},
			FlatOperator::Sqrt => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.sqrt()));
			},
			FlatOperator::Cbrt => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.cbrt()));
			},

			FlatOperator::Abs => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.abs()));
			},
			FlatOperator::Floor => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.floor()));
			},
			FlatOperator::Round => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.round()));
			},
			FlatOperator::Ceil => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.ceil()));
			},
			FlatOperator::Ln => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.ln()));
			},
			FlatOperator::Log => {
				let x = stack.pop_unchecked().as_float()?;
				let base = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.log(&base)));
			},
			FlatOperator::Log2 => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.log2()));
			},
			FlatOperator::Log10 => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.log10()));
			},
			FlatOperator::ExpE => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.exp()));
			},
			FlatOperator::Exp2 => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.exp2()));
			},
			FlatOperator::Cos => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.cos()));
			},
			FlatOperator::Acos => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.acos()));
			},
			FlatOperator::CosH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.cosh()));
			},
			FlatOperator::AcosH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.acosh()));
			},
			FlatOperator::Sin => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.sin()));
			},
			FlatOperator::Asin => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.asin()));
			},
			FlatOperator::SinH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.sinh()));
			},
			FlatOperator::AsinH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.asinh()));
			},
			FlatOperator::Tan => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.tan()));
			},
			FlatOperator::Atan => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.atan()));
			},
			FlatOperator::TanH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.tanh()));
			},
			FlatOperator::AtanH => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.atanh()));
			},
			FlatOperator::Atan2 => {
				let x = stack.pop_unchecked().as_float()?;
				let y = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(y.atan2(&x)));
			},
			FlatOperator::Hypot => {
				let y = stack.pop_unchecked().as_float()?;
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.hypot(&y)));
			},

			FlatOperator::Signum => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.signum()));
			},

			FlatOperator::Min => {
				let y = stack.pop_unchecked().as_float()?;
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.min(&y)));
			},
			FlatOperator::Max => {
				let y = stack.pop_unchecked().as_float()?;
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.max(&y)));
			},
			FlatOperator::Clamp => {
				let mut max = stack.pop_unchecked().as_float()?;
				let mut min = stack.pop_unchecked().as_float()?;
				let x = stack.pop_unchecked().as_float()?;
				if min > max {
					core::mem::swap(&mut min, &mut max);
				}
				stack.push(Value::Float(x.clamp(&min, &max)));
			},
			FlatOperator::Factorial => {
				let x = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(x.factorial()));
			},
			FlatOperator::Gcd => {
				let a = stack.pop_unchecked().as_float()?;
				let b = stack.pop_unchecked().as_float()?;
				stack.push(Value::Float(a.gcd(&b)));
			},

			FlatOperator::Range => {
				let end = stack.pop_unchecked();
				let start = stack.pop_unchecked();

				let result = eval_range(start, end)?;

				stack.push(Value::Tuple(result))
			},
			FlatOperator::RangeWithStep => {
				let step = stack.pop_unchecked();
				let end = stack.pop_unchecked();
				let start = stack.pop_unchecked();
				let result = eval_range_with_step(start, end, step)?;
				stack.push(Value::Tuple(result))
			},
			// Native functions
			FlatOperator::Sum { variable, expr } => {
				let tuple = stack.pop_unchecked().as_tuple()?;
				let mut result = F::ZERO;
				let mut o_override_vars: SmallVec<[(IStr, F); 5]> = smallvec::smallvec![(*variable, F::ZERO)];
				o_override_vars.extend_from_slice(override_vars);

				for value in tuple {
					o_override_vars[0].1 = value.as_float()?;
					let current = eval_priv_inner(expr, stack, context, &o_override_vars)?.as_float()?;
					result = result + current;
				}
				stack.push(Value::Float(result));
			},
			FlatOperator::Product { variable, expr } => {
				let tuple = stack.pop_unchecked().as_tuple()?;
				let mut result = F::ONE;
				let mut o_override_vars: SmallVec<[(IStr, F); 5]> = smallvec::smallvec![(*variable, F::ZERO)];
				o_override_vars.extend_from_slice(override_vars);

				for value in tuple {
					o_override_vars[0].1 = value.as_float()?;
					let current = eval_priv_inner(expr, stack, context, &o_override_vars)?.as_float()?;
					result = result * current;
				}
				stack.push(Value::Float(result));
			},
			FlatOperator::Integral(int) => {
				let upper = stack.pop_unchecked().as_float()?;
				let lower = stack.pop_unchecked().as_float()?;
				let result = match int.as_ref() {
					IntegralNode::UnpreparedExpr { .. } => {
						return Err(EvalexprError::CustomMessage(
							"Integrals outside functions are not supported.".to_string(),
						));
					},
					IntegralNode::PreparedFunc { func, additional_args, .. } => {
						for inverse_index in additional_args {
							let value = stack.get_unchecked(base_index - *inverse_index as usize);
							stack.push(value.clone());
						}
						stack.push(Value::Empty);

						let num_args = additional_args.len() + 1;

						stack.num_args = num_args;
						let result = integrate::integrate(
							lower,
							upper,
							|x| {
								*stack.last_mut().unwrap() = Value::Float(x);
								func.unchecked_call(stack, context)?.as_float()
							},
							&F::INTEGRATION_PRECISION,
						)?;
						for _ in 0..num_args {
							stack.pop();
						}

						result
					},
				};

				stack.push(Value::Float(result));
			},
			FlatOperator::ReadLocalVar { idx } => {
				let value = stack.get_unchecked(base_index + *idx as usize);
				stack.push(value.clone());
			},
			FlatOperator::ReadParam { inverse_index } => {
				let value = stack.get_unchecked(base_index - *inverse_index as usize);
				stack.push(value.clone());
			},
      FlatOperator::AccessX => {
        let value = stack.pop_unchecked().as_float2()?;
        stack.push(Value::Float(value.0));
      }
      FlatOperator::AccessY => {
        let value = stack.pop_unchecked().as_float2()?;
        stack.push(Value::Float(value.1));
      }
		}
	}

	let result = stack.pop().unwrap();
	stack.truncate(base_index);
	Ok(result)
}

#[inline(always)]
pub fn eval_range<F: EvalexprFloat>(start: Value<F>, end: Value<F>) -> EvalexprResult<ThinVec<Value<F>>, F> {
	let end = end.as_float()?;
	let (start, second) = match start {
		Value::Float(start) => (start, None),
		Value::Float2(start, second) => (start, Some(second)),
		_ => return Err(EvalexprError::expected_float(start)),
	};

	let step = if let Some(second) = second {
		second - start
	} else if start > F::ZERO && start < F::ONE {
		if start < end {
			start
		} else {
			-start
		}
	} else {
		if start < end {
			F::ONE
		} else {
			-F::ONE
		}
	};
	if start > end && step > F::ZERO {
		return Err(EvalexprError::CustomMessage(format!(
			"Invalid range params: When step {step} is positive, start: {start} must be less than end: {end}"
		)));
	} else if start < end && step < F::ZERO {
		return Err(EvalexprError::CustomMessage(format!(
			"Invalid range params: When step {step} is negative, start: {start} must be greater than end: \
			 {end}"
		)));
	};

	let estimated_len = ((end - start) / step).to_f64() as usize;
	let mut result = thin_vec::ThinVec::with_capacity(estimated_len.clamp(1, 1024));
	result.push(Value::Float(start));
	let mut next = start + step;

	if step > F::ZERO {
		while next <= end {
			result.push(Value::Float(next));
			next = next + step;
		}
	} else {
		while next >= end {
			result.push(Value::Float(next));
			next = next + step;
		}
	}
	Ok(result)
}
#[inline(always)]
pub fn eval_range_with_step<F: EvalexprFloat>(
	start: Value<F>, end: Value<F>, step: Value<F>,
) -> EvalexprResult<ThinVec<Value<F>>, F> {
	let step = step.as_float()?;
	let end = end.as_float()?;
	let start = start.as_float()?;
	if start > end && step > F::ZERO {
		return Err(EvalexprError::CustomMessage(format!(
			"Invalid range params: When step {step} is positive, start: {start} must be less than end: {end}"
		)));
	} else if start < end && step < F::ZERO {
		return Err(EvalexprError::CustomMessage(format!(
			"Invalid range params: When step {step} is negative, start: {start} must be greater than end: \
			 {end}"
		)));
	};

	let estimated_len = ((end - start) / step).to_f64() as usize;
	let mut result = thin_vec::ThinVec::with_capacity(estimated_len.clamp(1, 1024));
	result.push(Value::Float(start));
	let mut next = start + step;
	if step > F::ZERO {
		while next <= end {
			result.push(Value::Float(next));
			next = next + step;
		}
	} else {
		while next >= end {
			result.push(Value::Float(next));
			next = next + step;
		}
	}
	Ok(result)
}

#[inline(always)]
fn read_var<F: EvalexprFloat>(
	identifier: &IStr, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, F)],
) -> EvalexprResult<Value<F>, F> {
	for (var, val) in override_vars {
		if *identifier == *var {
			return Ok(Value::Float(*val));
		}
	}
	Ok(if let Some(val) = context.get_value(*identifier).cloned() {
		val
	} else {
		// Try as zero-argument function
		let prev_num_args = stack.num_args;
		stack.num_args = 0;
		let val = if let Some(expr_function) = context.expr_functions.get(identifier) {
			expr_function.unchecked_call(stack, context)?
		} else if let Some(function) = context.functions.get(identifier) {
			function.unchecked_call(stack, context)?
		} else {
			return Err(EvalexprError::VariableIdentifierNotFound(identifier.to_string()));
		};
		stack.num_args = prev_num_args;
		val
	})
}
