use core::ops::RangeBounds;

use smallvec::SmallVec;
use thin_vec::ThinVec;

use crate::error::EvalexprResultValue;
use crate::flat_node::{cold, ClosureNode, FlatOperator, MapOp};
use crate::function::rust_function::builtin_function;
use crate::math::integrate;
use crate::{EvalexprError, EvalexprFloat, EvalexprResult, FlatNode, HashMapContext, IStr, Value, ValueType};

#[inline(always)]
pub fn eval_flat_node<F: EvalexprFloat>(
	node: &FlatNode<F>, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, Value<F>)],
) -> EvalexprResultValue<F> {
	let stack_size = stack.len();
	let prev_num_args = stack.num_args;
	match eval_priv_inner(node, stack, context, override_vars, stack_size) {
		Ok(value) => {
			stack.truncate(stack_size);
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
	node: &FlatNode<F>, stack: &mut Stack<F>, context: &mut HashMapContext<F>,
) -> EvalexprResultValue<F> {
	eval_flat_node(node, stack, context, &[])
	// todo!()
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
		Some(self.stack.get(arg_i).unwrap())
		// debug_assert!(arg_i < self.stack.len());
		// Some(unsafe { self.stack.get_unchecked(arg_i) })
	}
	#[inline(always)]
	#[track_caller]
	pub(crate) fn get_unchecked(&self, index: usize) -> &Value<T> {
		// assert!(index < self.stack.len(), "index {} out of bounds {}", index, self.stack.len());
		// unsafe { self.stack.get_unchecked(index) }
		self.stack.get(index).unwrap()
	}
	#[inline(always)]
	#[track_caller]
	fn get_unchecked_mut(&mut self, index: usize) -> &mut Value<T> {
		// assert!(index < self.stack.len(), "index {} out of bounds {}", index, self.stack.len());
		// unsafe { self.stack.get_unchecked(index) }
		self.stack.get_mut(index).unwrap()
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
		// assert!(!self.is_empty());
		// unsafe { self.stack.pop().unwrap_unchecked() }
		self.stack.pop().unwrap()
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
	node: &FlatNode<F>, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, Value<F>)],
	base_index: usize,
) -> EvalexprResultValue<F> {
	for op in &node.ops {
		match op {
			// Binary arithmetic operators
			FlatOperator::Add => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(add(a, b)?);
			},
			FlatOperator::Sub => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(sub(a, b)?);
			},
			FlatOperator::Mul => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(mul(a, b)?);
			},
			FlatOperator::Div => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(div(a, b)?);
			},
			FlatOperator::Mod => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(mod_(a, b)?);
			},
			FlatOperator::Exp => {
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(exp(a, b)?);
			},
			FlatOperator::Neg => {
				let a = stack.pop_unchecked();
				stack.push(neg(a)?);
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

				// todo!()
				// Keep only the last value, discard the rest
				let start_idx = stack.len() - *len as usize;
				stack.drain(start_idx..stack.len() - 1);
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
				let mut sum = stack.pop_unchecked();
				for _ in 0..n - 1 {
					let value = stack.pop_unchecked();

					sum = add(sum, value)?;
				}
				stack.push(sum);
			},
			FlatOperator::SubN { n } => {
				let n = *n as usize;
				let mut diff = stack.pop_unchecked();
				for _ in 0..n - 1 {
					diff = sub(diff, stack.pop_unchecked())?;
				}
				stack.push(diff);
			},
			FlatOperator::MulN { n } => {
				let n = *n as usize;
				let mut product = stack.pop_unchecked();
				for _ in 0..n - 1 {
					product = mul(product, stack.pop_unchecked())?;
				}
				stack.push(product);
			},
			FlatOperator::DivN { n } => {
				let n = *n as usize;
				let mut res = stack.pop_unchecked();
				for _ in 0..n - 1 {
					res = div(res, stack.pop_unchecked())?;
				}
				stack.push(res);
			},

			// Fused operations
			FlatOperator::MulAdd => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(mul_add(a, b, c)?);
			},
			FlatOperator::MulSub => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(mul_sub(a, b, c)?);
			},
			FlatOperator::DivAdd => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(div_add(a, b, c)?);
			},
			FlatOperator::DivSub => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(div_sub(a, b, c)?);
			},
			FlatOperator::AddMul => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(add_mul(a, b, c)?);
			},
			FlatOperator::SubMul => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(sub_mul(a, b, c)?);
			},
			FlatOperator::AddDiv => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(add_div(a, b, c)?);
			},
			FlatOperator::SubDiv => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(sub_div(a, b, c)?);
			},
			FlatOperator::MulDiv => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(mul_div(a, b, c)?);
			},
			FlatOperator::DivMul => {
				let c = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				let a = stack.pop_unchecked();
				stack.push(div_mul(a, b, c)?);
			},
			FlatOperator::MulConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(mul(x, value.clone())?);
			},
			FlatOperator::AddConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(add(x, value.clone())?);
			},
			FlatOperator::DivConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(div(x, value.clone())?);
			},
			FlatOperator::ConstDiv { value } => {
				let x = stack.pop_unchecked();
				stack.push(div(value.clone(), x)?);
			},
			FlatOperator::SubConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(sub(x, value.clone())?);
			},
			FlatOperator::ConstSub { value } => {
				let x = stack.pop_unchecked();
				stack.push(sub(value.clone(), x)?);
			},
			FlatOperator::ExpConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(exp(x, value.clone())?);
			},
			FlatOperator::ModConst { value } => {
				let x = stack.pop_unchecked();
				stack.push(mod_(x, value.clone())?);
			},

			// Specialized math operations
			FlatOperator::Square => {
				let x = stack.pop_unchecked();
				stack.push(square(x)?);
			},
			FlatOperator::Cube => {
				let x = stack.pop_unchecked();
				stack.push(cube(x)?);
			},
			FlatOperator::Sqrt => {
				let x = stack.pop_unchecked();
				stack.push(sqrt(x)?);
			},
			FlatOperator::Cbrt => {
				let x = stack.pop_unchecked();
				stack.push(cbrt(x)?);
			},

			FlatOperator::Abs => {
				let x = stack.pop_unchecked();
				stack.push(abs(x)?);
			},
			FlatOperator::Floor => {
				let x = stack.pop_unchecked();
				stack.push(floor(x)?);
			},
			FlatOperator::Round => {
				let x = stack.pop_unchecked();
				stack.push(round(x)?);
			},
			FlatOperator::Ceil => {
				let x = stack.pop_unchecked();
				stack.push(ceil(x)?);
			},
			FlatOperator::Ln => {
				let x = stack.pop_unchecked();
				stack.push(ln(x)?);
			},
			FlatOperator::Log => {
				let x = stack.pop_unchecked();
				let base = stack.pop_unchecked();
				stack.push(log(x, base)?);
			},
			FlatOperator::Log2 => {
				let x = stack.pop_unchecked();
				stack.push(log2(x)?);
			},
			FlatOperator::Log10 => {
				let x = stack.pop_unchecked();
				stack.push(log10(x)?);
			},
			FlatOperator::ExpE => {
				let x = stack.pop_unchecked();
				stack.push(exp_e(x)?);
			},
			FlatOperator::Exp2 => {
				let x = stack.pop_unchecked();
				stack.push(exp_2(x)?);
			},
			FlatOperator::Cos => {
				let x = stack.pop_unchecked();
				stack.push(cos(x)?);
			},
			FlatOperator::Acos => {
				let x = stack.pop_unchecked();
				stack.push(acos(x)?);
			},
			FlatOperator::CosH => {
				let x = stack.pop_unchecked();
				stack.push(cosh(x)?);
			},
			FlatOperator::AcosH => {
				let x = stack.pop_unchecked();
				stack.push(acosh(x)?);
			},
			FlatOperator::Sin => {
				let x = stack.pop_unchecked();
				stack.push(sin(x)?);
			},
			FlatOperator::Asin => {
				let x = stack.pop_unchecked();
				stack.push(asin(x)?);
			},
			FlatOperator::SinH => {
				let x = stack.pop_unchecked();
				stack.push(sinh(x)?);
			},
			FlatOperator::AsinH => {
				let x = stack.pop_unchecked();
				stack.push(asinh(x)?);
			},
			FlatOperator::Tan => {
				let x = stack.pop_unchecked();
				stack.push(tan(x)?);
			},
			FlatOperator::Atan => {
				let x = stack.pop_unchecked();
				stack.push(atan(x)?);
			},
			FlatOperator::TanH => {
				let x = stack.pop_unchecked();
				stack.push(tanh(x)?);
			},
			FlatOperator::AtanH => {
				let x = stack.pop_unchecked();
				stack.push(atanh(x)?);
			},
			FlatOperator::Atan2 => {
				let x = stack.pop_unchecked();
				let y = stack.pop_unchecked();
				stack.push(atan2(y, x)?);
			},
			FlatOperator::Hypot => {
				let y = stack.pop_unchecked();
				let x = stack.pop_unchecked();
				stack.push(hypot(x, y)?);
			},

			FlatOperator::Signum => {
				let x = stack.pop_unchecked();
				stack.push(signum(x)?);
			},

			FlatOperator::Min => {
				let y = stack.pop_unchecked();
				let x = stack.pop_unchecked();
				stack.push(min(x, y)?);
			},
			FlatOperator::Max => {
				let y = stack.pop_unchecked();
				let x = stack.pop_unchecked();
				stack.push(max(x, y)?);
			},
			FlatOperator::Clamp => {
				let max = stack.pop_unchecked();
				let min = stack.pop_unchecked();
				let x = stack.pop_unchecked();

				stack.push(clamp(x, min, max)?);
			},
			FlatOperator::Factorial => {
				let x = stack.pop_unchecked();
				stack.push(factorial(x)?);
			},
			FlatOperator::Gcd => {
				let a = stack.pop_unchecked();
				let b = stack.pop_unchecked();
				stack.push(gcd(a, b)?);
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
			FlatOperator::Sum(closure) => match closure.as_ref() {
				ClosureNode::Unprepared {..} => {
					return Err(EvalexprError::CustomMessage(
						"Sums outside functions are not supported".to_string(),
					));
				},
				ClosureNode::Prepared { func, additional_args, .. } => {
					let tuple = stack.pop_unchecked().as_tuple()?;
					let mut result = F::ZERO;

							additional_args.push_values_to_stack(stack, base_index);
					stack.push(Value::Empty);

					let num_args = additional_args.len() + 1;
					let prev_num_args = stack.num_args;
					stack.num_args = num_args;

					for value in tuple {
						*stack.last_mut().unwrap() = value;
						let current = func.unchecked_call(stack, context)?.as_float()?;
						result = result + current;
					}
					for _ in 0..num_args {
						stack.pop();
					}
					stack.num_args = prev_num_args;
					stack.push(Value::Float(result));
				},
			},
			FlatOperator::Product(closure) => match closure.as_ref() {
				ClosureNode::Unprepared {..} => {
					return Err(EvalexprError::CustomMessage(
						"Products outside functions are not supported".to_string(),
					));
				},
				ClosureNode::Prepared { func, additional_args, .. } => {
					let tuple = stack.pop_unchecked().as_tuple()?;
					let mut result = F::ONE;

							additional_args.push_values_to_stack(stack, base_index);
					stack.push(Value::Empty);

					let num_args = additional_args.len() + 1;
					let prev_num_args = stack.num_args;
					stack.num_args = num_args;

					for value in tuple {
						*stack.last_mut().unwrap() = value;
						let current = func.unchecked_call(stack, context)?.as_float()?;
						result = result * current;
					}
					for _ in 0..num_args {
						stack.pop();
					}
					stack.num_args = prev_num_args;

					stack.push(Value::Float(result));
				},
			},
			FlatOperator::Integral(int) => {
				let upper = stack.pop_unchecked().as_float()?;
				let lower = stack.pop_unchecked().as_float()?;
				let result = match int.as_ref() {
					ClosureNode::Unprepared { .. } => {
						return Err(EvalexprError::CustomMessage(
							"Integrals outside functions are not supported.".to_string(),
						));
					},
					ClosureNode::Prepared { func, additional_args, .. } => {
							additional_args.push_values_to_stack(stack, base_index);
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
				// println!("READ LOCAL VAR base index {base_index} idx {idx} stack {:?}", stack.stack);
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
			},
			FlatOperator::AccessY => {
				let value = stack.pop_unchecked().as_float2()?;
				stack.push(Value::Float(value.1));
			},
			FlatOperator::AccessIndex { index } => {
				let value = stack.pop_unchecked();
				stack.push(access_index(value, *index)?);
			},
			FlatOperator::Access => {
				let field = stack.pop_unchecked();
				let value = stack.pop_unchecked();
				stack.push(access(value, field)?);
			},
			FlatOperator::Map(map_op) => {
				let tuple = stack.pop_unchecked();
				use thin_vec::thin_vec;
				let mut tuple = match tuple {
					Value::Tuple(tuple) => tuple,
					Value::Float2(x, y) => thin_vec![Value::Float(x), Value::Float(y)],
					_ => return Err(EvalexprError::expected_tuple(tuple)),
				};
				match map_op {
					MapOp::Closure(closure) => match closure.as_ref() {
						ClosureNode::Unprepared { .. } => {
							return Err(EvalexprError::CustomMessage(
								"Maps outside functions are not supported".to_string(),
							));
						},
						ClosureNode::Prepared { func, params, additional_args, .. } => {
							let total = tuple.len();

							additional_args.push_values_to_stack(stack, base_index);
							let num_args = additional_args.len() + params.len();
							let prev_num_args = stack.num_args;
							stack.num_args = num_args;

							// value arg
							let value_arg_idx = stack.len();
							stack.push(Value::Empty);

							let mut index_arg_idx = None;
							match params.len() {
								1 => {},
								2 => {
									index_arg_idx = Some(stack.len());
									stack.push(Value::Empty);
								},
								3 => {
									index_arg_idx = Some(stack.len());
									stack.push(Value::Empty);
									stack.push(Value::Float(F::from_usize(total)));
								},
								_ => {
									unreachable!()
								},
							}

							for (i, value) in tuple.iter_mut().enumerate() {
								*stack.get_unchecked_mut(value_arg_idx) = value.clone();
								if let Some(index_arg) = index_arg_idx {
									*stack.get_unchecked_mut(index_arg) = Value::Float(F::from_usize(i));
								}
								let mapped = func.unchecked_call(stack, context)?;
								*value = mapped;
							}
							for _ in 0..num_args {
								stack.pop();
							}
							stack.num_args = prev_num_args;

							stack.push(tuple_to_value(tuple));
						},
					},
					MapOp::Func { name } => {
						if let Some(expr_function) = context.expr_functions.get(name) {
							let mut args: SmallVec<[Value<F>; 3]> = smallvec::smallvec![Value::Float(F::ZERO)];
							let total = tuple.len();
							match expr_function.args.len() {
								1 => {},
								2 => {
									args.push(Value::Float(F::ZERO));
								},
								3 => {
									args.push(Value::Float(F::ZERO));
									args.push(Value::Float(F::from_usize(total)));
								},
								_ => {
									unreachable!()
								},
							}
							for (i, value) in tuple.iter_mut().enumerate() {
								args[0] = value.clone();
								if let Some(index_arg) = args.get_mut(1) {
									*index_arg = Value::Float(F::from_usize(i));
								}
								let mapped = expr_function.call(stack, context, &args)?;
								*value = mapped;
							}
							stack.push(tuple_to_value(tuple));
						} else {
							return Err(EvalexprError::FunctionIdentifierNotFound(name.to_string()));
						}
					},
				}
			},
			FlatOperator::If { true_expr, false_expr } => {
				let condition = stack.pop_unchecked().as_boolean()?;
				let result = if condition {
					match true_expr.as_ref() {
						ClosureNode::Unprepared { expr, .. } => {
							eval_priv_inner(expr, stack, context, override_vars, base_index)?
						},
						ClosureNode::Prepared { func, additional_args, .. } => {
							let num_args = additional_args.len();

							additional_args.push_values_to_stack(stack, base_index);

							let prev_num_args = stack.num_args;
							stack.num_args = num_args;
							let result = func.unchecked_call(stack, context)?;
							for _ in 0..num_args {
								stack.pop();
							}
							stack.num_args = prev_num_args;
							result
						},
					}
				} else {
					if let Some(false_expr) = false_expr {
						match false_expr.as_ref() {
							ClosureNode::Unprepared { expr, .. } => {
								eval_priv_inner(expr, stack, context, override_vars, base_index)?
							},
							ClosureNode::Prepared { func, additional_args, .. } => {
								let num_args = additional_args.len();
							additional_args.push_values_to_stack(stack, base_index);
								let prev_num_args = stack.num_args;
								stack.num_args = num_args;
								let result = func.unchecked_call(stack, context)?;
								for _ in 0..num_args {
									stack.pop();
								}
								stack.num_args = prev_num_args;
								result
							},
						}
					} else {
						Value::Empty
					}
				};
				stack.push(result);
			},
		}
	}

	let result = stack.pop().unwrap();
	Ok(result)
}
pub fn access<F: EvalexprFloat>(value: Value<F>, index: Value<F>) -> EvalexprResult<Value<F>, F> {
	let index = if let Value::Float(index) = index {
		index.to_usize()? as u32
	} else {
		return Err(EvalexprError::CustomMessage("Index must be a number".to_string()));
	};
	access_index(value, index)
}

pub fn access_index<F: EvalexprFloat>(value: Value<F>, index: u32) -> EvalexprResult<Value<F>, F> {
	match value {
		Value::Float2(first, second) => match index {
			0 => Ok(Value::Float(first)),
			1 => Ok(Value::Float(second)),
			_ => Err(EvalexprError::InvalidIndex { index, len: 2 }),
		},
		Value::Tuple(tuple) => tuple
			.get(index as usize)
			.cloned()
			.ok_or_else(|| EvalexprError::InvalidIndex { index, len: tuple.len() as u32 }),
		value => Err(EvalexprError::wrong_type_combination(
			"Dot Access",
			vec![(&value).into()],
			vec![ValueType::Float2, ValueType::Tuple],
		)),
	}
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
	identifier: &IStr, stack: &mut Stack<F>, context: &HashMapContext<F>, override_vars: &[(IStr, Value<F>)],
) -> EvalexprResult<Value<F>, F> {
	for (var, val) in override_vars {
		if *identifier == *var {
			return Ok(val.clone());
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

#[inline(always)]
pub fn math_binary<F: EvalexprFloat>(
	left: Value<F>, right: Value<F>, name: &'static str, op: impl Fn(F, F) -> F + Copy,
) -> EvalexprResult<Value<F>, F> {
	match (left, right) {
		(Value::Float(l), Value::Float(r)) => Ok(Value::Float(op(l, r))),

		(Value::Float2(l, r), Value::Float2(l2, r2)) => Ok(Value::Float2(op(l, l2), op(r, r2))),
		(Value::Float(l), Value::Float2(l2, r2)) => Ok(Value::Float2(op(l, l2), op(l, r2))),
		(Value::Float2(l, r), Value::Float(l2)) => Ok(Value::Float2(op(l, l2), op(r, l2))),
		(Value::Empty, _) | (_, Value::Empty) => Ok(Value::Empty),
		(left, right) => math_binary_inner(left, right, name, op),
	}
}
#[inline(never)]
fn math_binary_inner<F: EvalexprFloat>(
	left: Value<F>, right: Value<F>, name: &'static str, op: impl Fn(F, F) -> F + Copy,
) -> EvalexprResult<Value<F>, F> {
	match (left, right) {
		(Value::Tuple(l), Value::Tuple(r)) => {
			if l.len() != r.len() {
				return Err(EvalexprError::TuplesMismatchedLengths {
					left:  l.len() as u32,
					right: r.len() as u32,
				});
			}
			let mut result = l;
			for (l, r) in result.iter_mut().zip(r.into_iter()) {
				*l = math_binary(l.clone(), r, name, op)?;
			}
			Ok(Value::Tuple(result))
		},
		(Value::Float(l), Value::Tuple(r)) => {
			let mut result = r;
			for r in result.iter_mut() {
				*r = math_binary(Value::Float(l), r.clone(), name, op)?;
			}
			Ok(Value::Tuple(result))
		},
		(Value::Tuple(l), Value::Float(r)) => {
			let mut result = l;
			for l in result.iter_mut() {
				*l = math_binary(l.clone(), Value::Float(r), name, op)?;
			}
			Ok(Value::Tuple(result))
		},

		(Value::Float2(l1, l2), Value::Tuple(r2)) => {
			if r2.len() != 2 {
				return Err(EvalexprError::TuplesMismatchedLengths { left: 2, right: r2.len() as u32 });
			}
			let mut result = r2;
			result[0] = math_binary(Value::Float(l1), result[0].clone(), name, op)?;
			result[1] = math_binary(Value::Float(l2), result[1].clone(), name, op)?;
			Ok(Value::Tuple(result))
		},
		(Value::Tuple(l), Value::Float2(r1, r2)) => {
			if l.len() != 2 {
				return Err(EvalexprError::TuplesMismatchedLengths { left: 2, right: l.len() as u32 });
			}
			let mut result = l;
			result[0] = math_binary(result[0].clone(), Value::Float(r1), name, op)?;
			result[1] = math_binary(result[1].clone(), Value::Float(r2), name, op)?;
			Ok(Value::Tuple(result))
		},
		(l, r) => Err(EvalexprError::wrong_type_combination(
			name,
			vec![(&l).into(), (&r).into()],
			vec![ValueType::Float, ValueType::Float2, ValueType::Tuple],
		)),
	}
}
pub fn math_unary<F: EvalexprFloat>(
	value: Value<F>, name: &'static str, op: impl Fn(F) -> F + Copy,
) -> EvalexprResult<Value<F>, F> {
	match value {
		Value::Float(v) => Ok(Value::Float(op(v))),
		Value::Float2(v1, v2) => Ok(Value::Float2(op(v1), op(v2))),
		Value::Tuple(mut v) => {
			for v in v.iter_mut() {
				*v = math_unary(v.clone(), name, op)?;
			}
			Ok(Value::Tuple(v))
		},
		value => Err(EvalexprError::ExpectedFloatOrTuple { operator: name, actual: value.clone() }),
	}
}
pub fn add<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "+", |l, r| l + r)
}
pub fn sub<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "-", |l, r| l - r)
}
pub fn mul<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "*", |l, r| l * r)
}
pub fn div<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "/", |l, r| l / r)
}
pub fn mod_<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "%", |l, r| l % r)
}
pub fn exp<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "^", |l, r| l.pow(&r))
}
pub fn neg<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> { math_unary(value, "-", |v| -v) }

pub fn mul_add<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	add(mul(a, b)?, c)
}
pub fn mul_sub<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	sub(mul(a, b)?, c)
}
pub fn div_add<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	add(div(a, b)?, c)
}
pub fn div_sub<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	sub(div(a, b)?, c)
}
pub fn add_mul<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	mul(add(a, b)?, c)
}
pub fn sub_mul<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	mul(sub(a, b)?, c)
}
pub fn add_div<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	div(add(a, b)?, c)
}
pub fn sub_div<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	div(sub(a, b)?, c)
}
pub fn mul_div<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	div(mul(a, b)?, c)
}
pub fn div_mul<F: EvalexprFloat>(a: Value<F>, b: Value<F>, c: Value<F>) -> EvalexprResult<Value<F>, F> {
	mul(div(a, b)?, c)
}
pub fn square<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "^", |v| v * v)
}
pub fn cube<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "^", |v| v * v * v)
}
pub fn sqrt<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "sqrt", |v| v.sqrt())
}
pub fn cbrt<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "cbrt", |v| v.cbrt())
}
pub fn abs<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "abs", |v| v.abs())
}
pub fn floor<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "floor", |v| v.floor())
}
pub fn round<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "round", |v| v.round())
}
pub fn ceil<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "ceil", |v| v.ceil())
}
pub fn ln<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "ln", |v| v.ln())
}
pub fn log<F: EvalexprFloat>(value: Value<F>, base: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(value, base, "log", |l, r| l.log(&r))
}
pub fn log2<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "log2", |v| v.log2())
}
pub fn log10<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "log10", |v| v.log10())
}

pub fn exp_e<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "exp_e", |v| v.exp())
}
pub fn exp_2<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "exp_2", |v| v.exp2())
}
pub fn cos<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "cos", |v| v.cos())
}
pub fn acos<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "acos", |v| v.acos())
}
pub fn cosh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "cosh", |v| v.cosh())
}
pub fn acosh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "acosh", |v| v.acosh())
}
pub fn sin<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "sin", |v| v.sin())
}
pub fn asin<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "asin", |v| v.asin())
}
pub fn sinh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "sinh", |v| v.sinh())
}
pub fn asinh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "asinh", |v| v.asinh())
}
pub fn tan<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "tan", |v| v.tan())
}
pub fn atan<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "atan", |v| v.atan())
}
pub fn tanh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "tanh", |v| v.tanh())
}
pub fn atanh<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "atanh", |v| v.atanh())
}
pub fn atan2<F: EvalexprFloat>(y: Value<F>, x: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(y, x, "atan2", |l, r| l.atan2(&r))
}
pub fn hypot<F: EvalexprFloat>(x: Value<F>, y: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(x, y, "hypot", |l, r| l.hypot(&r))
}
pub fn signum<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "signum", |v| v.signum())
}
pub fn min<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "min", |l, r| l.min(&r))
}
pub fn max<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "max", |l, r| l.max(&r))
}
pub fn clamp<F: EvalexprFloat>(
	value: Value<F>, minv: Value<F>, maxv: Value<F>,
) -> EvalexprResult<Value<F>, F> {
	min(max(value, minv)?, maxv)
}
pub fn factorial<F: EvalexprFloat>(value: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_unary(value, "factorial", |v| v.factorial())
}
pub fn gcd<F: EvalexprFloat>(left: Value<F>, right: Value<F>) -> EvalexprResult<Value<F>, F> {
	math_binary(left, right, "gcd", |l, r| l.gcd(&r))
}

pub fn tuple_to_value<F: EvalexprFloat>(tuple: ThinVec<Value<F>>) -> Value<F> {
	match tuple.len() {
		2 => {
			if let (Ok(x), Ok(y)) = (tuple[0].as_float(), tuple[1].as_float()) {
				return Value::Float2(x, y);
			}
		},
		_ => {},
	}
	Value::Tuple(tuple)
}
