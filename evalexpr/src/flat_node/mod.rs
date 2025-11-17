use core::ops::RangeBounds;
use std::vec::Drain;

use smallvec::SmallVec;

use crate::{
    error::{expect_operator_argument_amount, EvalexprResultValue},
    flat_node::eval::eval_range_with_step,
    function::builtin::builtin_function,
    Context, ContextWithMutableVariables, EmptyType, EvalexprError, EvalexprFloat, EvalexprResult,
    HashMapContext, IStr, Node, Operator, TupleType, Value, EMPTY_VALUE,
};
pub(crate) mod eval;
pub(crate) mod inlining;
mod optimization;

use eval::eval_range;
use optimization::{
    compile_operator_function, extract_const_float, fuse_comutative, fuse_left, nary_op,
    FusingReturn,
};

#[cold]
pub fn cold() {}
// pub fn unlikely(x: bool) -> bool {
//     if x {
//         cold()
//     }
//     x
// }

#[derive(Debug, Clone, PartialEq)]
// NOTE: while repr(C) costs us 8 bytes, it generates much nicer match in `eval_priv`
#[repr(C)]
pub enum FlatOperator<F: EvalexprFloat> {
    // Arithmetic operators
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Exp,
    Neg,

    // Comparison operators
    Eq,
    Neq,
    Gt,
    Lt,
    Geq,
    Leq,

    // Logical operators
    And,
    Or,
    Not,

    // Assignment operators (may not be used in immutable context)
    Assign,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    ModAssign,
    ExpAssign,
    AndAssign,
    OrAssign,

    // Variable-length operators with explicit length
    /// Construct tuple from top `len` stack values
    Tuple {
        len: u32,
    },
    /// Execute `len` expressions, keep only the last result
    Chain {
        len: u32,
    },
    /// Call function with `len` arguments from stack
    FunctionCall {
        identifier: IStr,
        arg_num: u32,
    },

    // Constants and variables
    /// Push constant value onto stack
    PushConst {
        value: Value<F>,
    },
    /// Read variable and push onto stack
    ReadVar {
        identifier: IStr,
    },
    /// Reads a variable and pused `-value` to the stack
    ReadVarNeg {
        identifier: IStr,
    },
    /// Write to variable (pops value from stack)
    WriteVar {
        identifier: IStr,
    },

    // N-ary operations
    AddN {
        n: u32,
    },
    SubN {
        n: u32,
    },
    MulN {
        n: u32,
    },
    DivN {
        n: u32,
    },

    // Fused operations
    /// a*b + c
    MulAdd,
    /// a*b - c
    MulSub,
    /// a/b + c
    DivAdd,
    /// a/b - c
    DivSub,
    /// (a+b) * c
    AddMul,
    /// (a-b) * c
    SubMul,
    /// (a+b) / c
    AddDiv,
    /// (a-b) / c
    SubDiv,
    /// (a * b / c)
    MulDiv,
    /// (a / b * c)
    DivMul,
    /// (x * C or C * x)
    MulConst {
        value: F,
    },
    ///(x + C or C + x)
    AddConst {
        value: F,
    },
    ///(x/C)
    DivConst {
        value: F,
    },
    ///(C/x)
    ConstDiv {
        value: F,
    },
    ///(x - C)
    SubConst {
        value: F,
    },
    /// (C - x)
    ConstSub {
        value: F,
    },
    /// x^C
    ExpConst {
        value: F,
    },
    /// x%C
    ModConst {
        value: F,
    },
    // Specialized math operations
    Square,
    Cube,

    Sqrt,
    Cbrt,
    Abs,

    Floor,
    Round,
    Ceil,
    Ln,
    Log, // 2 params
    Log2,
    Log10,
    ExpE,
    Exp2,

    Cos,
    Acos,
    CosH,
    AcosH,

    Sin,
    Asin,
    SinH,
    AsinH,

    Tan,
    Atan,
    TanH, 
    AtanH,
    Atan2,//2 params
    Hypot, // 2 params
    Signum,

    Min,   // 2 params
    Max,   // 2 params
    Clamp, // 3 params

    Factorial,

    Range,
    RangeWithStep,
    /// ∑
    Sum {
        variable: IStr,
        expr: Box<FlatNode<F>>,
    },
    /// ∏
    Product {
        variable: IStr,
        expr: Box<FlatNode<F>>,
    },

    /// Duplicates the top value on the stack
    Duplicate,
}

/// Flat compiled node - linear sequence of operations
#[derive(Debug, Clone, PartialEq)]
pub struct FlatNode<F: EvalexprFloat> {
    ops: Vec<FlatOperator<F>>,
}

/// Helper function to extract exactly one child node
fn extract_one_node<F: EvalexprFloat>(mut children: Vec<Node<F>>) -> EvalexprResult<Node<F>, F> {
    expect_operator_argument_amount(children.len(), 1)?;
    Ok(children.pop().unwrap())
}

/// Helper function to extract exactly two child nodes
fn extract_two_nodes<F: EvalexprFloat>(
    mut children: Vec<Node<F>>,
) -> EvalexprResult<[Node<F>; 2], F> {
    expect_operator_argument_amount(children.len(), 2)?;
    let mut b = children.pop().unwrap();
    let mut a = children.pop().unwrap();
    if b.operator() == &Operator::RootNode && b.children.len() == 1 {
        b = b.children.pop().unwrap();
    }
    if a.operator() == &Operator::RootNode && a.children.len() == 1 {
        a = a.children.pop().unwrap();
    }

    Ok([a, b])
}

fn into_u32<F: EvalexprFloat>(value: usize) -> EvalexprResult<u32, F> {
    value.try_into().map_err(|_| {
        EvalexprError::CustomMessage("Length of tuples cannot exceed u32::MAX".to_string())
    })
}
/// Recursively compile a Node tree into flat operations
/// This validates the tree structure during compilation (like try_into for CompiledNode)
fn compile_to_flat_inner<F: EvalexprFloat>(
    node: Node<F>,
    ops: &mut Vec<FlatOperator<F>>,
) -> EvalexprResult<(), F> {
    use Operator::*;

    match node.operator {
        RootNode => {
            if node.children.len() > 1 {
                return Err(EvalexprError::wrong_operator_argument_amount(
                    node.children.len(),
                    1,
                ));
            }

            if let Some(child) = node.children.into_iter().next() {
                compile_to_flat_inner(child, ops)?;
            } else {
                // Empty expression
                ops.push(FlatOperator::PushConst {
                    value: Value::Empty,
                });
            }
        },

        // Binary operators - compile left child, right child, then operation
        Add => {
            let [a, b] = extract_two_nodes(node.children)?;

            // a + (b1 * b2) or (a1 * a2) + b
            let FusingReturn::DidNotFuse(a, b) =
                fuse_comutative(ops, a, b, Mul, FlatOperator::MulAdd)?
            else {
                return Ok(());
            };
            // a + (b1 / b2) or (a1 / a2) + b
            let FusingReturn::DidNotFuse(a, b) =
                fuse_comutative(ops, a, b, Div, FlatOperator::DivAdd)?
            else {
                return Ok(());
            };

            //  x + C or
            if let Some(c) = extract_const_float(&a)? {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::AddConst { value: c });
                return Ok(());
            }
            //C + x
            if let Some(c) = extract_const_float(&b)? {
                compile_to_flat_inner(a, ops)?;
                ops.push(FlatOperator::AddConst { value: c });
                return Ok(());
            }

            nary_op(ops, a, b, Add, FlatOperator::Add, |n| FlatOperator::AddN {
                n,
            })?;
        },
        Sub => {
            let [a, b] = extract_two_nodes(node.children)?;
            // (a1 * a2) - b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Mul, FlatOperator::MulSub)?
            else {
                return Ok(());
            };
            // (a1 / a2) - b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Div, FlatOperator::DivSub)?
            else {
                return Ok(());
            };

            //  x - C
            if let Some(c) = extract_const_float(&b)? {
                compile_to_flat_inner(a, ops)?;
                ops.push(FlatOperator::SubConst { value: c });
                return Ok(());
            }
            //  C - x
            if let Some(c) = extract_const_float(&a)? {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::ConstSub { value: c });
                return Ok(());
            }

            nary_op(ops, a, b, Sub, FlatOperator::Sub, |n| FlatOperator::SubN {
                n,
            })?;
        },
        Mul => {
            let [a, b] = extract_two_nodes(node.children)?;

            //  (a1 + a2) * b  or a * (b1 + b2)
            let FusingReturn::DidNotFuse(a, b) =
                fuse_comutative(ops, a, b, Add, FlatOperator::AddMul)?
            else {
                return Ok(());
            };
            //  (a1 - a2) * b  or a * (b1 - b2)
            let FusingReturn::DidNotFuse(a, b) =
                fuse_comutative(ops, a, b, Sub, FlatOperator::SubMul)?
            else {
                return Ok(());
            };

            // DivMul: (a1 / a2) * b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Div, FlatOperator::DivMul)?
            else {
                return Ok(());
            };

            //  x * C
            if let Some(c) = extract_const_float(&b)? {
                compile_to_flat_inner(a, ops)?;
                ops.push(FlatOperator::MulConst { value: c });
                return Ok(());
            }
            //  C * x
            if let Some(c) = extract_const_float(&a)? {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::MulConst { value: c });
                return Ok(());
            }

            nary_op(ops, a, b, Mul, FlatOperator::Mul, |n| FlatOperator::MulN {
                n,
            })?;
        },
        Div => {
            let [a, b] = extract_two_nodes(node.children)?;

            //  (a1 + a2) / b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Add, FlatOperator::AddDiv)?
            else {
                return Ok(());
            };
            //  (a1 - a2) / b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Sub, FlatOperator::SubDiv)?
            else {
                return Ok(());
            };

            // MulDiv: (a1 * a2) / b
            let FusingReturn::DidNotFuse(a, b) = fuse_left(ops, a, b, Mul, FlatOperator::MulDiv)?
            else {
                return Ok(());
            };

            //  x / C
            if let Some(c) = extract_const_float(&b)? {
                compile_to_flat_inner(a, ops)?;
                ops.push(FlatOperator::DivConst { value: c });
                return Ok(());
            }
            //  C / x
            if let Some(c) = extract_const_float(&a)? {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::ConstDiv { value: c });
                return Ok(());
            }

            nary_op(ops, a, b, Div, FlatOperator::Div, |n| FlatOperator::DivN {
                n,
            })?;
        },
        Mod => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            if let Some(c) = extract_const_float(&b)? {
                ops.push(FlatOperator::ModConst { value: c });
            } else {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::Mod);
            }
            // compile_to_flat_inner(b, ops)?;
            // ops.push(FlatOperator::Mod);
        },
        Exp => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            if let Some(c) = extract_const_float(&b)? {
                if c == F::f64_to_float(2.0) {
                    ops.push(FlatOperator::Square);
                } else if c == F::f64_to_float(3.0) {
                    ops.push(FlatOperator::Cube);
                } else {
                    ops.push(FlatOperator::ExpConst { value: c });
                }
            } else {
                compile_to_flat_inner(b, ops)?;
                ops.push(FlatOperator::Exp);
            }
            // compile_to_flat_inner(a, ops)?;
            // compile_to_flat_inner(b, ops)?;
            // ops.push(FlatOperator::Exp);
        },

        // Unary operators
        Neg => {
            let child = extract_one_node(node.children)?;
            if let Operator::Const { value } = &child.operator {
                let val = value.as_float()?;
                ops.push(FlatOperator::PushConst {
                    value: Value::Float(-val),
                });
            } else if let &Operator::VariableIdentifierRead { identifier } = &child.operator {
                ops.push(FlatOperator::ReadVarNeg { identifier });
            } else {
                compile_to_flat_inner(child, ops)?;
                ops.push(FlatOperator::Neg);
            }
        },
        Not => {
            let child = extract_one_node(node.children)?;
            compile_to_flat_inner(child, ops)?;
            ops.push(FlatOperator::Not);
        },
        Factorial => {
            let child = extract_one_node(node.children)?;
            compile_to_flat_inner(child, ops)?;
            ops.push(FlatOperator::Factorial);
        },

        // Comparison operators
        Eq => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Eq);
        },
        Neq => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Neq);
        },
        Gt => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Gt);
        },
        Lt => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Lt);
        },
        Geq => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Geq);
        },
        Leq => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Leq);
        },

        // Logical operators
        And => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::And);
        },
        Or => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Or);
        },

        // Assignment operators
        Assign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?; // Value first
            compile_to_flat_inner(a, ops)?; // Variable second (should emit WriteVar)
            ops.push(FlatOperator::Assign);
        },
        AddAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::AddAssign);
        },
        SubAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::SubAssign);
        },
        MulAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::MulAssign);
        },
        DivAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::DivAssign);
        },
        ModAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::ModAssign);
        },
        ExpAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::ExpAssign);
        },
        AndAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::AndAssign);
        },
        OrAssign => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(b, ops)?;
            compile_to_flat_inner(a, ops)?;
            ops.push(FlatOperator::OrAssign);
        },

        // Variable-length operators
        Tuple => {
            let len = into_u32(node.children.len())?;
            for child in node.children {
                compile_to_flat_inner(child, ops)?;
            }
            ops.push(FlatOperator::Tuple { len });
        },
        Chain => {
            let len = into_u32(node.children.len())?;
            for child in node.children {
                compile_to_flat_inner(child, ops)?;
            }
            ops.push(FlatOperator::Chain { len });
        },
        Range => {
            let [a, b] = extract_two_nodes(node.children)?;
            compile_to_flat_inner(a, ops)?;
            compile_to_flat_inner(b, ops)?;
            ops.push(FlatOperator::Range);
        },

        // Leaf nodes
        Const { value } => {
            ops.push(FlatOperator::PushConst { value });
        },
        VariableIdentifierRead { identifier } => {
            if let Some(FlatOperator::ReadVar {
                identifier: other_identifier,
            }) = ops.last()
            {
                if identifier == *other_identifier {
                    ops.push(FlatOperator::Duplicate);
                } else {
                    ops.push(FlatOperator::ReadVar { identifier });
                }
            } else {
                ops.push(FlatOperator::ReadVar { identifier });
            }
        },
        VariableIdentifierWrite { identifier } => {
            ops.push(FlatOperator::WriteVar { identifier });
        },
        FunctionIdentifier { identifier } => {
            let CompileNativeResult::NotNative(node) =
                compile_special_function(ops, identifier, node)?
            else {
                return Ok(());
            };

            let len = node.children.len();
            for child in node.children {
                compile_to_flat_inner(child, ops)?;
            }
            if let Some(op) = compile_operator_function(identifier, len)? {
                ops.push(op);
            } else {
                ops.push(FlatOperator::FunctionCall {
                    identifier,
                    arg_num: into_u32(len)?,
                });
            }
        },
    }

    Ok(())
}

enum CompileNativeResult<F: EvalexprFloat> {
    Compiled,
    NotNative(Node<F>),
}
fn compile_special_function<F: EvalexprFloat>(
    ops: &mut Vec<FlatOperator<F>>,
    identifier: IStr,
    mut node: Node<F>,
) -> EvalexprResult<CompileNativeResult<F>, F> {
    match identifier.to_str() {
        // "Deriv" | "Derivative" | "D" => Ok(CompileNativeResult::NotNative(node)),
        "Sum" | "Product" => {
            let len = node.children.len();
            if len != 3 {
                return Err(EvalexprError::CustomMessage(format!(
                    "Sum/Product functions must have 3 arguments: {identifier}(tuple_expr, \
                     variable_name, expression)"
                )));
            }
            let expr_child = node.children.pop().unwrap();
            let variable_name = node.children.pop().unwrap();
            let tuple_expr = node.children.pop().unwrap();

            compile_to_flat_inner(tuple_expr, ops)?;

            let Operator::VariableIdentifierRead {
                identifier: variable_ident,
            } = variable_name.operator
            else {
                return Err(EvalexprError::CustomMessage(
                    "Second argument of Sum function must be a variable name: {identifier}"
                        .to_string(),
                ));
            };

            let exp_node = compile_to_flat(expr_child)?;
            match identifier.to_str() {
                "Sum" => {
                    ops.push(FlatOperator::Sum {
                        variable: variable_ident,
                        expr: Box::new(exp_node),
                    });
                },
                "Product" => {
                    ops.push(FlatOperator::Product {
                        variable: variable_ident,
                        expr: Box::new(exp_node),
                    });
                },
                _ => unreachable!(),
            }
            Ok(CompileNativeResult::Compiled)
        },
        _ => Ok(CompileNativeResult::NotNative(node)),
    }
}

/// Convert Node directly to FlatNode (similar to Node -> CompiledNode)
pub fn compile_to_flat<F: EvalexprFloat>(node: Node<F>) -> EvalexprResult<FlatNode<F>, F> {
    let mut ops = Vec::new();
    compile_to_flat_inner(node, &mut ops)?;
    Ok(FlatNode { ops })
    // Ok(FlatNode { ops })
}
/// Stack type
#[derive(Default)]
pub struct Stack<T: EvalexprFloat, const MAX_FUNCTION_NESTING: usize = 512> {
    stack: Vec<Value<T>>,
    function_nesting: usize,
    num_args: usize,
}
impl<T: EvalexprFloat, const MAX_FUNCTION_NESTING: usize> Stack<T, MAX_FUNCTION_NESTING> {
    /// Create new stack
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            function_nesting: 0,
            num_args: 0,
        }
    }
    /// Create new stack with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            stack: Vec::with_capacity(capacity),
            function_nesting: 0,
            num_args: 0,
        }
    }
    #[inline(always)]
    fn push(&mut self, value: Value<T>) {
        self.stack.push(value);
    }
    /// Returns true if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
    fn pop(&mut self) -> Option<Value<T>> {
        self.stack.pop()
    }
    fn last(&self) -> Option<&Value<T>> {
        self.stack.last()
    }
    pub(crate) fn function_called(&mut self) -> EvalexprResult<(), T> {
        if self.function_nesting > MAX_FUNCTION_NESTING {
            return Err(EvalexprError::StackOverflow);
        }
        self.function_nesting += 1;
        Ok(())
    }
    pub(crate) fn function_returned(&mut self) {
        self.function_nesting -= 1;
    }

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
            pop_unchecked(self);
            self.num_args -= 1;
        }
        // self.stack.truncate(self.stack.len() - self.num_args);
        // self.num_args = 0;
    }
    /// Returns the number of arguments on the stack.
    pub fn num_args(&self) -> usize {
        self.num_args
    }
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
    fn len(&self) -> usize {
        self.stack.len()
    }
    fn truncate(&mut self, len: usize) {
        self.stack.truncate(len);
    }
    fn drain(&mut self, range: impl RangeBounds<usize>) -> Drain<'_, Value<T>> {
        self.stack.drain(range)
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

fn pop_unchecked<T: EvalexprFloat, const MAX_FUNCTION_NESTING: usize>(
    stack: &mut Stack<T, MAX_FUNCTION_NESTING>,
) -> Value<T> {
    debug_assert!(!stack.is_empty());
    unsafe { stack.pop().unwrap_unchecked() }
}
impl<F: EvalexprFloat> FlatNode<F> {
    /// Returns the constant value of this node it it only contains a single PushConst operator.
    pub fn as_constant(&self) -> Option<Value<F>> {
        if self.ops.len() == 1 {
            if let FlatOperator::PushConst { value } = &self.ops[0] {
                return Some(value.clone());
            }
        }
        None
    }
    /// Evaluate the flat node using a stack-based approach
    #[inline(always)]
    pub fn eval_priv<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
        override_vars: &[(IStr, F)],
    ) -> EvalexprResultValue<F> {
        let stack_size = stack.len();
        let prev_num_args = stack.num_args;
        match self.eval_priv_inner(stack, context, override_vars) {
            Ok(value) => {
                debug_assert_eq!(stack_size, stack.len());
                debug_assert_eq!(stack.num_args, 0);
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
    #[inline(always)]
    fn eval_priv_inner<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
        override_vars: &[(IStr, F)],
    ) -> EvalexprResultValue<F> {
        use FlatOperator::*;

        // println!("{:?}", self.ops);
        for op in &self.ops {
            // println!("{:?}", op);
            match op {
                // Binary arithmetic operators
                Add => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a + b));
                },
                Sub => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a - b));
                },
                Mul => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a * b));
                },
                Div => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a / b));
                },
                Mod => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a % b));
                },
                Exp => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a.pow(&b)));
                },
                Neg => {
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(-a));
                },

                // Comparison operators
                Eq => {
                    let b = pop_unchecked(stack);
                    let a = pop_unchecked(stack);
                    stack.push(Value::Boolean(a == b));
                },
                Neq => {
                    let b = pop_unchecked(stack);
                    let a = pop_unchecked(stack);
                    stack.push(Value::Boolean(a != b));
                },
                Gt => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Boolean(a > b));
                },
                Lt => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Boolean(a < b));
                },
                Geq => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Boolean(a >= b));
                },
                Leq => {
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Boolean(a <= b));
                },

                // Logical operators
                And => {
                    let b = pop_unchecked(stack).as_boolean()?;
                    let a = pop_unchecked(stack).as_boolean()?;
                    stack.push(Value::Boolean(a && b));
                },
                Or => {
                    let b = pop_unchecked(stack).as_boolean()?;
                    let a = pop_unchecked(stack).as_boolean()?;
                    stack.push(Value::Boolean(a || b));
                },
                Not => {
                    let a = pop_unchecked(stack).as_boolean()?;
                    stack.push(Value::Boolean(!a));
                },

                // Assignment operators
                Assign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                AddAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                SubAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                MulAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                DivAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                ModAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                ExpAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                AndAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },
                OrAssign => {
                    return Err(EvalexprError::ContextNotMutable);
                },

                // Variable-length operators
                Tuple { len } => {
                    // Special case: 2-element tuple with floats becomes Float2
                    if *len == 2 {
                        let b = pop_unchecked(stack);
                        let a = pop_unchecked(stack);

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
                Chain { len } => {
                    debug_assert!(
                        *len > 0,
                        "Chain with 0 length should be caught at compile time"
                    );

                    todo!()
                    // // Keep only the last value, discard the rest
                    // let start_idx = stack.len() - *len as usize;
                    // stack.drain(start_idx..stack.len() - 1);
                },
                FunctionCall {
                    identifier,
                    arg_num,
                } => {
                    let prev_num_args = stack.num_args;
                    stack.num_args = *arg_num as usize;
                    let result = match context.unchecked_call_function(stack, *identifier) {
                        Err(EvalexprError::FunctionIdentifierNotFound(_))
                            if !context.are_builtin_functions_disabled() =>
                        {
                            if let Some(builtin_function) = builtin_function(identifier) {
                                builtin_function.call(stack, context)?
                            } else {
                                return Err(EvalexprError::FunctionIdentifierNotFound(
                                    identifier.to_string(),
                                ));
                            }
                        },
                        Ok(val) => val,
                        Err(e) => return Err(e),
                    };

                    stack.pop_args();
                    stack.num_args = prev_num_args;
                    stack.push(result);
                },

                // Constants and variables
                PushConst { value } => {
                    stack.push(value.clone());
                },
                ReadVar { identifier } => {
                    let value = read_var(identifier, stack, context, override_vars)?;
                    stack.push(value);
                },
                ReadVarNeg { identifier } => {
                    let value = read_var(identifier, stack, context, override_vars)?;
                    stack.push(Value::Float(-value.as_float()?));
                },
                WriteVar { .. } => {
                    // Not implemented in immutable context
                    return Err(EvalexprError::ContextNotMutable);
                },

                // N-ary operations
                AddN { n } => {
                    let n = *n as usize;
                    let mut sum = pop_unchecked(stack).as_float()?;
                    for _ in 0..n - 1 {
                        sum = sum + pop_unchecked(stack).as_float()?;
                    }
                    stack.push(Value::Float(sum));
                },
                SubN { n } => {
                    let n = *n as usize;
                    let mut diff = pop_unchecked(stack).as_float()?;
                    for _ in 0..n - 1 {
                        diff = diff - pop_unchecked(stack).as_float()?;
                    }
                    stack.push(Value::Float(diff));
                },
                MulN { n } => {
                    let n = *n as usize;
                    let mut product = pop_unchecked(stack).as_float()?;
                    for _ in 0..n - 1 {
                        product = product * pop_unchecked(stack).as_float()?;
                    }
                    stack.push(Value::Float(product));
                },
                DivN { n } => {
                    let n = *n as usize;
                    let mut res = pop_unchecked(stack).as_float()?;
                    for _ in 0..n - 1 {
                        res = res / pop_unchecked(stack).as_float()?;
                    }
                    stack.push(Value::Float(res));
                },

                // Fused operations
                MulAdd => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a * b + c));
                },
                MulSub => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a * b - c));
                },
                DivAdd => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a / b + c));
                },
                DivSub => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a / b - c));
                },
                AddMul => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float((a + b) * c));
                },
                SubMul => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float((a - b) * c));
                },
                AddDiv => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float((a + b) / c));
                },
                SubDiv => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float((a - b) / c));
                },
                MulDiv => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a * b / c));
                },
                DivMul => {
                    let c = pop_unchecked(stack).as_float()?;
                    let b = pop_unchecked(stack).as_float()?;
                    let a = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(a / b * c));
                },
                MulConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x * *value));
                },
                AddConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x + *value));
                },
                DivConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x / *value));
                },
                ConstDiv { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(*value / x));
                },
                SubConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x - *value));
                },
                ConstSub { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(*value - x));
                },
                ExpConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.pow(value)));
                },
                ModConst { value } => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x % *value));
                },

                // Specialized math operations
                Square => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x * x));
                },
                Cube => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x * x * x));
                },
                Sqrt => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.sqrt()));
                },
                Cbrt => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.cbrt()));
                },

                Abs => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.abs()));
                },
                Floor => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.floor()));
                },
                Round => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.round()));
                },
                Ceil => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.ceil()));
                },
                Ln => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.ln()));
                },
                Log => {
                    let x = pop_unchecked(stack).as_float()?;
                    let base = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.log(&base)));
                },
                Log2 => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.log2()));
                },
                Log10 => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.log10()));
                },
                ExpE => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.exp()));
                },
                Exp2 => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.exp2()));
                },
                Cos => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.cos()));
                },
                Acos => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.acos()));
                },
                CosH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.cosh()));
                },
                AcosH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.acosh()));
                },
                Sin => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.sin()));
                },
                Asin => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.asin()));
                },
                SinH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.sinh()));
                },
                AsinH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.asinh()));
                },
                Tan => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.tan()));
                },
                Atan => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.atan()));
                },
                TanH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.tanh()));
                },
                AtanH => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.atanh()));
                },
                Atan2 => {
                    let x = pop_unchecked(stack).as_float()?;
                    let y = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(y.atan2(&x)));
                },
                Hypot => {
                    let y = pop_unchecked(stack).as_float()?;
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.hypot(&y)));
                },

                Signum => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.signum()));
                },

                Min => {
                    let y = pop_unchecked(stack).as_float()?;
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.min(&y)));
                },
                Max => {
                    let y = pop_unchecked(stack).as_float()?;
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.max(&y)));
                },
                Clamp => {
                    let mut max = pop_unchecked(stack).as_float()?;
                    let mut min = pop_unchecked(stack).as_float()?;
                    let x = pop_unchecked(stack).as_float()?;
                    if min > max {
                        core::mem::swap(&mut min, &mut max);
                    }
                    stack.push(Value::Float(x.clamp(&min, &max)));
                },
                Factorial => {
                    let x = pop_unchecked(stack).as_float()?;
                    stack.push(Value::Float(x.factorial()));
                },

                Range => {
                    let end = pop_unchecked(stack);
                    let start = pop_unchecked(stack);

                    let result = eval_range(start, end)?;

                    stack.push(Value::Tuple(result))
                },
                RangeWithStep => {
                    let step = pop_unchecked(stack);
                    let end = pop_unchecked(stack);
                    let start = pop_unchecked(stack);
                    let result = eval_range_with_step(start, end, step)?;
                    stack.push(Value::Tuple(result))
                },
                // Native functions
                Sum { variable, expr } => {
                    let tuple = pop_unchecked(stack).as_tuple()?;
                    let mut result = F::ZERO;
                    let mut o_override_vars: SmallVec<[(IStr, F); 5]> =
                        smallvec::smallvec![(*variable, F::ZERO)];
                    o_override_vars.extend_from_slice(override_vars);

                    for value in tuple {
                        o_override_vars[0].1 = value.as_float()?;
                        let current = expr
                            .eval_priv_inner(stack, context, &o_override_vars)?
                            .as_float()?;
                        result = result + current;
                    }
                    stack.push(Value::Float(result));
                },
                Product { variable, expr } => {
                    let tuple = pop_unchecked(stack).as_tuple()?;
                    let mut result = F::ONE;
                    let mut o_override_vars: SmallVec<[(IStr, F); 5]> =
                        smallvec::smallvec![(*variable, F::ZERO)];
                    o_override_vars.extend_from_slice(override_vars);

                    for value in tuple {
                        o_override_vars[0].1 = value.as_float()?;
                        let current = expr
                            .eval_priv_inner(stack, context, &o_override_vars)?
                            .as_float()?;
                        result = result * current;
                    }
                    stack.push(Value::Float(result));
                },
                Duplicate => {
                    let value = stack.last().cloned().unwrap();
                    stack.push(value.clone());
                },
            }
        }

        // Return the top value or Empty if stack is empty
        Ok(stack.pop().unwrap_or(Value::Empty))
    }
    /// Evaluates the operator with the given arguments and mutable context.
    fn eval_mut_priv<C: ContextWithMutableVariables + Context<NumericTypes = F>>(
        &self,
        _stack: &mut Stack<F>,
        _context: &mut C,
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
    /// Evaluates the operator tree rooted at this node with empty context.
    pub fn eval(&self) -> EvalexprResultValue<F> {
        let context = HashMapContext::<F>::new();
        let mut stack = Stack::new();
        self.eval_priv(&mut stack, &context, &[])
    }
    /// Evaluates the operator tree rooted at this node with the given context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_with_context<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
    ) -> EvalexprResultValue<F> {
        self.eval_priv(stack, context, &[])
    }
    /// Evaluates the operator tree rooted at this node with the given mutable context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_with_context_mut<C: ContextWithMutableVariables + Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &mut C,
    ) -> EvalexprResultValue<F> {
        self.eval_mut_priv(stack, context)
    }

    /// Evaluates the operator tree rooted at this node into a float with an the given context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_float_with_context<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
    ) -> EvalexprResult<F, F> {
        match self.eval_with_context(stack, context) {
            Ok(Value::Float(float)) => Ok(float),
            Ok(value) => Err(EvalexprError::expected_float(value)),
            Err(error) => Err(error),
        }
    }
    /// Evaluates the operator tree rooted at this node with the given context and override vars
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_with_context_and_override<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
        override_vars: &[(IStr, F)],
    ) -> EvalexprResultValue<F> {
        self.eval_priv(stack, context, override_vars)
    }

    /// Evaluates the operator tree rooted at this node into a float with an the given context.
    /// If the result of the expression is an integer, it is silently converted into a float.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_float_with_context_and_override<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
        override_vars: &[(IStr, F)],
    ) -> EvalexprResult<F, F> {
        match self.eval_with_context_and_override(stack, context, override_vars) {
            Ok(Value::Float(float)) => Ok(float),
            Ok(value) => Err(EvalexprError::expected_float(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into a boolean with an the given context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_boolean_with_context<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
    ) -> EvalexprResult<bool, F> {
        match self.eval_with_context(stack, context) {
            Ok(Value::Boolean(boolean)) => Ok(boolean),
            Ok(value) => Err(EvalexprError::expected_boolean(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into a tuple with an the given context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_tuple_with_context<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
    ) -> EvalexprResult<TupleType<F>, F> {
        match self.eval_with_context(stack, context) {
            Ok(Value::Tuple(tuple)) => Ok(tuple),
            Ok(value) => Err(EvalexprError::expected_tuple(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into an empty value with an the given context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_empty_with_context<C: Context<NumericTypes = F>>(
        &self,
        stack: &mut Stack<F>,
        context: &C,
    ) -> EvalexprResult<EmptyType, F> {
        match self.eval_with_context(stack, context) {
            Ok(Value::Empty) => Ok(EMPTY_VALUE),
            Ok(value) => Err(EvalexprError::expected_empty(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into a float with an the given mutable context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_float_with_context_mut<
        C: ContextWithMutableVariables + Context<NumericTypes = F>,
    >(
        &self,
        stack: &mut Stack<F>,
        context: &mut C,
    ) -> EvalexprResult<F, F> {
        match self.eval_with_context_mut(stack, context) {
            Ok(Value::Float(float)) => Ok(float),
            Ok(value) => Err(EvalexprError::expected_float(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into a boolean with an the given mutable context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_boolean_with_context_mut<
        C: ContextWithMutableVariables + Context<NumericTypes = F>,
    >(
        &self,
        stack: &mut Stack<F>,
        context: &mut C,
    ) -> EvalexprResult<bool, F> {
        match self.eval_with_context_mut(stack, context) {
            Ok(Value::Boolean(boolean)) => Ok(boolean),
            Ok(value) => Err(EvalexprError::expected_boolean(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into a tuple with an the given mutable context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_tuple_with_context_mut<
        C: ContextWithMutableVariables + Context<NumericTypes = F>,
    >(
        &self,
        stack: &mut Stack<F>,
        context: &mut C,
    ) -> EvalexprResult<TupleType<F>, F> {
        match self.eval_with_context_mut(stack, context) {
            Ok(Value::Tuple(tuple)) => Ok(tuple),
            Ok(value) => Err(EvalexprError::expected_tuple(value)),
            Err(error) => Err(error),
        }
    }

    /// Evaluates the operator tree rooted at this node into an empty value with an the given mutable context.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_empty_with_context_mut<
        C: ContextWithMutableVariables + Context<NumericTypes = F>,
    >(
        &self,
        stack: &mut Stack<F>,
        context: &mut C,
    ) -> EvalexprResult<EmptyType, F> {
        match self.eval_with_context_mut(stack, context) {
            Ok(Value::Empty) => Ok(EMPTY_VALUE),
            Ok(value) => Err(EvalexprError::expected_empty(value)),
            Err(error) => Err(error),
        }
    }
    /// Evaluates the operator tree rooted at this node into a float.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_float(&self) -> EvalexprResult<F, F> {
        let mut stack = Stack::new();
        self.eval_float_with_context_mut(&mut stack, &mut HashMapContext::new())
    }
    /// Evaluates the operator tree rooted at this node into a boolean.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_boolean(&self) -> EvalexprResult<bool, F> {
        let mut stack = Stack::new();
        self.eval_boolean_with_context_mut(&mut stack, &mut HashMapContext::new())
    }

    /// Evaluates the operator tree rooted at this node into a tuple.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_tuple(&self) -> EvalexprResult<TupleType<F>, F> {
        let mut stack = Stack::new();
        self.eval_tuple_with_context_mut(&mut stack, &mut HashMapContext::new())
    }

    /// Evaluates the operator tree rooted at this node into an empty value.
    ///
    /// Fails, if one of the operators in the expression tree fails.
    pub fn eval_empty(&self) -> EvalexprResult<EmptyType, F> {
        let mut stack = Stack::new();
        self.eval_empty_with_context_mut(&mut stack, &mut HashMapContext::new())
    }

    /// Returns an iterator over all nodes in this tree.
    pub fn iter(&self) -> impl Iterator<Item = &FlatOperator<F>> {
        let mut ops = Vec::new();
        for op in self.ops.iter() {
            match op {
                FlatOperator::Product { expr, .. } | FlatOperator::Sum { expr, .. } => {
                    ops.extend(expr.iter());
                },
                op => {
                    ops.push(op);
                },
            }
        }
        ops.into_iter()
    }
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
        self.iter().filter_map(|node| match node {
            FlatOperator::ReadVar { identifier }
            | FlatOperator::ReadVarNeg { identifier }
            | FlatOperator::WriteVar { identifier }
            | FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
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
        self.iter().filter_map(|node| match node {
            FlatOperator::ReadVar { identifier }
            | FlatOperator::ReadVarNeg { identifier }
            | FlatOperator::WriteVar { identifier } => Some(identifier.to_str()),
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
        self.iter().filter_map(|node| match node {
            FlatOperator::ReadVar { identifier } => Some(identifier.to_str()),
            FlatOperator::ReadVarNeg { identifier } => Some(identifier.to_str()),
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
        self.iter().filter_map(|node| match node {
            FlatOperator::WriteVar { identifier } => Some(identifier.to_str()),
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
        self.iter().filter_map(|node| match node {
            FlatOperator::FunctionCall { identifier, .. } => Some(identifier.to_str()),
            _ => None,
        })
    }
}

fn read_var<F: EvalexprFloat>(
    identifier: &IStr,
    stack: &mut Stack<F>,
    context: &impl Context<NumericTypes = F>,
    override_vars: &[(IStr, F)],
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
        match context.unchecked_call_function(stack, *identifier) {
            Err(EvalexprError::FunctionIdentifierNotFound(_))
                if !context.are_builtin_functions_disabled() =>
            {
                return Err(EvalexprError::VariableIdentifierNotFound(
                    identifier.to_string(),
                ));
            },
            Ok(val) => {
                stack.num_args = prev_num_args;

                val
            },
            Err(e) => return Err(e),
        }
    })
}
