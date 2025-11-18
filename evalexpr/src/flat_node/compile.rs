use crate::{
    EvalexprError, EvalexprFloat, EvalexprResult, FlatNode, IStr, Node, Operator, Value, error::{expect_function_argument_amount, expect_operator_argument_amount}, flat_node::FlatOperator
};
/// Helper function to extract exactly one child node
fn extract_one_node<F: EvalexprFloat>(mut children: Vec<Node<F>>) -> EvalexprResult<Node<F>, F> {
    expect_operator_argument_amount(children.len(), 1)?;
    Ok(children.pop().unwrap())
}

/// Compile Node to FlatNode
pub fn compile_to_flat<F: EvalexprFloat>(node: Node<F>) -> EvalexprResult<FlatNode<F>, F> {
    let mut ops = Vec::new();
    compile_to_flat_inner(node, &mut ops)?;
    Ok(FlatNode { ops })
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
            ops.push(FlatOperator::ReadVar { identifier });
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

enum FusingReturn<T: EvalexprFloat> {
    Fused,
    DidNotFuse(Node<T>, Node<T>),
}
fn fuse_comutative<T: EvalexprFloat>(
    ops: &mut Vec<FlatOperator<T>>,
    a: Node<T>,
    b: Node<T>,
    fusing_op: Operator<T>,
    result_op: FlatOperator<T>,
) -> EvalexprResult<FusingReturn<T>, T> {
    if a.operator == fusing_op && a.children.len() == 2 {
        let mut a_children = a.children;
        let a_right = a_children.pop().unwrap();
        let a_left = a_children.pop().unwrap();

        compile_to_flat_inner(a_left, ops)?;
        compile_to_flat_inner(a_right, ops)?;
        compile_to_flat_inner(b, ops)?;
        ops.push(result_op);
        Ok(FusingReturn::Fused)
    } else if b.operator == fusing_op && b.children.len() == 2 {
        let mut b_children = b.children;
        let b_right = b_children.pop().unwrap();
        let b_left = b_children.pop().unwrap();

        compile_to_flat_inner(b_left, ops)?;
        compile_to_flat_inner(b_right, ops)?;
        compile_to_flat_inner(a, ops)?;
        ops.push(result_op);
        Ok(FusingReturn::Fused)
    } else {
        Ok(FusingReturn::DidNotFuse(a, b))
    }
}
fn fuse_left<T: EvalexprFloat>(
    ops: &mut Vec<FlatOperator<T>>,
    a: Node<T>,
    b: Node<T>,
    fusing_op: Operator<T>,
    result_op: FlatOperator<T>,
) -> EvalexprResult<FusingReturn<T>, T> {
    if a.operator == fusing_op && a.children.len() == 2 {
        let mut a_children = a.children;
        let a_right = a_children.pop().unwrap();
        let a_left = a_children.pop().unwrap();

        compile_to_flat_inner(a_left, ops)?;
        compile_to_flat_inner(a_right, ops)?;
        compile_to_flat_inner(b, ops)?;
        ops.push(result_op);
        Ok(FusingReturn::Fused)
    } else {
        Ok(FusingReturn::DidNotFuse(a, b))
    }
}

fn nary_op<T: EvalexprFloat>(
    ops: &mut Vec<FlatOperator<T>>,
    a: Node<T>,
    b: Node<T>,

    fusing_op: Operator<T>,
    result_op: FlatOperator<T>,
    result_nary_op: impl Fn(u32) -> FlatOperator<T>,
) -> EvalexprResult<(), T> {
    let operands = collect_same_operator(
        Node {
            operator: fusing_op.clone(),
            children: vec![a, b],
        },
        &fusing_op,
    );

    let n = operands.len();

    if n > 2 {
        for operand in operands.into_iter().rev() {
            compile_to_flat_inner(operand, ops)?;
        }
        ops.push(result_nary_op(n as u32));
    } else {
        for operand in operands {
            compile_to_flat_inner(operand, ops)?;
        }
        ops.push(result_op);
    }
    Ok(())
}

/// recursively collect all operands of the same binary operator
fn collect_same_operator<NumericTypes: EvalexprFloat>(
    node: Node<NumericTypes>,
    target_op: &Operator<NumericTypes>,
) -> Vec<Node<NumericTypes>> {
    if &node.operator == target_op && node.children.len() == 2 {
        let mut children = node.children;
        let right = children.pop().unwrap();
        let left = children.pop().unwrap();

        let mut result = collect_same_operator(left, target_op);
        result.extend(collect_same_operator(right, target_op));
        result
    } else {
        vec![node]
    }
}
fn extract_const_float<T: EvalexprFloat>(node: &Node<T>) -> EvalexprResult<Option<T>, T> {
    match &node.operator {
        Operator::Const { value } => Ok(Some(value.as_float()?)),
        _ => Ok(None),
    }
}

/// Detect function calls that can be replaced with specialized ops
fn compile_operator_function<NumericTypes: EvalexprFloat>(
    ident: IStr,
    arg_num: usize,
) -> EvalexprResult<Option<FlatOperator<NumericTypes>>, NumericTypes> {
    Ok(match ident.to_str() {
        "sqrt" => {
            expect_function_argument_amount(arg_num, 1)?;

            Some(FlatOperator::Sqrt)
        },
        "cbrt" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Cbrt)
        },
        "abs" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Abs)
        },
        "floor" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Floor)
        },
        "round" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Round)
        },
        "ceil" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Ceil)
        },
        "ln" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Ln)
        },
        "log" => {
            expect_function_argument_amount(arg_num, 2)?;
            Some(FlatOperator::Log)
        },
        "log2" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Log2)
        },
        "log10" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Log10)
        },
        "exp" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::ExpE)
        },
        "exp2" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Exp2)
        },
        "cos" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Cos)
        },
        "acos" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Acos)
        },
        "cosh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::CosH)
        },
        "acosh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::AcosH)
        },
        "sin" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Sin)
        },
        "asin" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Asin)
        },
        "sinh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::SinH)
        },
        "asinh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::AsinH)
        },
        "tan" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Tan)
        },
        "atan" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Atan)
        },
        "tanh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::TanH)
        },
        "atanh" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::AtanH)
        },
        "atan2" => {
            expect_function_argument_amount(arg_num, 2)?;
            Some(FlatOperator::Atan2)
        },
        "hypot" => {
            expect_function_argument_amount(arg_num, 2)?;
            Some(FlatOperator::Hypot)
        },
        "signum" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Signum)
        },
        "min" => {
            expect_function_argument_amount(arg_num, 2)?;
            Some(FlatOperator::Min)
        },
        "max" => {
            expect_function_argument_amount(arg_num, 2)?;
            Some(FlatOperator::Max)
        },
        "clamp" => {
            expect_function_argument_amount(arg_num, 3)?;
            Some(FlatOperator::Clamp)
        },
        "fact" | "factorial" => {
            expect_function_argument_amount(arg_num, 1)?;
            Some(FlatOperator::Factorial)
        },
        "range" => {
            if arg_num == 2 {
                Some(FlatOperator::Range)
            } else if arg_num == 3 {
                Some(FlatOperator::RangeWithStep)
            } else {
                return Err(EvalexprError::wrong_function_argument_amount_range(
                    arg_num,
                    2..=3,
                ));
            }
        },

        _ => None,
    })
}
