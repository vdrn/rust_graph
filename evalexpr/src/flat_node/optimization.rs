use crate::{
    error::expect_function_argument_amount,
    flat_node::{compile_to_flat_inner, FlatOperator},
    EvalexprError, EvalexprFloat, EvalexprResult, IStr, Node, Operator,
};

pub enum FusingReturn<T: EvalexprFloat> {
    Fused,
    DidNotFuse(Node<T>, Node<T>),
}
pub fn fuse_comutative<T: EvalexprFloat>(
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
pub fn fuse_left<T: EvalexprFloat>(
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

pub fn nary_op<T: EvalexprFloat>(
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

/// Helper to recursively collect all operands of the same binary operator
/// For example: Add(Add(a, b), c) -> [a, b, c]
pub fn collect_same_operator<NumericTypes: EvalexprFloat>(
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
pub fn extract_const_float<T: EvalexprFloat>(
    node: &Node<T>,
) -> EvalexprResult<Option<T>, T> {
    match &node.operator {
        Operator::Const { value } => Ok(Some(value.as_float()?)),
        _ => Ok(None),
    }
}
/// Detect function calls that can be replaced with specialized ops
pub fn compile_specialized_function<NumericTypes: EvalexprFloat>(
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
        "fact"|"factorial" => {
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
