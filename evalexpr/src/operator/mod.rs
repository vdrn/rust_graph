use crate::{
    istr,
    value::{
        numeric_types::{default_numeric_types::DefaultNumericTypes, EvalexprFloat},
        Value,
    },
    IStr,
};

mod display;

/// An enum that represents operators in the operator tree.
#[derive(Debug, PartialEq, Clone)]
pub enum Operator<NumericTypes: EvalexprFloat = DefaultNumericTypes> {
    /// A root node in the operator tree.
    /// The whole expression is stored under a root node, as well as each subexpression surrounded by parentheses.
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
}

impl<NumericTypes: EvalexprFloat> Operator<NumericTypes> {
    pub(crate) fn value(value: Value<NumericTypes>) -> Self {
        Operator::Const { value }
    }

    pub(crate) fn variable_identifier_write(identifier: &str) -> Self {
        Operator::VariableIdentifierWrite {
            identifier: istr(identifier),
        }
    }

    pub(crate) fn variable_identifier_read(identifier: &str) -> Self {
        Operator::VariableIdentifierRead {
            identifier: istr(identifier),
        }
    }

    pub(crate) fn function_identifier(identifier: &str) -> Self {
        Operator::FunctionIdentifier {
            identifier: istr(identifier),
        }
    }

    /// Returns the precedence of the operator.
    /// A high precedence means that the operator has priority to be deeper in the tree.
    pub(crate) const fn precedence(&self) -> i32 {
        use crate::operator::Operator::*;
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

            Assign | AddAssign | SubAssign | MulAssign | DivAssign | ModAssign | ExpAssign
            | AndAssign | OrAssign => 50,

            Tuple => 40,
            Range=>20,
            Chain => 0,

            Const { .. } => 200,
            VariableIdentifierWrite { .. } | VariableIdentifierRead { .. } => 200,
            FunctionIdentifier { .. } => 190,
        }
    }

    /// Returns true if chains of operators with the same precedence as this one should be evaluated left-to-right,
    /// and false if they should be evaluated right-to-left.
    /// Left-to-right chaining has priority if operators with different order but same precedence are chained.
    pub(crate) const fn is_left_to_right(&self) -> bool {
        use crate::operator::Operator::*;
        !matches!(self, Assign | FunctionIdentifier { .. })
    }

    /// Returns true if chains of this operator should be flattened into one operator with many arguments.
    pub(crate) const fn is_sequence(&self) -> bool {
        use crate::operator::Operator::*;
        matches!(self, Tuple | Chain)
    }

    /// True if this operator is a leaf, meaning it accepts no arguments.
    // Make this a const fn as soon as whatever is missing gets stable (issue #57563)
    pub(crate) fn is_leaf(&self) -> bool {
        self.max_argument_amount() == Some(0)
    }

    pub(crate) fn is_postfix(&self)->bool{
        matches!(self, Operator::Factorial)
    }

    /// Returns the maximum amount of arguments required by this operator.
    pub(crate) const fn max_argument_amount(&self) -> Option<usize> {
        use crate::operator::Operator::*;
        match self {
            Add | Sub | Mul | Div | Mod | Exp | Eq | Neq | Gt | Lt | Geq | Leq | And | Or
            | Assign | AddAssign | SubAssign | MulAssign | DivAssign | ModAssign | ExpAssign
            | AndAssign | OrAssign|Range => Some(2),
            Tuple | Chain => None,
            Not | Neg |Factorial| RootNode => Some(1),
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
