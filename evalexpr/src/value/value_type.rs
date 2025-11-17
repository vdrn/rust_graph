use crate::Value;

use super::numeric_types::EvalexprFloat;

/// The type of a `Value`.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum ValueType {
    // /// The `Value::String` type.
    // String,
    /// The `Value::Float` type.
    Float,
    /// The `Value::Float2` type.
    Float2,
    // /// The `Value::Int` type.
    // Int,
    /// The `Value::Boolean` type.
    Boolean,
    /// The `Value::Tuple` type.
    Tuple,
    /// The `Value::Empty` type.
    Empty,
}

impl<NumericTypes: EvalexprFloat> From<&Value<NumericTypes>> for ValueType {
    fn from(value: &Value<NumericTypes>) -> Self {
        match value {
            // Value::String(_) => ValueType::String,
            Value::Float(_) => ValueType::Float,
            Value::Float2(_,_) => ValueType::Float2,
            // Value::Int(_) => ValueType::Int,
            Value::Boolean(_) => ValueType::Boolean,
            Value::Tuple(_) => ValueType::Tuple,
            Value::Empty => ValueType::Empty,
        }
    }
}

impl<NumericTypes: EvalexprFloat> From<&mut Value<NumericTypes>> for ValueType {
    fn from(value: &mut Value<NumericTypes>) -> Self {
        From::<&Value<NumericTypes>>::from(value)
    }
}

impl<NumericTypes: EvalexprFloat> From<&&mut Value<NumericTypes>> for ValueType {
    fn from(value: &&mut Value<NumericTypes>) -> Self {
        From::<&Value<NumericTypes>>::from(*value)
    }
}
