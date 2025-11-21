use std::fmt::{Display, Error, Formatter};

use crate::Value;

use super::numeric_types::EvalexprFloat;

impl<NumericTypes: EvalexprFloat> Display for Value<NumericTypes> {
	fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
		match self {
			// Value::String(string) => write!(f, "\"{}\"", string),
			Value::Float(float) => write!(f, "{}", float.human_display(false)),
			Value::Float2(f1, f2) => write!(f, "(){},{})", f1.human_display(false), f2.human_display(false)),
			// Value::Int(int) => write!(f, "{}", int),
			Value::Boolean(boolean) => write!(f, "{}", boolean),
			Value::Tuple(tuple) => {
				write!(f, "(")?;
				let mut once = false;
				for value in tuple {
					if once {
						write!(f, ", ")?;
					} else {
						once = true;
					}
					value.fmt(f)?;
				}
				write!(f, ")")
			},
			Value::Empty => write!(f, "()"),
		}
	}
}
impl<NumericTypes: EvalexprFloat> Value<NumericTypes> {
  /// Returns a string representation of `self` for human display.
  pub fn human_display(&self, rational: bool) -> String {
    match self {
      Value::Float(float) => float.human_display(rational),
      Value::Float2(f1, f2) => format!("({},{})", f1.human_display(rational), f2.human_display(rational)),
      Value::Boolean(boolean) => boolean.to_string(),
      Value::Tuple(tuple) => {
        let mut result = String::new();
        result.push('(');
        let mut once = false;
        for value in tuple {
          if once {
            result.push_str(", ");
          } else {
            once = true;
          }
          result.push_str(&value.human_display(rational));
        }
        result.push(')');
        result
      },
      Value::Empty => String::from("()"),
    }
  }
}
