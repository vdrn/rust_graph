use std::sync::LazyLock;

use evalexpr::{IStr, IStrSet, istr};

const BUILTIN_VARIABLE_NAMES: &[&str] = &["x", "y", "z"];

#[rustfmt::skip]
const BUILTIN_CONSTANTS:&[&str] = &[
  "e", 
  "pi", "π",
  "τ", "tau",
  "∞", "inf", "infinity",
];
#[rustfmt::skip]
const BUILTIN_FUNCTIONS:&[&str] = &[
  "normaldist", "get",

  "is_nan", "is_finite", "is_infinite", "is_normal", "if", "If",
  "contains", "contains_any", "len", "random",

  "∫", "integral", "Integral",
  "∑", "sum", "Sum",
  "∏", "product", "Product",
  "sqrt", "cbrt",
  "abs", "floor", "round", "ceil",
  "ln", "log", "log2", "log10",
  "exp", "exp2",
  "cos", "acos", "cosh", "acosh",
  "sin", "asin", "sinh", "asinh",
  "tan", "atan", "tanh", "atanh", "atan2",
  "hypot",
  "signum", "min", "max", "clamp",
  "fact", "factorial",
  "gcd",
  "range",
];

static BUILTINS: LazyLock<(IStrSet, IStrSet, IStrSet)> = LazyLock::new(|| {
	let mut variables = IStrSet::default();
	let mut constants = IStrSet::default();
	let mut functions = IStrSet::default();

	for name in BUILTIN_VARIABLE_NAMES {
		variables.insert(istr(name));
	}
	for name in BUILTIN_CONSTANTS {
		constants.insert(istr(name));
	}
	for name in BUILTIN_FUNCTIONS {
		functions.insert(istr(name));
	}

	(variables, constants, functions)
});
pub enum BuiltinType {
	Variable,
	Constant,
	Function,
}
impl core::fmt::Display for BuiltinType {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
		match self {
			BuiltinType::Variable => write!(f, "variable"),
			BuiltinType::Constant => write!(f, "constant"),
			BuiltinType::Function => write!(f, "function"),
		}
	}
}
pub fn is_builtin(name: IStr) -> Option<BuiltinType> {
	if BUILTINS.0.contains(&name) {
		return Some(BuiltinType::Variable);
	}
	if BUILTINS.1.contains(&name) {
		return Some(BuiltinType::Constant);
	}
	if BUILTINS.2.contains(&name) {
		return Some(BuiltinType::Function);
	}
	None
}
