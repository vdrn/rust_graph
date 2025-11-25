use std::sync::LazyLock;

use eframe::egui::{self, RichText};
use evalexpr::{EvalexprError, EvalexprFloat, EvalexprResult, IStr, IStrSet, Value, istr};

use crate::entry::f64_to_value;

const BUILTIN_VARIABLE_NAMES: &[&str] = &["x", "y", "z"];

#[rustfmt::skip]
const BUILTIN_CONSTANTS:&[(&str, f64)] = &[
  ("e", core::f64::consts::E),
  ("pi", core::f64::consts::PI),
  ("π", core::f64::consts::PI),
  ("τ", core::f64::consts::TAU),
  ("tau", core::f64::consts::TAU),
  ("∞", f64::INFINITY),
  ("inf", f64::INFINITY),
  ("infinity", f64::INFINITY),
];
#[rustfmt::skip]
const BUILTIN_CONSTANTS_JOINED: &[(&str, &str)] = &[
  ("e", "2.718281828459045"),
  ("π/pi", "3.141592653589793"),
  ("τ/tau", "6.283185307179586"),
  ("∞/inf/infinity/", "infinity"),
];

#[rustfmt::skip]
const BUILTIN_FUNCTION_NAMES:&[&str] = &[
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

#[rustfmt::skip]
const BUILTIN_FUNCTIONS_DESCRIPTIONS: &[(&str, &str)] = &[
	("if(bool_expr,true_expr,false_expr)", " If the bool_expr is true, then evaluate the true_expr, otherwise evaluate the false_expr.",),
  ("get(tuple,index)", " Get the value at the index from the tuple."),
	("", ""),
	("max(a, b)", " Returns the maximum of the two numbers."),
	("min(a, b)", " Returns the minimum of the two numbers."),
	("floor(a)", " Returns the largest integer less than or equal to a."),
	("round(a)", " Returns the nearest integer to a. If a value is half-way between two integers, round away from 0.0.",),
	("ceil(a)", "Returns the smallesst integer greater than or equal to a."),
	("signnum(a)", " Returns the sign of a."),
	("abs(a)", " Returns the absolute value of a."),
	("", ""),
	("ln(a)", " Compute the natural logarithm."),
	("ln2(a)", " Compute the logarithm base 2."),
	("ln10(a)", " Compute the logarithm base 10."),
	("log(a,base)", " Compute the logarithm to a certain base."),
	("exp(a)", " Exponentiate with base e."),
	("exp2(a)", " Exponentiate with base 2."),
	("pow(a,b)", " Compute the power of a to the exponent b."),
	("sqrt(a)", " Compute the square root."),
	("cbrt(a)", " Compute the cubic root."),
	("", ""),
	("cos(a)", " Compute the cosine."),
	("cosh(a)", " Compute the hyperbolic cosine."),
	("acos(a)", " Compute the arccosine."),
	("acosh(a)", " Compute the hyperbolic arccosine."),
	("sin(a)", " Compute the sine."),
	("sinh(a)", " Compute the hyperbolic sine."),
	("asin(a)", " Compute the arcsine."),
	("asinh(a)", " Compute the hyperbolic arcsine."),
	("tan(a)", " Compute the tangent."),
	("tanh(a)", " Compute the hyperbolic tangent."),
	("atan(a)", " Compute the arctangent."),
	("atanh(a)", " Compute the hyperbolic arctangent."),
	("atan2(a,b)", " Compute the four quadrant arctangent."),
	("", ""),
	("hypot(a,b)", " Compute the distance between the origin and a point (a,b) on the Euclidean plane."),

	("", ""),
  ("normaldist(a, mean, std_dev)", " Compute the probability density of normal distribution at a with mean(default 0) and standard deviation (default 1)."),

];
const AVAILABLE_OPERATORS: &[(&str, &str)] = &[
	("(...)", "Parentheses grouping"),
	("a + b", "Addition"),
	("a - b", "Subtraction"),
	("a * b", "Multiplication"),
	("a / b", "Division"),
	("a % b", "Modulo"),
	("a ^ b", "Power"),
	("-a", "Negation"),
	("f(param)", "Function call"),
	("f", "Also a function call (when f has no args)"),
	("f param", "Also a function call (when f has 1 arg)"),
];
const AVAILABLE_BOOLEAN_OPERATORS: &[(&str, &str)] = &[
	("a == b", "Equal to"),
	("a != b", "Not equal to"),
	("a < b", "Less than"),
	("a <= b", "Less than or equal to"),
	("a > b", "Greater than"),
	("a >= b", "Greater than or equal to"),
	("a && b", "Logical AND"),
	("a || b", "Logical OR"),
	("!a", "Logical NOT"),
];

pub fn show_builtin_information(ui: &mut egui::Ui) {
	ui.columns_const::<3, _>(|columns| {
		columns[0].heading("Operators");
		for &(name, value) in AVAILABLE_OPERATORS {
			columns[0].horizontal_wrapped(|ui| {
				ui.label(RichText::new(name).monospace().strong());
				ui.label(value);
			});
		}
		columns[1].heading("Boolean Operators");
		for &(name, value) in AVAILABLE_BOOLEAN_OPERATORS {
			columns[1].horizontal_wrapped(|ui| {
				ui.label(RichText::new(name).monospace().strong());
				ui.label(value);
			});
		}
		columns[2].heading("Builtin Constants");
		for &(name, value) in BUILTIN_CONSTANTS_JOINED {
			columns[2].horizontal_wrapped(|ui| {
				ui.label(RichText::new(name).monospace().strong());
				ui.label(value);
			});
		}
	});

	ui.separator();
	ui.heading("Builtin Functions");
	for &(name, value) in BUILTIN_FUNCTIONS_DESCRIPTIONS {
		if name.is_empty() {
			ui.separator();
		} else {
			ui.horizontal_wrapped(|ui| {
				ui.label(RichText::new(name).monospace().strong());
				ui.label(value);
			});
		}
	}
}

fn expect_function_argument_amount<NumericTypes: EvalexprFloat>(
	actual: usize, expected: usize,
) -> EvalexprResult<(), NumericTypes> {
	if actual == expected {
		Ok(())
	} else {
		Err(EvalexprError::wrong_function_argument_amount(actual, expected))
	}
}

pub fn init_builtins<T: EvalexprFloat>(ctx: &mut evalexpr::HashMapContext<T>) {
	for (name, value) in BUILTIN_CONSTANTS {
		ctx.set_value(istr(name), f64_to_value(*value)).unwrap();
	}

	ctx.set_function(
		istr("normaldist"),
		evalexpr::RustFunction::new(move |s, _| {
			let zero = T::from_f64(0.0);
			let one = T::from_f64(1.0);
			if s.num_args() == 0 {
				return Err(EvalexprError::wrong_function_argument_amount_range(0, 1..=3));
			}

			let (x, mean, std_dev) = if let Ok(x) = s.get_arg(0).unwrap().as_float() {
				(x, zero, one)
			} else {
				let x = s.get_arg(0).unwrap().as_float()?;
				let mean: T = s.get_arg(1).unwrap_or(&Value::Float(T::from_f64(0.0))).as_float()?;
				let std_dev: T = s.get_arg(2).unwrap_or(&Value::Float(T::from_f64(1.0))).as_float()?;
				(x, mean, std_dev)
			};

			let two = T::from_f64(2.0);
			let coefficient = T::from_f64(1.0) / (std_dev * T::from_f64(2.0 * core::f64::consts::PI).sqrt());
			let diff: T = x - mean;
			let exponent = -(diff.pow(&two)) / (T::from_f64(2.0) * std_dev.pow(&two));
			Ok(Value::Float(coefficient * exponent.exp()))
		}),
	);
	ctx.set_function(
		istr("get"),
		evalexpr::RustFunction::new(|s, _| {
			expect_function_argument_amount(s.num_args(), 2)?;

			let tuple = s.get_arg(0).unwrap();

			let index: T = s.get_arg(1).unwrap().as_float()?;

			let index = index.to_f64() as usize;
			let value = match tuple {
				Value::Tuple(t) => t
					.get(index)
					.ok_or_else(|| {
						EvalexprError::CustomMessage(format!(
							"Index out of bounds: index = {index} but the length was {}",
							t.len()
						))
					})
					.cloned()?,
				Value::Float2(f1, f2) => match index {
					0 => Value::Float(*f1),
					1 => Value::Float(*f2),
					_ => {
						return Err(EvalexprError::CustomMessage(format!(
							"Index out of bounds: index = {index} but the length was 2"
						)));
					},
				},
				_ => {
					return Err(EvalexprError::CustomMessage(format!("Expected a tuple but got {:?}", tuple)));
				},
			};

			Ok(value)
		}),
	);
	// };
}

static BUILTINS: LazyLock<(IStrSet, IStrSet, IStrSet)> = LazyLock::new(|| {
	let mut variables = IStrSet::default();
	let mut constants = IStrSet::default();
	let mut functions = IStrSet::default();

	for name in BUILTIN_VARIABLE_NAMES {
		variables.insert(istr(name));
	}
	for (name, _) in BUILTIN_CONSTANTS {
		constants.insert(istr(name));
	}
	for name in BUILTIN_FUNCTION_NAMES {
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
