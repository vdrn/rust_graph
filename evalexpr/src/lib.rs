//!
//! ## Quickstart
//!
//! Add `evalexpr` as dependency to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! evalexpr = "<desired version>"
//! ```
//!
//! Then you can use `evalexpr` to **evaluate expressions** 
//!

#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![allow(clippy::get_first)]
#![feature(vec_into_raw_parts)]

pub use crate::context::HashMapContext;
pub use crate::error::{EvalexprError, EvalexprResult};
pub use crate::flat_node::{optimize_flat_node, FlatNode, FlatOperator, Stack};
pub use crate::function::expression_function::ExpressionFunction;
pub use crate::function::rust_function::RustFunction;
pub use crate::interface::*;
pub use crate::token::PartialToken;
pub use crate::tree::{Node, Operator};
pub use crate::value::numeric_types::default_numeric_types::DefaultNumericTypes;
pub use crate::value::numeric_types::f32_numeric_types::F32NumericTypes;
pub use crate::value::numeric_types::EvalexprFloat;
pub use crate::value::value_type::ValueType;
pub use crate::value::{EmptyType, TupleType, Value, EMPTY_VALUE};
/// Interned string for identifiers
pub type IStr = istr::IStr;
pub(crate) type IStrMap<T> = istr::IStrMap<T>;
/// Interned string set
pub type IStrSet = istr::IStrSet;
/// Interned string constructor
pub fn istr(string: &str) -> IStr { istr::IStr::new(string) }
/// Empty interned string
pub fn istr_empty() -> IStr { istr::IStr::empty() }

pub use thin_vec::ThinVec;

// mod compiled_node;
mod context;
pub mod error;
#[cfg(feature = "serde")]
mod feature_serde;
mod flat_node;
mod function;
mod interface;
mod math;
mod token;
mod tree;
mod value;

