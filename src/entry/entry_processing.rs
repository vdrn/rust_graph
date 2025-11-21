use std::sync::LazyLock;

use evalexpr::{EvalexprFloat, ExpressionFunction, FlatNode, IStr, istr};
use regex::Regex;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::entry::{DragPoint, Entry, EntryType, FunctionType, PointDragType, RESERVED_NAMES};
use crate::scope;

pub fn preprecess_fn(text: &str) -> Result<Option<String>, String> {
	// regex to check if theres an y identifier (single y not surrounded by alphanumerics on either
	// side)
	static RE_Y: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])y(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_X: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])x(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_ZY: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])zy(?:$|[^a-zA-Z0-9])").unwrap());
	static RE_ZX: LazyLock<Regex> =
		LazyLock::new(|| Regex::new(r"(?:^|[^a-zA-Z0-9])zx(?:$|[^a-zA-Z0-9])").unwrap());

	let text_b = text.as_bytes();
	let mut split = None;
	for (i, &c) in text_b.iter().enumerate() {
		if c == b'=' {
			let prev = text_b.get(i - 1).copied();
			let next = text_b.get(i + 1).copied();
			static IGNORE_PREV: &[u8] = b"!=<>+-*/%^|&";
			if next != Some(b'=') && IGNORE_PREV.iter().all(|&c| Some(c) != prev) {
				split = Some((&text[0..i], text.get(i + 1..).unwrap_or("")));
				break;
			}
		}
	}

	let Some((left, right)) = split else {
		return Ok(None);
	};
	let left = left.trim();
	let right = right.trim();
	if left.is_empty() || right.is_empty() {
		return Err("Something is needed on both sides of the = sign".to_string());
	}
	if left == "y" {
		if RE_Y.is_match(right) || RE_ZY.is_match(right) || RE_ZX.is_match(right) {
			return Ok(Some(format!("y - ({right})")));
		}
		return Ok(Some(right.to_string()));
	}
	if left == "x" {
		if RE_X.is_match(right) || RE_ZY.is_match(right) || RE_ZX.is_match(right) {
			return Ok(Some(format!("x - ({right})")));
		}
		if RE_Y.is_match(right) {
			return Ok(Some(right.to_string()));
		}
		return Ok(Some(format!("{right} + y*0")));
	}
	let new = format!("{} - ({})", left, right);
	Ok(Some(new))
}

pub fn prepare_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), (u64, String)> {
	if RESERVED_NAMES.contains(&entry.name.trim()) {
		return Ok(());
	}
	match &mut entry.ty {
		EntryType::Folder { entries } => {
			for entry in entries {
				prepare_entry(entry, ctx)?;
			}
		},
		EntryType::Points { points, .. } => {
			for point in points {
				if let (Some(x), Some(y)) = (&point.x.node, &point.y.node) {
					let x_state = analyze_node(x);
					let y_state = analyze_node(y);

					let both_dirs_available = x_state.constants.iter().all(|c| !y_state.constants.contains(c));
					if !both_dirs_available
						&& point.both_drag_dirs_available
						&& !matches!(point.drag_type, PointDragType::NoDrag)
					{
						point.drag_type = PointDragType::X;
					}
					point.both_drag_dirs_available = both_dirs_available;

					match point.drag_type {
						PointDragType::NoDrag => {
							point.drag_point = None;
						},
						PointDragType::Both => {
							if x_state.is_literal && y_state.is_literal {
								point.drag_point = Some(DragPoint::BothCoordLiterals);
							} else if x_state.is_literal
								&& let Some(y_const) = y_state.constants.first()
							{
								point.drag_point = Some(DragPoint::XLiteralYConstant(istr(y_const)));
							} else if y_state.is_literal
								&& let Some(x_const) = x_state.constants.first()
							{
								point.drag_point = Some(DragPoint::YLiteralXConstant(istr(x_const)));
							} else if let (Some(x_const), Some(y_const)) =
								(x_state.constants.first(), y_state.constants.first())
							{
								// todo:
								if x_const == y_const {
									if let Some(y_const) = y_state.constants.get(1) {
										point.drag_point =
											Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
									} else if let Some(x_const) = x_state.constants.get(1) {
										point.drag_point =
											Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
									} else {
										point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
									}
								} else {
									point.drag_point =
										Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
								}
							} else if let Some(x_const) = x_state.constants.first()
							// && y_state.first_constant.is_none()
							{
								point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
							} else if let Some(y_const) = y_state.constants.first() {
								point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
							} else {
								point.drag_point = None;
							}
						},
						PointDragType::X => {
							if x_state.is_literal {
								println!("DP LIETRAL");
								point.drag_point = Some(DragPoint::XLiteral);
							} else if let Some(x_const) = x_state.constants.first() {
								if x_state.num_identifiers_and_special_ops == 1 {
									point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
								} else if y_state.constants.iter().any(|c| c == x_const) {
									point.drag_point = Some(DragPoint::SameConstantBothCoords(istr(x_const)));
								} else {
									point.drag_point = Some(DragPoint::XConstant(istr(x_const)));
								}
							}
						},
						PointDragType::Y => {
							if y_state.is_literal {
								point.drag_point = Some(DragPoint::YLiteral);
							} else if let Some(y_const) = y_state.constants.first() {
								if y_state.num_identifiers_and_special_ops == 1 {
									point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
								} else if x_state.constants.iter().any(|c| c == y_const) {
									point.drag_point = Some(DragPoint::SameConstantBothCoords(istr(y_const)));
								} else {
									point.drag_point = Some(DragPoint::YConstant(istr(y_const)));
								}
							}
						},
					}
				} else {
					point.drag_point = None;
				}
			}
		},
		EntryType::Label { .. } => {},
		EntryType::Constant { value, istr_name, .. } => {
			*istr_name = istr(entry.name.as_str());
			if !entry.name.is_empty() {
				ctx.set_value(*istr_name, evalexpr::Value::<T>::Float(*value)).unwrap();
			}
		},
		EntryType::Function { func, identifier, ty, .. } => {
			if let Some(func_node) = func.node.clone() {
				func.args.clear();

				let name_ast = evalexpr::build_ast::<T>(&entry.name).map_err(|e| (entry.id, e.to_string()))?;
				// println!("name ast {:#?}", name_ast);

				let first_ast_node =
					if name_ast.children().is_empty() { &name_ast } else { &name_ast.children()[0] };

				match first_ast_node.operator() {
					evalexpr::Operator::VariableIdentifierRead { .. } | evalexpr::Operator::RootNode => {
						*ty = FunctionType::Expression;
						let mut has_x = false;
						let mut has_y = false;
						// let mut has_complex = false;
						for ident in func_node.iter_identifiers() {
							if ident == "x" {
								has_x = true;
							} else if ident == "y" {
								has_y = true;
							}
						}
						if has_x {
							func.args.push(istr("x"));
						}
						if has_y {
							func.args.push(istr("y"));
						}
						if let evalexpr::Operator::VariableIdentifierRead { identifier: function_ident } =
							first_ast_node.operator()
						{
							*identifier = *function_ident;
						} else {
							*identifier = istr("");
						}
					},
					evalexpr::Operator::FunctionIdentifier { identifier: function_ident } => {
						*ty = FunctionType::WithCustomParams;
						for child in first_ast_node.children() {
							if let evalexpr::Operator::VariableIdentifierRead { identifier: arg_ident } =
								child.operator()
							{
								func.args.push(*arg_ident);
							} else {
								return Err((entry.id, "Invalid function parameter name".to_string()));
							}
						}
						*identifier = *function_ident;
					},
					_ => {
						*identifier = istr("");
						return Err((entry.id, "Invalid function name".to_string()));
					},
				}
			}
		},
	}
	Ok(())
}
struct NodeAnalysis<'a> {
	is_literal:                      bool,
	num_identifiers_and_special_ops: u32,
	constants:                       SmallVec<[&'a str; 6]>,
}
#[rustfmt::skip]
fn analyze_node<T: EvalexprFloat>(node: &FlatNode<T>) -> NodeAnalysis<'_> {
	let mut is_literal = true;
	let mut constants = SmallVec::new();
	let mut num_identifiers = 0;

	for i in node.iter_variable_identifiers() {
		constants.push(i);
		is_literal = false;
	}

	for _ in node.iter_identifiers() {
		is_literal = false;
		num_identifiers += 1;
	}

	for op in node.iter() {
		use evalexpr::FlatOperator as O;
		match op {
			O::Mod | O::Exp | O::Square | O::Cube | O::Sqrt |
      O::Cbrt | O::Abs | O::Floor | O::Round | O::Ceil | O::Ln | O::Log |
      O::Log2 | O::Log10 | O::ExpE | O::Exp2 | O::Cos | O::Acos | O::CosH |
      O::AcosH | O::Sin | O::Asin | O::SinH | O::AsinH | O::Tan | O::Atan |
      O::TanH | O::AtanH | O::Atan2 | O::Hypot | O::Signum | O::Min | O::Max |
      O::Clamp | O::Factorial | O::Gcd | O::Sum { .. } | O::Product { .. } | O::Integral(_) => {
				is_literal = false;
				num_identifiers += 1;
			},
			_ => {},
		}
	}

	NodeAnalysis { is_literal, constants, num_identifiers_and_special_ops: num_identifiers }
}
pub fn optimize_entries<T: EvalexprFloat>(
	root_entries: &mut [Entry<T>], ctx: &mut evalexpr::HashMapContext<T>,
	parsing_errors: &mut FxHashMap<u64, String>,
) {
	scope!("optimizing_pass");

	use smallvec::smallvec;
	let mut functions: Vec<GraphEntry> = Vec::with_capacity(root_entries.len());
	// collect all functions that need optimization
	add_entries(None, root_entries, &mut functions);

	// determine inter-function dependencies
	for i in 0..functions.len() {
		if functions[i].depends_on.is_some() {
			let mut depends_on = smallvec![];
			if let Some(node) = get_function_node(root_entries, functions[i].root_idx, functions[i].idx) {
				for ident in node.iter_identifiers() {
					if let Some(d) = functions.iter().find(|e| e.identifier.to_str() == ident) {
						depends_on.push(d.identifier);
					}
				}
			}
			functions[i].depends_on = Some(depends_on);
		}
	}

	while !functions.is_empty() {
		let mut optimized = 0;

		let mut i = 0;
		while i < functions.len() {
			if functions[i].depends_on.as_ref().is_none_or(|d| d.is_empty()) {
				let entry = if let Some(root_idx) = functions[i].root_idx {
					let EntryType::Folder { entries } = &mut root_entries[root_idx].ty else {
						unreachable!();
					};
					&mut entries[functions[i].idx]
				} else {
					&mut root_entries[functions[i].idx]
				};
				if let Err((id, e)) = inline_and_fold_entry(entry, ctx) {
					parsing_errors.insert(id, e);
				} else {
					parsing_errors.remove(&entry.id);
				}
				let ge = functions.swap_remove(i);

				for graph_entry in functions.iter_mut() {
					if let Some(depends_on) = graph_entry.depends_on.as_mut() {
						depends_on.retain(|d| *d != ge.identifier);
					}
				}

				optimized += 1;
			} else {
				i += 1;
			}
		}

		if optimized == 0 && !functions.is_empty() {
			// We have a cycle.
			// Choose one function to optimize without all its dependencies optimized.
			// In practice this only means its dependencies will not be inlined.
			// We choose the one with least number of dependencies.
			// NOTE: we might need to rethink this
			functions.sort_unstable_by_key(|e| e.depends_on.as_ref().map(|d| d.len()).unwrap_or(0));
			println!("Cycle detected. Force compiling first function: {functions:?}. ");
			functions[0].depends_on = None;
		}
	}
}

#[derive(Debug)]
struct GraphEntry {
	identifier: IStr,
	root_idx:   Option<usize>,
	idx:        usize,
	depends_on: Option<SmallVec<[IStr; 6]>>,
}
fn add_entries<T: EvalexprFloat>(root_idx: Option<usize>, entries: &[Entry<T>], graph: &mut Vec<GraphEntry>) {
	use smallvec::smallvec;
	for (i, entry) in entries.iter().enumerate() {
		match &entry.ty {
			EntryType::Function { func, identifier, .. } => {
				let depends_on = if func.node.is_some() { Some(smallvec![]) } else { None };
				graph.push(GraphEntry { identifier: *identifier, root_idx, idx: i, depends_on });
			},
			EntryType::Folder { entries } => {
				add_entries(Some(i), entries, graph);
			},
			_ => {},
		}
	}
}
fn get_function_node<T: EvalexprFloat>(
	root_entries: &[Entry<T>], root_idx: Option<usize>, idx: usize,
) -> Option<&FlatNode<T>> {
	let entry = if let Some(root_idx) = root_idx {
		let EntryType::Folder { entries } = &root_entries[root_idx].ty else {
			unreachable!();
		};
		&entries[idx]
	} else {
		&root_entries[idx]
	};
	match &entry.ty {
		EntryType::Function { func, .. } => func.node.as_ref(),
		// EntryType::Integral { func, .. } => func.node.as_ref(),
		_ => None,
	}
}

fn inline_and_fold_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), (u64, String)> {
	match &mut entry.ty {
		EntryType::Function { func, identifier, .. } => {
			let Some(node) = &func.node else { return Ok(()) };
			// println!("INLINING FUNC {} {}", entry.name, func.text);
			let inlined_node =
				evalexpr::optimize_flat_node(node, ctx).map_err(|e| (entry.id, e.to_string()))?;
			// println!("INLINED FUNC: {:#?}", inlined_node);

			// let thread_local_context = thread_local_context.clone();
			// let ty = *ty;
			let expr_function = ExpressionFunction::new(inlined_node, &func.args, &mut Some(ctx))
				.map_err(|e| (entry.id, e.to_string()))?;
			// println!("FUNC EXPRESSION {:#?}", expr_function);

			if identifier.to_str() != "" {
				ctx.set_expression_function(*identifier, expr_function.clone());
			}
			func.expr_function = Some(expr_function);
		},
		EntryType::Label { size, .. } => {
			let Some(node) = size.node.clone() else {
				return Ok(());
			};
			let expr_function =
				ExpressionFunction::new(node, &[istr("x"), istr("y")], &mut Some(ctx))
					.map_err(|e| (entry.id, e.to_string()))?;
			size.expr_function = Some(expr_function);
		},
		_ => {},
	}
	Ok(())
}
