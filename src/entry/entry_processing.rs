use core::fmt::Write;

use evalexpr::{
	EvalexprFloat, ExpressionFunction, FlatNode, IStr, Node, Operator, Stack, ThinVec, Value, istr, istr_empty
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::builtins::is_builtin;
use crate::entry::entry_plot_elements::eval_point2;
use crate::entry::{DragPoint, Entry, EntryType, EquationType, FunctionType, PointDragType, PointsType};
use crate::scope;

pub fn preprocess_ast<T: EvalexprFloat>(mut ast: Node<T>) -> Result<(Node<T>, EquationType), String> {
	let mut equation_type = EquationType::None;
	fn walk_ast<T: EvalexprFloat>(
		ast: &mut Node<T>, equation_type: &mut EquationType, mut replace_writes: bool,
	) -> Result<(), String> {
		let op = ast.operator_mut();
		if *op == Operator::Assign {
			replace_writes = true;
		}
		match op {
			Operator::FunctionIdentifier { .. } => {
				// <=, <, >=, > inside function call params are not threated as equation types, but boolean
				// ops
				return Ok(());
			},
			Operator::Assign | Operator::Lt | Operator::Gt | Operator::Leq | Operator::Geq => {
				if *equation_type != EquationType::None {
					return Err("You cannot have multiple `=`, `<`, `>`, `<=` or `>=` operators in the same \
					            function."
						.to_string());
				}
				*equation_type = match op {
					Operator::Assign => EquationType::Equality,
					Operator::Lt => EquationType::LessThan,
					Operator::Gt => EquationType::GreaterThan,
					Operator::Leq => EquationType::LessThanOrEqual,
					Operator::Geq => EquationType::GreaterThanOrEqual,
					_ => unreachable!(),
				};
				*op = Operator::Sub;
			},
			Operator::VariableIdentifierWrite { identifier } => {
				if replace_writes {
					*op = Operator::VariableIdentifierRead { identifier: core::mem::take(identifier) };
				}
			},
			_ => {},
		}
		for child in ast.children_mut() {
			walk_ast(child, equation_type, replace_writes)?;
		}
		Ok(())
	}
	walk_ast(&mut ast, &mut equation_type, false)?;

	Ok((ast, equation_type))
}

pub fn prepare_entries<T: EvalexprFloat>(
	entries: &mut [Entry<T>], ctx: &mut evalexpr::HashMapContext<T>,
	prepare_erros: &mut FxHashMap<u64, String>,
) {
	for entry in entries.iter_mut() {
		match &mut entry.ty {
			EntryType::Folder { entries } => {
				prepare_entries(entries, ctx, prepare_erros);
			},
			_ => {
				if let Err(err) = prepare_entry(entry, ctx) {
					prepare_erros.insert(entry.id, err);
				} else {
					prepare_erros.remove(&entry.id);
				}
			},
		}
	}
}
fn prepare_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), String> {
	match &mut entry.ty {
		EntryType::Folder { .. } => {
			//handled in outer scope
		},
		EntryType::Points { points_ty, identifier, .. } => {
			*identifier = istr(entry.name.as_str().trim());

			if let Some(builtin_type) = is_builtin(*identifier) {
				*identifier = istr_empty();
				return Err(format!("Cannot use builtin {} as name: {}", builtin_type, identifier));
			}
			match points_ty {
				PointsType::Separate(points) => {
					for point in points {
						let (Some(x), Some(y)) = (&point.x.node, &point.y.node) else {
							point.drag.drag_point = None;

							return Ok(());
						};
						let x_state = analyze_node(x);
						let y_state = analyze_node(y);

						let both_dirs_available =
							x_state.constants.iter().all(|c| !y_state.constants.contains(c));
						if !both_dirs_available
							&& point.drag.both_drag_dirs_available
							&& !matches!(point.drag.drag_type, PointDragType::NoDrag)
						{
							point.drag.drag_type = PointDragType::X;
						}
						point.drag.both_drag_dirs_available = both_dirs_available;

						match point.drag.drag_type {
							PointDragType::NoDrag => {
								point.drag.drag_point = None;
							},
							PointDragType::Both => {
								if x_state.is_literal && y_state.is_literal {
									point.drag.drag_point = Some(DragPoint::BothCoordLiterals);
								} else if x_state.is_literal
									&& let Some(y_const) = y_state.constants.first()
								{
									point.drag.drag_point = Some(DragPoint::XLiteralYConstant(istr(y_const)));
								} else if y_state.is_literal
									&& let Some(x_const) = x_state.constants.first()
								{
									point.drag.drag_point = Some(DragPoint::YLiteralXConstant(istr(x_const)));
								} else if let (Some(x_const), Some(y_const)) =
									(x_state.constants.first(), y_state.constants.first())
								{
									// todo:
									if x_const == y_const {
										if let Some(y_const) = y_state.constants.get(1) {
											point.drag.drag_point = Some(DragPoint::BothCoordConstants(
												istr(x_const),
												istr(y_const),
											));
										} else if let Some(x_const) = x_state.constants.get(1) {
											point.drag.drag_point = Some(DragPoint::BothCoordConstants(
												istr(x_const),
												istr(y_const),
											));
										} else {
											point.drag.drag_point = Some(DragPoint::XConstant(istr(x_const)));
										}
									} else {
										point.drag.drag_point =
											Some(DragPoint::BothCoordConstants(istr(x_const), istr(y_const)));
									}
								} else if let Some(x_const) = x_state.constants.first()
								// && y_state.first_constant.is_none()
								{
									point.drag.drag_point = Some(DragPoint::XConstant(istr(x_const)));
								} else if let Some(y_const) = y_state.constants.first() {
									point.drag.drag_point = Some(DragPoint::YConstant(istr(y_const)));
								} else {
									point.drag.drag_point = None;
								}
							},
							PointDragType::X => {
								if x_state.is_literal {
									point.drag.drag_point = Some(DragPoint::XLiteral);
								} else if let Some(x_const) = x_state.constants.first() {
									if x_state.num_identifiers_and_special_ops == 1 {
										point.drag.drag_point = Some(DragPoint::XConstant(istr(x_const)));
									} else if y_state.constants.iter().any(|c| c == x_const) {
										point.drag.drag_point =
											Some(DragPoint::SameConstantBothCoords(istr(x_const)));
									} else {
										point.drag.drag_point = Some(DragPoint::XConstant(istr(x_const)));
									}
								}
							},
							PointDragType::Y => {
								if y_state.is_literal {
									point.drag.drag_point = Some(DragPoint::YLiteral);
								} else if let Some(y_const) = y_state.constants.first() {
									if y_state.num_identifiers_and_special_ops == 1 {
										point.drag.drag_point = Some(DragPoint::YConstant(istr(y_const)));
									} else if x_state.constants.iter().any(|c| c == y_const) {
										point.drag.drag_point =
											Some(DragPoint::SameConstantBothCoords(istr(y_const)));
									} else {
										point.drag.drag_point = Some(DragPoint::YConstant(istr(y_const)));
									}
								}
							},
						}
					}
				},
				PointsType::SingleExpr { expr: _, val: _ } => {},
			}
		},
		EntryType::Constant { .. } => {
			prepare_constant(entry, ctx)?;
		},
		EntryType::Function { func, identifier, ty, .. } => {
			if let Some(func_node) = &func.node {
				func.args.clear();

				let name_ast = evalexpr::build_ast::<T>(&entry.name).map_err(|e| e.to_string())?;
				// println!("name ast {:#?}", name_ast);

				let first_ast_node =
					if name_ast.children().is_empty() { &name_ast } else { &name_ast.children()[0] };

				// println!("first ast node {:#?}", first_ast_node);
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
							*identifier = istr_empty();
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
								return Err("Invalid function parameter name".to_string());
							}
						}
						*identifier = *function_ident;
					},
					_ => {
						if let Some(builtin_type) = is_builtin(*identifier) {
							let err = format!("Cannot use builtin {} as name: {}", builtin_type, identifier);
							*identifier = istr_empty();
							return Err(err);
						}
						*identifier = istr_empty();
						return Err("Invalid function name".to_string());
					},
				}
				if let Some(builtin_type) = is_builtin(*identifier) {
					let err = format!("Cannot use builtin {} as name: {}", builtin_type, identifier);
					*identifier = istr_empty();
					return Err(err);
				}
			}
		},
	}
	Ok(())
}

pub fn prepare_constants<T: EvalexprFloat>(
	entries: &mut [Entry<T>], ctx: &mut evalexpr::HashMapContext<T>,
	prepare_errors: &mut FxHashMap<u64, String>,
) {
	for entry in entries.iter_mut() {
		match &mut entry.ty {
			EntryType::Constant { .. } => {
				if let Err(e) = prepare_constant(entry, ctx) {
					prepare_errors.insert(entry.id, e);
				} else {
					prepare_errors.remove(&entry.id);
				}
			},
			EntryType::Folder { entries } => {
				prepare_constants(entries, ctx, prepare_errors);
			},
			_ => {},
		}
	}
}
fn prepare_constant<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), String> {
	let EntryType::Constant { value, istr_name, .. } = &mut entry.ty else {
		return Ok(());
	};

	*istr_name = istr(entry.name.as_str().trim());

	if let Some(builtin_type) = is_builtin(*istr_name) {
		let err = format!("Cannot use builtin {} as name: {}", builtin_type, *istr_name);
		*istr_name = istr_empty();
		return Err(err);
	} else if !entry.name.is_empty() {
		ctx.set_value(*istr_name, evalexpr::Value::<T>::Float(*value)).unwrap();
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
	optimization_errors: &mut FxHashMap<u64, String>,
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
			if let Some(node) = get_entry(root_entries, functions[i].root_idx, functions[i].idx) {
				match &node.ty {
					EntryType::Function { func, .. } => {
						if let Some(node) = func.node.as_ref() {
							for ident in node.iter_identifiers() {
								if let Some(d) = functions.iter().find(|e| e.identifier.to_str() == ident) {
									depends_on.push(d.identifier);
								}
							}
						}
					},
					EntryType::Points { points_ty, .. } => match points_ty {
						PointsType::Separate(points) => {
							for c_node in points
								.iter()
								.flat_map(|p| [p.x.node.as_ref(), p.y.node.as_ref()].into_iter())
								.flatten()
							{
								for ident in c_node.iter_identifiers() {
									if let Some(d) = functions.iter().find(|e| e.identifier.to_str() == ident)
									{
										depends_on.push(d.identifier);
									}
								}
							}
						},
						PointsType::SingleExpr { expr, val: _ } => {
							if let Some(node) = &expr.node {
								for ident in node.iter_identifiers() {
									if let Some(d) = functions.iter().find(|e| e.identifier.to_str() == ident)
									{
										depends_on.push(d.identifier);
									}
								}
							}
						},
					},
					_ => {
						unreachable!()
					},
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
					optimization_errors.insert(id, e);
				} else {
					optimization_errors.remove(&entry.id);
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
			// functions.sort_by_key(|e| (e.prio, e.depends_on.as_ref().map(|d| d.len()).unwrap_or(0)));
			functions.sort_by_key(|e| e.prio);
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
	prio:       u32,
}
fn add_entries<T: EvalexprFloat>(root_idx: Option<usize>, entries: &[Entry<T>], graph: &mut Vec<GraphEntry>) {
	use smallvec::smallvec;
	for (i, entry) in entries.iter().enumerate() {
		match &entry.ty {
			EntryType::Function { func, identifier, .. } => {
				let depends_on = if func.node.is_some() { Some(smallvec![]) } else { None };
				graph.push(GraphEntry { identifier: *identifier, root_idx, idx: i, depends_on, prio: 0 });
			},
			EntryType::Folder { entries } => {
				add_entries(Some(i), entries, graph);
			},
			EntryType::Points { identifier, .. } => {
				let depends_on = Some(smallvec![]);
				graph.push(GraphEntry { identifier: *identifier, root_idx, idx: i, depends_on, prio: 1 });
			},
			_ => {},
		}
	}
}
fn get_entry<T: EvalexprFloat>(
	root_entries: &[Entry<T>], root_idx: Option<usize>, idx: usize,
) -> Option<&Entry<T>> {
	let entry = if let Some(root_idx) = root_idx {
		let EntryType::Folder { entries } = &root_entries[root_idx].ty else {
			unreachable!();
		};
		&entries[idx]
	} else {
		&root_entries[idx]
	};
	match &entry.ty {
		EntryType::Function { .. } => Some(entry),
		EntryType::Points { .. } => Some(entry),
		// EntryType::Integral { func, .. } => func.node.as_ref(),
		_ => None,
	}
}

fn inline_and_fold_entry<T: EvalexprFloat>(
	entry: &mut Entry<T>, ctx: &mut evalexpr::HashMapContext<T>,
) -> Result<(), (u64, String)> {
	match &mut entry.ty {
		EntryType::Function { func, identifier, .. } => {
			func.expr_function = None;

			if ctx.has_value(*identifier) {
				let err = format!("Cannot use identifier {} as name: it is already defined.", identifier);
				*identifier = istr_empty();
				return Err((entry.id, err));
			}

			let Some(node) = &func.node else {
				return Ok(());
			};
			// println!("node: {identifier}: {:#?}", node);
			let inlined_node =
				evalexpr::optimize_flat_node(node, ctx).map_err(|e| (entry.id, e.to_string()))?;
			// println!("inlined node {identifier}: {:#?}", inlined_node);
			let expr_function = ExpressionFunction::new(inlined_node, &func.args, &mut Some(ctx))
				.map_err(|e| (entry.id, e.to_string()))?;
			// println!("expr_func node: {identifier} {:#?}", expr_function);

			if identifier.to_str() != "" && func.equation_type == EquationType::None {
				ctx.set_expression_function(*identifier, expr_function.clone());
			}
			func.expr_function = Some(expr_function);
		},
		EntryType::Points { identifier, points_ty, .. } => {
			let mut stack = Stack::<T>::with_capacity(0);
			let mut warning = None;
			match points_ty {
				PointsType::Separate(points) => {
					for p in points.iter_mut() {
						match eval_point2(&mut stack, ctx, p.x.node.as_ref(), p.y.node.as_ref()) {
							Ok(Some(v)) => {
								p.val = Some(v);
							},
							Ok(None) => {
								p.val = None;
							},
							Err(e) => {
								return Err((entry.id, e));
							},
						}
					}
				},
				PointsType::SingleExpr { expr, val } => {
					val.clear();
					if let Some(node) = &expr.node {
						let eval_result =
							node.eval_with_context(&mut stack, ctx).map_err(|e| (entry.id, e.to_string()))?;
						match eval_result {
							Value::Float(f) => {
								return Err((
									entry.id,
									format!("Expected a point or list of points, got a float: {}", f),
								));
							},
							Value::Boolean(b) => {
								return Err((
									entry.id,
									format!("Expected a point or list of points, got a boolean: {}", b),
								));
							},
							Value::Empty => {},
							Value::Float2(x, y) => {
								val.push((x, y));
							},
							Value::Tuple(thin_vec) => {
								let mut invalid_indices = String::new();
								for (i, v) in thin_vec.into_iter().enumerate() {
									match v {
										Value::Float2(x, y) => {
											val.push((x, y));
										},
										v => {
											if !invalid_indices.is_empty() {
												invalid_indices.push_str(", ");
											}
											write!(&mut invalid_indices, "{i}: {v}").unwrap();
											// invalid_indices.(&format!("{}: {}", i, v));
										},
									}
								}
								if !invalid_indices.is_empty() {
									warning = Some(format!("Invalid points: {}", invalid_indices));
								}
							},
						}
					}
				},
			}

			if identifier.to_str() != "" {
				if ctx.has_value(*identifier) {
					let err = format!("Cannot use identifier {} as name: it is already defined.", identifier);
					*identifier = istr_empty();
					return Err((entry.id, err));
				}

				match points_ty {
					PointsType::Separate(points) => {
						if points.len() == 1 {
							if let Some(va) = points[0].val {
								ctx.set_value(*identifier, Value::<T>::Float2(va.0, va.1)).unwrap();
							}
						} else if points.len() > 1 {
							let mut tuple = ThinVec::with_capacity(points.len());
							for p in points {
								if let Some(va) = p.val {
									tuple.push(Value::<T>::Float2(va.0, va.1));
								} else {
									tuple.push(Value::<T>::Empty);
								}
							}
							ctx.set_value(*identifier, Value::<T>::Tuple(tuple)).unwrap();
						}
					},
					PointsType::SingleExpr { expr: _, val } => {
						if val.len() == 1 {
							ctx.set_value(*identifier, Value::<T>::Float2(val[0].0, val[0].1)).unwrap();
						} else {
							ctx.set_value(
								*identifier,
								Value::<T>::Tuple(val.iter().map(|v| Value::<T>::Float2(v.0, v.1)).collect()),
							)
							.unwrap();
						}
					},
				}
			}
			if let Some(warning) = warning {
				return Err((entry.id, warning));
			}
		},
		_ => {},
	}
	Ok(())
}
